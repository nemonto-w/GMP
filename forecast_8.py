import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
from pathlib import Path
from preprocessing_3 import prepare_data
from model_2 import GRUModel,LSTMModel, SensorDataset
from config_1 import CONSTANTS,DATA,GRU_PARAMS,LSTM_PARAMS
from scipy.signal import butter, sosfiltfilt

input_seconds = CONSTANTS["input_seconds"]
output_seconds = CONSTANTS["output_seconds"]
fs = CONSTANTS["fs"]
target_fs = CONSTANTS["target_fs"]
downsample_factor = fs // target_fs
input_steps = CONSTANTS["input_seconds"] * CONSTANTS["target_fs"]
output_steps = CONSTANTS["output_seconds"] * CONSTANTS["target_fs"]
train_date = DATA["train_date"]
test_date = DATA["test_date"]
scaler_type = DATA["scaler_type"]
filter_type = DATA["filter_type"]
test_skip_seconds = DATA["test_skip_seconds"]

def forecasting(
    input_10sec_s1: np.ndarray, 
    input_10sec_s2: np.ndarray, 
    model: torch.nn.Module, 
    device: torch.device, 
    scaler1: MinMaxScaler, 
    scaler2: MinMaxScaler
) -> np.ndarray:
    """
    Takes two separate 10-second (640-sample) RAW NumPy arrays (one for each sensor),
    handles all scaling, runs the model, and returns the 1-second (64-sample) 
    RESCALED prediction as a single (64, 2) array.

    Args:
        input_10sec_s1 (np.ndarray): Sensor 1 input data, expected shape (640,) or (640, 1)
        input_10sec_s2 (np.ndarray): Sensor 2 input data, expected shape (640,) or (640, 1)
        model (torch.nn.Module): Your trained PyTorch model.
        device (torch.device): The device to run the model on (e.g., 'cpu' or 'cuda')
        scaler1 (MinMaxScaler): The *fitted* scaler for the first feature/channel.
        scaler2 (MinMaxScaler): The *fitted* scaler for the second feature/channel.
        
    Returns:
        np.ndarray: The model's rescaled prediction, shape (64, 2)
    """

    # --- 1. Validate and Reshape Inputs ---

    if filter_type == 'filt':
        # Apply filtering and downsampling
        sos = butter(4, [0.01, 0.6], btype='band', output='sos', fs=fs)
        s1_raw_filt = sosfiltfilt(sos, input_10sec_s1, axis=0)
        s2_raw_filt = sosfiltfilt(sos, input_10sec_s1, axis=0)
        s1_raw_ds = s1_raw_filt[::downsample_factor]
        s2_raw_ds = s2_raw_filt[::downsample_factor]
    elif filter_type == 'no_filt':
        # No filtering, only downsampling
        s1_raw_ds = input_10sec_s1[::downsample_factor]
        s2_raw_ds = input_10sec_s1[::downsample_factor]
    
    data = np.stack([s1_raw_ds, s2_raw_ds], axis=-1)

    # --- 2. Scale the Raw Input Data ---
    s1_scaled_flat = scaler1.transform(data[:, 0].reshape(-1, 1))
    s2_scaled_flat = scaler2.transform(data[:, 1].reshape(-1, 1))
    
    # --- 3. Combine for Model Input ---
    # Stack the two (640, 1) arrays horizontally to create the (640, 2) array
    scaled_input_array = np.hstack([s1_scaled_flat, s2_scaled_flat])
    
    # --- 4. Run the Model Prediction (on scaled data) ---
    model.eval()
    with torch.no_grad():
        # Shape: (640, 2) -> (1, 640, 2)
        input_tensor = torch.from_numpy(scaled_input_array).float().unsqueeze(0).to(device)
        
        # Output shape: (1, 64, 2)
        scaled_prediction_tensor = model(input_tensor)
        
        # Shape: (1, 64, 2) -> (64, 2)
        scaled_output_array = scaled_prediction_tensor.cpu().squeeze(0).numpy()

    # --- 5. Inverse-Scale the Output Prediction ---
    # Split the (64, 2) output into two (64, 1) arrays
    pred_s1_scaled_flat = scaled_output_array[:, 0].reshape(-1, 1)
    pred_s2_scaled_flat = scaled_output_array[:, 1].reshape(-1, 1)
    
    # Apply inverse-transform to get real-world values
    pred_s1_rescaled = scaler1.inverse_transform(pred_s1_scaled_flat)
    pred_s2_rescaled = scaler2.inverse_transform(pred_s2_scaled_flat)
    
    # Recombine into the final (64, 2) output array
    final_prediction_rescaled = np.hstack([pred_s1_rescaled, pred_s2_rescaled])
        
    return final_prediction_rescaled

def execute_forecasting(model_name,raw_input_s1, raw_input_s2,input_start_index,input_end_index):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_filepath = os.path.join("saved_models", f"best_{model_name}_model.pt")
    if not os.path.exists(model_filepath):
        raise FileNotFoundError(f"Model file not found at {model_filepath}. Please train the model first.")
    # print(f"Loading model: {model_filepath}")
    if model_name == 'gru':
        model = GRUModel(params=GRU_PARAMS)
    elif model_name == 'lstm':
        model = LSTMModel(params=LSTM_PARAMS)
    else:
        # Add other model instantiations here if needed
        raise ValueError("Unknown model_name specified.")
    
    model.load_state_dict(torch.load(model_filepath, map_location=device))
    model.to(device)
    model.eval()
    _, scalers, _ = prepare_data(plot_type=False, filter_type=filter_type)
    scaler_s1 = scalers['sensor1']
    scaler_s2 = scalers['sensor2']

    # Give it two RAW 1D arrays, get ONE 2D array back
    raw_prediction = forecasting(
        raw_input_s1, 
        raw_input_s2,
        model, 
        device, 
        scaler_s1, 
        scaler_s2
    )
    
    print(f"Successfully generated forecast!")
    print(f"Input S1 shape: {raw_input_s1.shape}")
    print(f"Input S2 shape: {raw_input_s2.shape}")
    print(f"Input sample index:{input_start_index}:{input_end_index}")
    print(f"Output prediction shape: {raw_prediction.shape}") # Should be (64, 2)
    filepath = Path(f'data/predictions/{test_date}_one_sec_pre.txt')
    filepath.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(filepath, raw_prediction)
    return raw_prediction

#test forecasting function
if __name__ == '__main__':

    #change raw data path here
    data_path1 = f"data/data_1105_00_06/s1_{test_date}.txt"
    data_path2 = f"data/data_1105_00_06/s2_{test_date}.txt"
    #load target data for comparison
    data_path3 = f"data/lstm Train_1105_00_multi_scaler_filt/TestData_1105_02/target_1105_02_ds.txt" 

    data1 = np.loadtxt(data_path1)
    data2 = np.loadtxt(data_path2)
    data3 = np.loadtxt(data_path3)

    removal = test_skip_seconds*target_fs
    #skip first 20 seconds
    input_start = removal + 100*64
    input_end = input_start+input_steps
    input_10sec_s1 = data1[input_start:input_end]
    input_10sec_s2 = data2[input_start:input_end]

    model_name ='lstm'
    #execute forecasting, get raw prediction
    raw_prediction = execute_forecasting(model_name,input_10sec_s1, input_10sec_s2,input_start,input_end)
    #get corresponding target values for comparison
    output_target = data3[input_end:input_end+output_steps]

    print("\nFirst 5 predicted values [Sensor1, Sensor2]:")
    print(raw_prediction[:5])
    print("\nCorresponding 5 target values [Sensor1, Sensor2]:")
    print(output_target[:5])
