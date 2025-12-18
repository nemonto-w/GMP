import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import pandas as pd

from preprocessing_3 import prepare_data, select_data
from scipy.signal import butter, sosfiltfilt
from model_2 import GRUModel,LSTMModel
from config_1 import CONSTANTS,DATA,GRU_PARAMS,LSTM_PARAMS

# training_type = DATA["training_type"]   
fs = CONSTANTS["fs"]
target_fs = CONSTANTS["target_fs"]
input_steps = CONSTANTS["input_seconds"] * CONSTANTS["target_fs"]
output_steps = CONSTANTS["output_seconds"] * CONSTANTS["target_fs"]
downsample_factor = fs // target_fs
batch_size = CONSTANTS["BATCH_SIZE"]
train_date = DATA["train_date"]
test_date = DATA["test_date"]
scaler_type = DATA["scaler_type"]
filter_type = DATA["filter_type"]

def get_target_signal(output_type):

    _,_, test_raw = select_data()
    test_ds = test_raw[::downsample_factor]

    if output_type == True:
        # print("len(test_raw):", len(test_raw))
        # print("len(test_filtered):", len(test_filtered))
        # print("len(test_ds):", len(test_ds))
        # print("-----------------------------")
        print("Test Raw Shape:", test_raw.shape)
        # print("Test Filtered Shape:", test_filtered.shape)
        print("Test Downsampled Shape:", test_ds.shape)
        # print("-----------------------------")
        # print("Original:", test_raw[:5])
        # print("Filtered:", test_filtered[:5])
        # print("Downsampled:", test_ds[:5])

    return test_ds

def save_target_signal(model_name,test_ds):
    #filepath1 = Path(f'data/{model_name} Train_{train_date}_{scaler_type}_{filter_type}/TestData_{test_date}/target_{test_date}_raw.txt')

    filepath = Path(f'data/{model_name} Train_{train_date}_{scaler_type}_{filter_type}/TestData_{test_date}/target_{test_date}_ds.txt')
    filepath.parent.mkdir(parents=True, exist_ok=True)
    # np.savetxt(filepath1, test_raw)
    np.savetxt(filepath, test_ds)
    print(f"Target signal save to {filepath}")

def get_predicted_signal(model_name, output_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_filepath = os.path.join("saved_models", f"best_{model_name}_model.pt")
    if not os.path.exists(model_filepath):
        raise FileNotFoundError(f"Model file not found at {model_filepath}. Please train the model first.")
    # 1. Instantiate and Load the Model ---
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
    _, scalers, test_sets = prepare_data(plot_type=False,filter_type=filter_type)
    test_signal_ds = test_sets[1]
    X_test_full = test_sets[2]
    all_predictions_list = []

    with torch.no_grad():
        for i in range(0, len(X_test_full), batch_size):
            batch_X = X_test_full[i:i + batch_size].to(device)
            batch_pred_scaled = model(batch_X)
            all_predictions_list.append(batch_pred_scaled.cpu())

    all_predictions_scaled = torch.cat(all_predictions_list, dim=0)
    all_predictions_scaled_np = all_predictions_scaled.numpy()
    
    # 2. Inverse-transform all predictions
    # print("Inverse-transforming all predictions...")
    scaler_s1 = scalers['sensor1']
    scaler_s2 = scalers['sensor2']
    
    num_samples, output_steps, num_features = all_predictions_scaled_np.shape
    pred_s1_flat = all_predictions_scaled_np[:, :, 0].reshape(-1, 1)
    pred_s2_flat = all_predictions_scaled_np[:, :, 1].reshape(-1, 1)
    
    pred_s1_rescaled_flat = scaler_s1.inverse_transform(pred_s1_flat)
    pred_s2_rescaled_flat = scaler_s2.inverse_transform(pred_s2_flat)
    
    all_predictions_rescaled = np.stack([
        pred_s1_rescaled_flat.flatten(), 
        pred_s2_rescaled_flat.flatten()
    ], axis=-1).reshape(num_samples, output_steps, num_features)

    # 3. Stitch predictions into a continuous signal
    full_pre = np.full_like(test_signal_ds, np.nan)
    for i in range(num_samples):
        start_index = i + input_steps
        if start_index < len(full_pre):
            full_pre[start_index, :] = all_predictions_rescaled[i, 0, :]
    
    if output_type == True:
        # print("Prediction Length:", len(full_pre))
        print("Prediction Shape:", full_pre.shape)
        # print("Prediction (first 5 samples):", full_pre[:5])
    
    return full_pre

def save_predicted_signal(model_name, pre_signal):
    filepath = Path(f'data/{model_name} Train_{train_date}_{scaler_type}_{filter_type}/TestData_{test_date}/full_pre_{test_date}_ds.txt')
    np.savetxt(filepath, pre_signal)
    print(f"Predicted signal saved to '{filepath}'")

def save_combined_signals(model_name, test_ds, pre_signal,output_type):
    time_axis = np.arange(len(test_ds)) / target_fs
    combined_data = np.concatenate((test_ds, pre_signal), axis=1)
    combined_data = np.column_stack((time_axis, combined_data))
    if output_type == True:
        print("Combined Data Shape:", combined_data.shape)
        # print("Combined Data (first 5 samples):", combined_data[:5])
    filepath = Path(f'data/{model_name} Train_{train_date}_{scaler_type}_{filter_type}/TestData_{test_date}/combined_results_{test_date}_ds.txt')
    np.savetxt(filepath, combined_data)
    print(f"Combined signals saved to '{filepath}'")

if __name__ == "__main__":

    model_name = 'lstm'  # 'gru' or 'lstm'
    test_ds = get_target_signal(output_type=True)
    save_target_signal(model_name,test_ds)
    pre_signal = get_predicted_signal(model_name, output_type=True)
    save_predicted_signal(model_name, pre_signal)
    save_combined_signals(model_name, test_ds, pre_signal, output_type=True)
