import math
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from torch.utils.data import DataLoader

from preprocessing_3 import prepare_data, create_windows
from model_2 import GRUModel,LSTMModel, MSPELoss, SensorDataset
from config_1 import CONSTANTS,DATA,GRU_PARAMS,LSTM_PARAMS

input_seconds = CONSTANTS["input_seconds"]
output_seconds = CONSTANTS["output_seconds"]
target_fs = CONSTANTS["target_fs"]
input_steps = CONSTANTS["input_seconds"] * CONSTANTS["target_fs"]
output_steps = CONSTANTS["output_seconds"] * CONSTANTS["target_fs"]
batch_size = CONSTANTS["BATCH_SIZE"]
scaler_type = DATA["scaler_type"]
filter_type = DATA["filter_type"]
train_date = DATA["train_date"]
test_date = DATA["test_date"]
test_skip_seconds = DATA["test_skip_seconds"]

def evaluate_model(model_name):
    """
    Loads a trained PyTorch model, evaluates it on the test set, and
    visualizes its predictions after correct inverse scaling.
    """
    # --- 1. Configuration and Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model_filepath = os.path.join("saved_models", f"best_{model_name}_model.pt")
    if not os.path.exists(model_filepath):
        raise FileNotFoundError(f"Model file not found at {model_filepath}. Please train the model first.")
    
    # --- 2. Instantiate and Load the Model ---
    print(f"Loading model: {model_filepath}")
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
    #print(model)

    # --- 3. Prepare Test Data and Scalers ---
    print("Preparing test data loader and scalers...")
    loaders, _, _ = prepare_data(plot_type = False,filter_type=filter_type)
    test_scaled = loaders[2]
    l1=len(test_scaled)
    # print('Shape before skipping first 20 seconds:',test_scaled.shape)
    #remove first 20 seconds
    removal = test_skip_seconds*target_fs
    test_scaled = test_scaled[removal:]
    # print('Shape after skipping first 20 seconds:',test_scaled.shape)
    l2=len(test_scaled)
    removal_seconds=(l1-l2)/target_fs
    # print('Calculated removal seconds:',removal_seconds)

    x_test, y_test = create_windows(test_scaled, input_steps, output_steps)
    test_dataset = SensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    print("Data preparation complete.")

    # --- 4. Evaluate the Model ---
    print("Evaluating model on the test set...")

    all_labels = torch.cat([labels for _, labels in test_loader])
    labels_std = torch.std(all_labels.float()).item() #By default, torch.std uses Bessel’s correction (unbiased=True), which means it divides by N-1.
    labels_range = (torch.max(all_labels) - torch.min(all_labels)).item()
    count_zeros = (all_labels == 0).sum().item()
    epsilon = 1e-3
    epsilon1 = 1e-4
    epsilon2 = 1e-5
    count_near_zero = (torch.abs(all_labels) < epsilon).sum().item()
    count_near_zero1 = (torch.abs(all_labels) < epsilon1).sum().item()
    count_near_zero2 = (torch.abs(all_labels) < epsilon2).sum().item()

    # criterion1 = MSPELoss(labels_std)
    criterion2 = nn.MSELoss()
    # total_loss1 = 0.0
    total_loss2 = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            # loss1 = criterion1(outputs, labels)
            # total_loss1 += loss1.item() * inputs.size(0)
            loss2 = criterion2(outputs, labels)
            total_loss2 += loss2.item() * inputs.size(0)
            
    # test_mspe = total_loss1 / len(test_loader.dataset)
    # test_rmspe = np.sqrt(test_mspe)
    test_mse = total_loss2 / len(test_loader.dataset)
    test_rmse = np.sqrt(test_mse)
    print("\n--- Test Set Evaluation Results ---")
    print(f'{model_name}: {train_date} (train) -> {test_date} (test)')
    print(f"{target_fs} Hz,{input_seconds}:{output_seconds},{scaler_type},{filter_type}")
    print("-----------------------------")
    # print("Numbers of labels with exact value 0:", count_zeros)
    # print(f"Number of labels near 0 (within {epsilon}): {count_near_zero}")
    # print(f"Number of labels near 0 (within {epsilon1}): {count_near_zero1}")
    # print(f"Number of labels near 0 (within {epsilon2}): {count_near_zero2}")
    # print("-----------------------------")
    # print("Total Test MSPE", np.round(total_loss1,4))
    print("Total Test MSE", np.round(total_loss2,4))
    print("Number of samples in test dataset:", len(test_loader.dataset))
    print("-----------------------------")
    print("all_labels shape:", all_labels.shape)
    print("max label value:", np.round(torch.max(all_labels).item(), 4))
    print("min label value:", np.round(torch.min(all_labels).item(), 4))
    print("label range:", np.round(labels_range, 4))
    print("label sample std deviation:", np.round(labels_std, 4))
    print("-----------------------------")
    # print(f"Test MSPE: {test_mspe:.6f}")
    # print(f"Test RMSPE: {test_rmspe:.6f}")
    print(f"Test MSE: {test_mse:.6f}")
    print(f"Test RMSE: {test_rmse:.6f}")
    print("-----------------------------")

if __name__ == '__main__':
    MODEL_TO_EVALUATE = 'lstm' #models: 'gru','lstm'
    evaluate_model(model_name=MODEL_TO_EVALUATE) 