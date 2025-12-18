import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

from model_2 import GRUModel,LSTMModel
from config_1 import CONSTANTS,DATA,GRU_PARAMS,LSTM_PARAMS
from preprocessing_3 import prepare_data

EPOCHS = CONSTANTS["EPOCHS"]
input_seconds = CONSTANTS["input_seconds"]
output_seconds = CONSTANTS["output_seconds"]
target_fs = CONSTANTS["target_fs"]
BATCH_SIZE = CONSTANTS["BATCH_SIZE"]
PATIENCE = CONSTANTS["PATIENCE"]
INPUT_STEPS = input_seconds * target_fs
OUTPUT_STEPS = output_seconds * target_fs
scaler_type = DATA["scaler_type"]
filter_type = DATA["filter_type"]
train_date = DATA["train_date"]
test_date = DATA["test_date"]
# training_type = DATA["training_type"]   

def train_model(modelname):

    # Set device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define the directory and filepath for saving the best model
    os.makedirs("saved_models", exist_ok=True)
    model_filepath = os.path.join("saved_models", f"best_{modelname}_model.pt")

    # --- Prepare Data ---
    print("Preparing data loaders and scalers...")
    print("Scaled on training dataset only." if scaler_type == 'single_scaler' else "Scaled on each dataset individually.")
    loaders, _, _ = prepare_data (plot_type = False,filter_type=filter_type)
    train_loader, val_loader, _ = loaders

    # --- Build the Model, Loss Function, and Optimizer ---
    print(f"Building PyTorch {modelname} model...")
    # Instantiate the model from models.py and move it to the selected device

    if modelname == 'gru':
        model = GRUModel(params=GRU_PARAMS)
        #when fine-tuning a pre-trained model, uncomment the next line
        #model.load_state_dict(torch.load(model_filepath, map_location=device))
        model.to(device)
    elif modelname == 'lstm':
        model = LSTMModel(params=LSTM_PARAMS)
        #when fine-tuning a pre-trained model, uncomment the next line
        #model.load_state_dict(torch.load(model_filepath, map_location=device))
        model.to(device)
    else:
        raise ValueError("Unknown model_name specified.")
    #print(model)

    # Define the loss function
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONSTANTS["LEARNING_RATE"])

    # --- Training Loop with Checkpointing and Early Stopping ---
    # Variables for tracking training progress and implementing callbacks
    history = {
        'loss': [], 'val_loss': [],
        'rmse': [], 'val_rmse': []
    }
    best_val_loss = float('inf')
    epochs_no_improve = 0

    print(f"{target_fs} Hz, {input_seconds}:{output_seconds}")
    print(f"Training on data set: {train_date}")
    for epoch in range(EPOCHS):
        # -- Training Phase --
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_rmse = np.sqrt(epoch_loss)
        history['loss'].append(epoch_loss)
        history['rmse'].append(epoch_rmse)

        # -- Validation Phase --
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad(): # No need to calculate gradients for validation
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)

        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_rmse = np.sqrt(val_epoch_loss)
        history['val_loss'].append(val_epoch_loss)
        history['val_rmse'].append(val_epoch_rmse)

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Loss: {epoch_loss:.6f} | RMSE: {epoch_rmse:.6f} | "
              f"Val Loss: {val_epoch_loss:.6f} | Val RMSE: {val_epoch_rmse:.6f}")

        # -- Checkpointing and Early Stopping Logic --
        # Equivalent to ModelCheckpoint(save_best_only=True)
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save(model.state_dict(), model_filepath)
            print(f"Validation loss improved. Saving model to {model_filepath}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"Validation loss did not improve. Early stopping counter: {epochs_no_improve}/{PATIENCE}")

        # Equivalent to EarlyStopping callback
        if epochs_no_improve >= PATIENCE:
            print("Early stopping triggered.")
            break

    print("Training finished.")

    # --- 5. Plot and Save Training History ---
    plot_training_history(history, modelname)

def plot_training_history(history, modelname):
    """
    Plots the training and validation loss and RMSE from the model's history.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f'Training "{modelname.upper()}" with signal {train_date}\n'
                 f'{target_fs} Hz,{input_seconds}:{output_seconds},{scaler_type},{filter_type}', fontsize=20)
    # Plotting training & validation loss from the history dictionary 
    ax1.plot(history['loss'], label='Training Loss (MSE)')
    ax1.plot(history['val_loss'], label='Validation Loss (MSE)')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # Plotting training & validation RMSE from the history dictionary
    ax2.plot(history['rmse'], label='Training RMSE')
    ax2.plot(history['val_rmse'], label='Validation RMSE')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Root Mean Squared Error')
    ax2.set_title('Training and Validation RMSE')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    # Save the figure to a file
    filepath = Path(f"plots/{modelname} Train_{train_date}_{scaler_type}_{filter_type}/{modelname}_training.png")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filepath)
    print(f"Training history plot saved as '{filepath}'")
    #plt.show()

if __name__ == '__main__':
    model_to_train = 'lstm' #models: 'gru','lstm'
    train_model(model_to_train)