import torch
import torch.nn as nn
from config_1 import CONSTANTS
from torch.utils.data import Dataset

class MSPELoss(nn.Module):
    """
    Implements the custom loss:
    a = (true - pred) / max(true, sigma)
    loss = mean(a^2)
    """
    def __init__(self, sigma=1.0):
        super(MSPELoss, self).__init__()
        
        # register_buffer makes 'sigma' part of the module's state
        # (like parameters), but it is NOT a trainable parameter.
        self.register_buffer('sigma', torch.tensor(sigma))

    def forward(self, pred, true):
        # 1. Calculate 'a'
        # torch.maximum is the element-wise max.
        denominator = torch.maximum(true, self.sigma)
        a = (true - pred) / denominator
        
        # 2. Calculate the error: sum(a^2) / N
        # sum(a^2) / N is just the Mean Squared Error of 'a'.
        loss = torch.mean(a ** 2)
        
        return loss

class SensorDataset(Dataset):
    """
    Custom PyTorch Dataset for creating time-series windows.
    This version handles data that has already been scaled.
    """
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
        self.num_samples = len(x_data)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x_tensor = torch.from_numpy(self.x_data[idx]).float()
        y_tensor = torch.from_numpy(self.y_data[idx]).float()
        return x_tensor, y_tensor

class GRUModel(nn.Module):
    """A GRU model built with PyTorch."""
    def __init__(self, params):
        super().__init__()
        self.output_steps = CONSTANTS["output_seconds"] * CONSTANTS["target_fs"]

        # Layer definitions
        self.gru1 = nn.GRU(
            input_size=CONSTANTS["NUM_FEATURES"],
            hidden_size=params["gru_units_layer1"],
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(params["dropout"])
        self.gru2 = nn.GRU(
            input_size=params["gru_units_layer1"],
            hidden_size=params["gru_units_layer2"],
            batch_first=True
        )
        self.dropout2 = nn.Dropout(params["dropout"])
        self.dense1 = nn.Linear(params["gru_units_layer2"], params["dense_units"])
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(params["dense_units"], CONSTANTS["output_seconds"] * CONSTANTS["target_fs"] * CONSTANTS["NUM_FEATURES"])

    def forward(self, x):
        x, _ = self.gru1(x)
        x = self.dropout1(x)
        x, _ = self.gru2(x)
        x = x[:, -1, :]
        x = self.dropout2(x)

        x = self.dense1(x)
        x = self.relu(x)
        x = self.output_layer(x)

        outputs = x.view(-1, self.output_steps, CONSTANTS["NUM_FEATURES"])
        return outputs

class LSTMModel(nn.Module):
    """An LSTM model built with PyTorch."""
    def __init__(self, params):
        super().__init__()
        self.output_steps = CONSTANTS["output_seconds"] * CONSTANTS["target_fs"]

        # Layer definitions
        self.lstm1 = nn.LSTM(
            input_size=CONSTANTS["NUM_FEATURES"],
            hidden_size=params["lstm_units_layer1"],
            batch_first=True
        )
        self.dropout1 = nn.Dropout(params["dropout"])
        self.lstm2 = nn.LSTM(
            input_size=params["lstm_units_layer1"],
            hidden_size=params["lstm_units_layer2"],
            batch_first=True
        )
        self.dropout2 = nn.Dropout(params["dropout"])
        self.dense1 = nn.Linear(params["lstm_units_layer2"], params["dense_units"])
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(params["dense_units"], CONSTANTS["output_seconds"] * CONSTANTS["target_fs"] * CONSTANTS["NUM_FEATURES"])

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]
        x = self.dropout2(x)

        x = self.dense1(x)
        x = self.relu(x)
        x = self.output_layer(x)

        outputs = x.view(-1, self.output_steps, CONSTANTS["NUM_FEATURES"])
        return outputs