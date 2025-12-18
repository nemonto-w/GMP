import torch
from torch.utils.data import DataLoader
import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from pathlib import Path
import math

from config_1 import CONSTANTS, DATA
from model_2 import SensorDataset

fs = CONSTANTS["fs"]
target_fs = CONSTANTS["target_fs"]
downsample_factor = fs // CONSTANTS["target_fs"]
input_steps = CONSTANTS["input_seconds"] * CONSTANTS["target_fs"]
output_steps = CONSTANTS["output_seconds"] * CONSTANTS["target_fs"]
batch_size = CONSTANTS["BATCH_SIZE"]
# training_type = DATA["training_type"]   
train_date = DATA["train_date"]
scaler_type = DATA["scaler_type"]
filter_type = DATA["filter_type"]

def load_data(str):
    signal_1 = np.loadtxt(DATA["path1"].format(date=str))
    signal_2 = np.loadtxt(DATA["path2"].format(date=str))
    signal = np.stack([signal_1, signal_2], axis=-1)
    return signal

def select_data():
    # load txt file based on date, see config.py DATA dictionary
    # the input of load_data() function need to be changed when using new data
    # if type == 1:
    #     train_signal_raw = load_data('1105_00')
    #     test_signal_raw = load_data('1105_01')
    
    train_signal_raw = load_data('1105_04')
    test_signal_raw = load_data('1105_05')


    # elif type == 2:
    #     train_signal_raw = np.concatenate([load_data('00'), load_data('01'),load_data('02')], axis=0)
    #     test_signal_raw = load_data('03')

    # elif type == 3:
    #     train_signal_raw = np.concatenate([load_data('0915'), load_data('0916')], axis=0)
    #     test_signal_raw = load_data('0917')

    train_size = int(len(train_signal_raw) * 0.8)
    val_size = int(len(train_signal_raw) * 0.2)
    train_raw = train_signal_raw[:train_size]
    val_raw = train_signal_raw[train_size: train_size + val_size]
    test_raw = test_signal_raw

    return train_raw, val_raw, test_raw

def fit_data(data):
    scaler_s1 = MinMaxScaler(feature_range=(-1, 1))
    scaler_s2 = MinMaxScaler(feature_range=(-1, 1))
    scaler_s1.fit(data[:, 0].reshape(-1, 1))
    scaler_s2.fit(data[:, 1].reshape(-1, 1))

    return (scaler_s1, scaler_s2)

def transform_data(data, scaler_s1, scaler_s2):
    scaled_data = np.hstack([
        scaler_s1.transform(data[:, 0].reshape(-1, 1)),
        scaler_s2.transform(data[:, 1].reshape(-1, 1))
    ])
    return scaled_data

def create_windows(data, input_steps, output_steps):

    X, Y = [], []
    for i in range(len(data) - input_steps - output_steps + 1):
        X.append(data[i:(i + input_steps), :])
        Y.append(data[(i + input_steps):(i + input_steps + output_steps), :])
    return np.array(X), np.array(Y)

def plot_pre(raw_signal, ds_signal, scaled_signal, plot_duration_seconds=None):
    """
    Plots the raw, downsampled, and scaled signals for a specified duration.
    """

    if plot_duration_seconds is not None:
        # Calculate the number of samples for the given duration
        num_samples_raw = int(plot_duration_seconds * fs)
        num_samples_ds = int(plot_duration_seconds * target_fs)
        num_samples_scaled = int(plot_duration_seconds * target_fs)
        
        # Slice all signal arrays to the calculated length
        raw_signal = raw_signal[:num_samples_raw]
        ds_signal = ds_signal[:num_samples_ds]
        scaled_signal = scaled_signal[:num_samples_scaled]

    # Create a time axis in seconds for each (potentially sliced) signal
    time_raw = np.arange(raw_signal.shape[0]) / fs
    time_ds = np.arange(ds_signal.shape[0]) / target_fs
    time_scaled = np.arange(scaled_signal.shape[0]) / target_fs

    # Create 3 subplots that share the same x-axis
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    fig.suptitle(f'signal_{train_date}_{scaler_type}_{filter_type}', fontsize=16)

    ax1.plot(time_raw, raw_signal[:, 0], label='Raw Sensor 1', color='gray', alpha=0.8)
    ax1.plot(time_raw, raw_signal[:, 1], label='Raw Sensor 2', color='gray', alpha=0.8)
    ax1.set_title(f'1. Raw Signal Data (Showing {plot_duration_seconds} seconds)')
    ax1.set_ylabel('Amplitude')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(time_ds, ds_signal[:, 0], label='Downsampled Sensor 1', color='blue')
    ax2.plot(time_ds, ds_signal[:, 1], label='Downsampled Sensor 2', color='green')
    if filter_type == 'no_filt':
        ax2.set_title('2. Signal After Downsampling')
    elif filter_type == 'filt':
        ax2.set_title('2. Signal After Filtering and Downsampling')
    ax2.set_ylabel('Amplitude')
    ax2.legend()
    ax2.grid(True)
    
    ax3.plot(time_scaled, scaled_signal[:, 0], 'o-', label=f'Downsampled Sensor 1 ({target_fs} Hz)', color='red', markersize=1.5)
    ax3.plot(time_scaled, scaled_signal[:, 1], 'o-', label=f'Downsampled Sensor 2 ({target_fs} Hz)', color='orange', markersize=1.5)
    ax3.set_title(f'3. Signal After Scaling to [-1, 1]')
    ax3.set_xlabel('Time [seconds]')
    ax3.set_ylabel('Amplitude')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    filepath = Path(f'plots/preprocessing/signal_{train_date}/{scaler_type}_{filter_type}_{target_fs}Hz_{plot_duration_seconds}s.png')
    filepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filepath)
    print(f"Plot saved as '{filepath}'")
    plt.close()

def prepare_data(plot_type,filter_type):

    train_raw, val_raw, test_raw = select_data()

    if filter_type == 'filt':
        # Apply filtering and downsampling
        sos = butter(4, [0.01, 0.6], btype='band', output='sos', fs=fs)
        train_signal_filtered = sosfiltfilt(sos, train_raw, axis=0)
        val_signal_filtered = sosfiltfilt(sos, val_raw, axis=0)
        test_signal_filtered = sosfiltfilt(sos, test_raw, axis=0)

        train_signal_ds = train_signal_filtered[::downsample_factor]
        val_signal_ds = val_signal_filtered[::downsample_factor]
        test_signal_ds = test_signal_filtered[::downsample_factor]
    elif filter_type == 'no_filt':
        # No filtering, only downsampling
        train_signal_ds = train_raw[::downsample_factor]
        val_signal_ds = val_raw[::downsample_factor]
        test_signal_ds = test_raw[::downsample_factor]

    if scaler_type == 'single_scaler':
        # Scale data, fit on training data only
        scaler_s1, scaler_s2 = fit_data(train_signal_ds)
        train_scaled = transform_data(train_signal_ds, scaler_s1, scaler_s2)
        val_scaled = transform_data(val_signal_ds, scaler_s1, scaler_s2)
        test_scaled = transform_data(test_signal_ds, scaler_s1, scaler_s2)

    elif scaler_type == 'multi_scaler':
        # Scale data, fit on each individually
        scaler_s1_train, scaler_s2_train = fit_data(train_signal_ds)
        train_scaled = transform_data(train_signal_ds, scaler_s1_train, scaler_s2_train)
        scaler_s1_val, scaler_s2_val = fit_data(val_signal_ds)
        val_scaled = transform_data(val_signal_ds, scaler_s1_val, scaler_s2_val)
        scaler_s1, scaler_s2 = fit_data(test_signal_ds)
        test_scaled = transform_data(test_signal_ds, scaler_s1, scaler_s2)

    # Create windowed data from the SCALED signals
    x_train, y_train = create_windows(train_scaled, input_steps, output_steps)
    x_val, y_val = create_windows(val_scaled, input_steps, output_steps)
    x_test, y_test = create_windows(test_scaled, input_steps, output_steps)

    # --- Convert the full windowed test sets to tensors ---
    X_test_full = torch.from_numpy(x_test).float()
    # y_test_full = torch.from_numpy(y_test).float()

    # Create datasets and dataloaders
    train_dataset = SensorDataset(x_train, y_train)
    val_dataset = SensorDataset(x_val, y_val)

    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)


    if plot_type == True:
        # print("len(train_raw):", len(train_raw))
        # print("len(train_signal_ds):", len(train_signal_ds))
        # print("len(train_scaled):", len(train_scaled))
        # print("Train Raw Shape:", train_raw.shape)
        # print("Train Filtered and Downsampled Shape:", train_signal_ds.shape)
        # print("Train Scaled Shape:", train_scaled.shape)
        # print("-----------------------------")
        # print("Original:", train_raw[:5])
        # print("Filtered and Downsampled:", train_signal_ds[:5])
        # print("Scaled:", train_scaled[:5])
        # print(f"x_train shape: {x_train.shape}")
        # print(f"y_train shape: {y_train.shape}")
        # print(f"Train samples: {len(train_dataset)}")
        # print(f"Train batches: {len(train_loader)}")
        # print("-----------------------------")
        # print("len(test_raw):", len(test_raw))
        # print("len(test_signal_ds):", len(test_signal_ds))
        # print("len(test_scaled):", len(test_scaled))
        print("-----------------------------")
        print("Scaled on training dataset only." if scaler_type == 'single_scaler' else "Scaled on each dataset individually.")
        print("Training Data Statistics:")
        print("Original Std Dev:", np.round(np.std(train_raw), 4))
        print("Downsampled Std Dev:", np.round(np.std(train_signal_ds), 4))
        print("Scaled Std Dev:", np.round(np.std(train_scaled), 4))
        print("Original Mean:", np.round(np.mean(train_raw), 4))
        print("Downsampled Mean:", np.round(np.mean(train_signal_ds), 4))
        print("Scaled Mean:", np.round(np.mean(train_scaled), 4))
        print("Scaled Min:", np.round(np.min(train_scaled), 4))
        print("Scaled Max:", np.round(np.max(train_scaled), 4))
        print("Scaled Range:", np.round(np.max(train_scaled) - np.min(train_scaled), 4))
        print("-----------------------------")

        plot_pre(
            raw_signal=train_raw, 
            ds_signal=train_signal_ds,
            scaled_signal=train_scaled,
            plot_duration_seconds=20,
        )

        plot_pre(
            raw_signal=train_raw, 
            ds_signal=train_signal_ds,
            scaled_signal=train_scaled,
            plot_duration_seconds=120,
        )

        plot_pre(
            raw_signal=train_raw, 
            ds_signal=train_signal_ds,
            scaled_signal=train_scaled,
            plot_duration_seconds=200,
        )

    loaders = (train_loader, val_loader, test_scaled)
    #train_sets = (train_raw, train_signal_ds, train_scaled)
    scalers = {'sensor1': scaler_s1, 'sensor2': scaler_s2}
    test_sets = (test_raw, test_signal_ds, X_test_full)
    return loaders, scalers, test_sets

if __name__ == "__main__":
    # plot_type: True/False to generate output and plots
    prepare_data(plot_type = True,filter_type=filter_type) 

