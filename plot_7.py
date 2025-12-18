import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from config_1 import CONSTANTS,DATA

input_seconds = CONSTANTS["input_seconds"]
output_seconds = CONSTANTS["output_seconds"]
target_fs = CONSTANTS["target_fs"]
INPUT_STEPS = input_seconds * target_fs
OUTPUT_STEPS = output_seconds * target_fs
train_date = DATA["train_date"]
test_date = DATA["test_date"]
scaler_type = DATA["scaler_type"]
filter_type = DATA["filter_type"]
test_skip_seconds = DATA["test_skip_seconds"]

def read_signals(model_name):
    path1 = Path(f'data/{model_name} Train_{train_date}_{scaler_type}_{filter_type}/TestData_{test_date}/target_{test_date}_ds.txt')
    path2 = Path(f'data/{model_name} Train_{train_date}_{scaler_type}_{filter_type}/TestData_{test_date}/full_pre_{test_date}_ds.txt')
    test_ds = np.loadtxt(path1)
    pre_ds = np.loadtxt(path2)
    #remove first 20 seconds
    removal = test_skip_seconds*target_fs
    test_ds = test_ds[removal:]
    pre_ds = pre_ds[removal:]
    return test_ds, pre_ds

def plot_full(test_ds, pre_ds, model_name, sensor_plot):

    segment_length = 120 * target_fs
    #num_plots = int(np.ceil(len(test_ds) / segment_length))
    num_plots = 3  # Limit to first 3 plots for brevity
    global_ymin = np.nanmin([np.nanmin(test_ds), np.nanmin(pre_ds)])
    global_ymax = np.nanmax([np.nanmax(test_ds), np.nanmax(pre_ds)])
    padding = (global_ymax - global_ymin) * 0.05 # 5% padding
    global_ymin -= padding
    global_ymax += padding
    # print(f"Generating full prediction plots...")
    
    for i in range(num_plots):
        start_idx = i * segment_length
        end_idx = start_idx + segment_length
        time_axis_full = np.arange(len(test_ds)) / target_fs
        plt.figure(figsize=(18, 7))
        
        if sensor_plot == 'Both':
            plt.plot(time_axis_full[start_idx:end_idx], test_ds[start_idx:end_idx, 0], 
                    color='blue', alpha=0.8, label='Target Data (Sensor 1)')
            plt.plot(time_axis_full[start_idx:end_idx], pre_ds[start_idx:end_idx, 0], 
                    'r--', linewidth=2, label='Stitched Prediction (Sensor 1)')
            plt.plot(time_axis_full[start_idx:end_idx], test_ds[start_idx:end_idx, 1], 
                    color='green', alpha=0.8, label='Target Data (Sensor 2)')
            plt.plot(time_axis_full[start_idx:end_idx], pre_ds[start_idx:end_idx, 1], 
                    'k--', linewidth=2, label='Stitched Prediction (Sensor 2)')
        elif sensor_plot == 'Sensor1':
            # --- Sensor 1 Plots ---
            plt.plot(time_axis_full[start_idx:end_idx], test_ds[start_idx:end_idx, 0], 
                    color='blue', alpha=0.8, label='Target Data (Sensor 1)')
            plt.plot(time_axis_full[start_idx:end_idx], pre_ds[start_idx:end_idx, 0], 
                    'r--', linewidth=2, label='Stitched Prediction (Sensor 1)')
        elif sensor_plot == 'Sensor2':
            # --- Sensor 2 Plots ---
            plt.plot(time_axis_full[start_idx:end_idx], test_ds[start_idx:end_idx, 1], 
                    color='green', alpha=0.8, label='Target Data (Sensor 2)')
            plt.plot(time_axis_full[start_idx:end_idx], pre_ds[start_idx:end_idx, 1], 
                    'k--', linewidth=2, label='Stitched Prediction (Sensor 2)')
        plt.title(f'"{model_name.upper()}": 2 Minutes Prediction ({sensor_plot}, {i+1} of {num_plots})\n'
                  f'Training: {train_date}, Testing: {test_date},\n'
                  f'{target_fs} Hz,{input_seconds}:{output_seconds},{scaler_type},{filter_type}', fontsize=16)
        plt.ylabel('Amplitude')
        plt.xlabel('Time (seconds)')
        plt.legend()
        plt.grid(True)
        plt.xlim(time_axis_full[start_idx], time_axis_full[min(end_idx, len(time_axis_full)-1)])
        plt.ylim(global_ymin, global_ymax)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        filepath = Path(f'plots/{model_name} Train_{train_date}_{scaler_type}_{filter_type}/Test_{test_date}/Full Prediction Plots/{model_name}_{sensor_plot}_{i+1}.png')
        # filepath = Path(f'plotteeee/Full Prediction Plots/{model_name}_{sensor_plot}_{i+1}.png') 
        filepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath)
        print(f"Plot saved as '{filepath}'")
        plt.close()


def plot_subtraction(test_ds, pre_ds, model_name):

    subtraction = test_ds - pre_ds
    segment_length = 120 * target_fs
    #num_plots = int(np.ceil(len(test_ds) / segment_length))
    num_plots = 3  # Limit to first 3 plots for brevity
    
    global_ymin = np.nanmin(subtraction)
    global_ymax = np.nanmax(subtraction)
    padding = (global_ymax - global_ymin) * 0.05 # 5% padding
    global_ymin -= padding
    global_ymax += padding
    # print(f"Generating subtraction plots...")
    
    for i in range(num_plots):
        start_idx = i * segment_length
        end_idx = start_idx + segment_length
        time_axis_full = np.arange(len(test_ds)) / target_fs

        plt.figure(figsize=(18, 7))
        
        plt.plot(time_axis_full[start_idx:end_idx], subtraction[start_idx:end_idx, 0], 
                 color='purple', alpha=0.9, label='Prediction Error (Sensor 1)')
        plt.plot(time_axis_full[start_idx:end_idx], subtraction[start_idx:end_idx, 1], 
                 color='orange', alpha=0.9, label='Prediction Error (Sensor 2)')

        plt.title(f'"{model_name.upper()}": 2 Minutes Prediction Error ({i+1} of {num_plots})\n'
                  f'Training: {train_date}, Testing: {test_date}\n'
                  f'{target_fs} Hz,{input_seconds}:{output_seconds},{scaler_type},{filter_type}', fontsize=16)
        plt.ylabel('Amplitude Difference')
        plt.xlabel('Time (seconds)')
        plt.legend()
        plt.grid(True, linestyle='--')
        plt.xlim(time_axis_full[start_idx], time_axis_full[min(end_idx, len(time_axis_full)-1)])
        plt.ylim(global_ymin, global_ymax)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        filepath = Path(f'plots/{model_name} Train_{train_date}_{scaler_type}_{filter_type}/Test_{test_date}/Subtraction Plots/{model_name}_{i+1}.png')
        # filepath = Path(f'plotteeee/Subtraction Plots/{model_name}_{i+1}.png') 
        filepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath)
        print(f"Plot saved as '{filepath}'")
        plt.close()

if __name__ == "__main__":
    model_name = 'lstm'  # 'gru' or 'lstm'
    test_ds, pre_ds = read_signals(model_name)
    plot_full(test_ds, pre_ds, model_name, 'Both')
    plot_full(test_ds, pre_ds, model_name, 'Sensor1')
    plot_full(test_ds, pre_ds, model_name, 'Sensor2')
    plot_subtraction(test_ds, pre_ds, model_name)