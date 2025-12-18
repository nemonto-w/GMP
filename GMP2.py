import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path
from scipy.signal import butter, sosfiltfilt, welch, coherence
from sklearn.preprocessing import MinMaxScaler
import math
from IPython.display import display, Latex

# ==========================================
# 1. Helper Classes (Models & Utilities)
# ==========================================

class MSPELoss(nn.Module):
    def __init__(self, sigma=1.0):
        super(MSPELoss, self).__init__()
        self.register_buffer('sigma', torch.tensor(sigma))

    def forward(self, pred, true):
        denominator = torch.maximum(true, self.sigma)
        a = (true - pred) / denominator
        loss = torch.mean(a ** 2)
        return loss

class SensorDataset(Dataset):
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
    def __init__(self, params, constants):
        super().__init__()
        self.output_steps = constants["output_seconds"] * constants["target_fs"]
        self.num_features = constants["NUM_FEATURES"]
        self.constants = constants

        self.gru1 = nn.GRU(
            input_size=self.num_features,
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
        self.output_layer = nn.Linear(
            params["dense_units"], 
            self.output_steps * self.num_features
        )

    def forward(self, x):
        x, _ = self.gru1(x)
        x = self.dropout1(x)
        x, _ = self.gru2(x)
        x = x[:, -1, :]
        x = self.dropout2(x)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.output_layer(x)
        outputs = x.view(-1, self.output_steps, self.num_features)
        return outputs

class LSTMModel(nn.Module):
    def __init__(self, params, constants):
        super().__init__()
        self.output_steps = constants["output_seconds"] * constants["target_fs"]
        self.num_features = constants["NUM_FEATURES"]
        self.constants = constants

        self.lstm1 = nn.LSTM(
            input_size=self.num_features,
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
        self.output_layer = nn.Linear(
            params["dense_units"], 
            self.output_steps * self.num_features
        )

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]
        x = self.dropout2(x)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.output_layer(x)
        outputs = x.view(-1, self.output_steps, self.num_features)
        return outputs

# ==========================================
# 2. Main Class: GMP
# ==========================================

class gmp:
    def __init__(self, constants, data_config, gru_params, lstm_params):
        self.constants = constants
        self.data_config = data_config
        self.gru_params = gru_params
        self.lstm_params = lstm_params
        
        self.fs = constants["fs"]
        self.target_fs = constants["target_fs"]
        self.downsample_factor = self.fs // self.target_fs
        self.input_steps = constants["input_seconds"] * constants["target_fs"]
        self.output_steps = constants["output_seconds"] * constants["target_fs"]
        self.batch_size = constants["BATCH_SIZE"]
        self.epochs = constants["EPOCHS"]
        self.patience = constants["PATIENCE"]
        self.learning_rate = constants["LEARNING_RATE"]
        
        # Filter Params
        self.filter_low = data_config.get("filter_low", 0.01)
        self.filter_high = data_config.get("filter_high", 0.6)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = None
        self.model_name = None
        self.scalers = {}
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.train_raw = None
        self.val_raw = None
        self.test_raw = None
        self.test_scaled = None 
        self.X_test_full = None 

    def _load_data_file(self, date_str):
        path1 = self.data_config["path1"].format(date=date_str)
        path2 = self.data_config["path2"].format(date=date_str)
        if not os.path.exists(path1) or not os.path.exists(path2):
            raise FileNotFoundError(f"Data files for {date_str} not found.")
        s1 = np.loadtxt(path1)
        s2 = np.loadtxt(path2)
        return np.stack([s1, s2], axis=-1)

    def fit_scalers(self, data):
        s1 = MinMaxScaler((-1, 1)).fit(data[:, 0].reshape(-1, 1))
        s2 = MinMaxScaler((-1, 1)).fit(data[:, 1].reshape(-1, 1))
        return (s1, s2)

    def transform_data(self, data, s1, s2):
        return np.hstack([
            s1.transform(data[:, 0].reshape(-1, 1)),
            s2.transform(data[:, 1].reshape(-1, 1))
        ])

    def create_windows(self, data):
        X, Y = [], []
        for i in range(len(data) - self.input_steps - self.output_steps + 1):
            X.append(data[i:(i + self.input_steps), :])
            Y.append(data[(i + self.input_steps):(i + self.input_steps + self.output_steps), :])
        return np.array(X), np.array(Y)

    def preprocess_data(self, plot_type=False):
        print("\n--- Preprocessing ---")
        train_d = self.data_config["train_date"]
        test_d = self.data_config["test_date"]
        
        train_raw_sig = self._load_data_file(train_d)
        test_raw_sig = self._load_data_file(test_d)

        ts = int(len(train_raw_sig) * 0.8)
        vs = int(len(train_raw_sig) * 0.2)
        
        self.train_raw = train_raw_sig[:ts]
        self.val_raw = train_raw_sig[ts: ts + vs]
        self.test_raw = test_raw_sig

        ft = self.data_config["filter_type"]
        
        if ft == 'filt':
            f_lo, f_hi = self.filter_low, self.filter_high
            if f_lo >= f_hi or f_lo <= 0: f_lo, f_hi = 0.01, 0.6
            print(f"Filter: {f_lo}-{f_hi} Hz")
            sos = butter(4, [f_lo, f_hi], 'band', output='sos', fs=self.fs)
            
            tr_filt = sosfiltfilt(sos, self.train_raw, axis=0)
            val_filt = sosfiltfilt(sos, self.val_raw, axis=0)
            te_filt = sosfiltfilt(sos, self.test_raw, axis=0)
            
            tr_ds = tr_filt[::self.downsample_factor]
            val_ds = val_filt[::self.downsample_factor]
            te_ds = te_filt[::self.downsample_factor]
        else:
            tr_ds = self.train_raw[::self.downsample_factor]
            val_ds = self.val_raw[::self.downsample_factor]
            te_ds = self.test_raw[::self.downsample_factor]

        st = self.data_config["scaler_type"]
        if st == 'single_scaler':
            s1, s2 = self.fit_scalers(tr_ds)
            tr_sc = self.transform_data(tr_ds, s1, s2)
            val_sc = self.transform_data(val_ds, s1, s2)
            te_sc = self.transform_data(te_ds, s1, s2)
            self.scalers['sensor1'], self.scalers['sensor2'] = s1, s2
        else:
            s1_tr, s2_tr = self.fit_scalers(tr_ds)
            tr_sc = self.transform_data(tr_ds, s1_tr, s2_tr)
            
            s1_val, s2_val = self.fit_scalers(val_ds)
            val_sc = self.transform_data(val_ds, s1_val, s2_val)
            
            s1_te, s2_te = self.fit_scalers(te_ds)
            te_sc = self.transform_data(te_ds, s1_te, s2_te)
            
            self.scalers['sensor1'], self.scalers['sensor2'] = s1_te, s2_te

        self.test_scaled = te_sc

        x_tr, y_tr = self.create_windows(tr_sc)
        x_val, y_val = self.create_windows(val_sc)
        x_te, y_te = self.create_windows(te_sc)

        self.train_loader = DataLoader(SensorDataset(x_tr, y_tr), batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(SensorDataset(x_val, y_val), batch_size=self.batch_size, shuffle=False)
        self.X_test_full = torch.from_numpy(x_te).float()

        if plot_type:
            self._plot_pre_helper(self.train_raw, tr_ds, tr_sc, 20)
        
        return self.train_loader, self.val_loader, self.scalers

    def _plot_pre_helper(self, raw, ds, sc, sec):
        nr = int(sec * self.fs)
        nd = int(sec * self.target_fs)
        
        t_r = np.arange(nr) / self.fs
        t_d = np.arange(nd) / self.target_fs
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
        ax1.plot(t_r, raw[:nr, 0], 'gray', alpha=0.8, label='S1')
        ax1.plot(t_r, raw[:nr, 1], 'gray', alpha=0.8, label='S2')
        ax1.set_title(f'Raw ({sec}s)')
        ax2.plot(t_d, ds[:nd, 0], 'b', label='DS S1')
        ax2.plot(t_d, ds[:nd, 1], 'g', label='DS S2')
        ax2.set_title('Downsampled')
        ax3.plot(t_d, sc[:nd, 0], 'r.-', label='Sc S1')
        ax3.plot(t_d, sc[:nd, 1], 'orange', label='Sc S2')
        ax3.set_title('Scaled')
        plt.tight_layout()
        plt.savefig(f'plots/preprocessing_{sec}s.png')
        plt.close()

    def build_model(self, model_name='lstm'):
        self.model_name = model_name.lower()
        if self.model_name == 'gru':
            self.model = GRUModel(self.gru_params, self.constants)
        elif self.model_name == 'lstm':
            self.model = LSTMModel(self.lstm_params, self.constants)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        self.model.to(self.device)

    def train_model(self):
        if self.model is None: raise RuntimeError("Build model first.")
        if self.train_loader is None: self.preprocess_data()

        os.makedirs("saved_models", exist_ok=True)
        mpath = os.path.join("saved_models", f"best_{self.model_name}_model.pt")
        crit = nn.MSELoss()
        opt = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        hist = {'loss': [], 'val_loss': []}
        best_vl = float('inf')
        pats = 0

        print(f"Training {self.model_name} on {self.device}...")
        for ep in range(self.epochs):
            self.model.train()
            rl = 0.0
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad()
                out = self.model(x)
                loss = crit(out, y)
                loss.backward()
                opt.step()
                rl += loss.item() * x.size(0)
            
            el = rl / len(self.train_loader.dataset)
            hist['loss'].append(el)

            self.model.eval()
            vrl = 0.0
            with torch.no_grad():
                for x, y in self.val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out = self.model(x)
                    l = crit(out, y)
                    vrl += l.item() * x.size(0)
            
            vel = vrl / len(self.val_loader.dataset)
            hist['val_loss'].append(vel)
            
            print(f"Epoch {ep+1} | Loss: {el:.6f} | Val: {vel:.6f}")

            if vel < best_vl:
                best_vl = vel
                torch.save(self.model.state_dict(), mpath)
                pats = 0
            else:
                pats += 1
            if pats >= self.patience:
                print("Early stopping.")
                break
        
        self._plot_hist(hist)

    def _plot_hist(self, hist):
        plt.figure(figsize=(10, 5))
        plt.plot(hist['loss'], label='Train')
        plt.plot(hist['val_loss'], label='Val')
        plt.title('Loss')
        plt.legend()
        plt.savefig(f"plots/{self.model_name}_loss.png")
        plt.close()

    def evaluate_model(self):
        mpath = os.path.join("saved_models", f"best_{self.model_name}_model.pt")
        if os.path.exists(mpath):
            self.model.load_state_dict(torch.load(mpath, map_location=self.device))
        self.model.eval()
        
        skip = self.data_config["test_skip_seconds"] * self.target_fs
        if self.test_scaled is None: self.preprocess_data()
        
        x_te, y_te = self.create_windows(self.test_scaled[skip:])
        loader = DataLoader(SensorDataset(x_te, y_te), batch_size=self.batch_size, shuffle=False)
        
        crit = nn.MSELoss()
        tl = 0.0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                tl += crit(out, y).item() * x.size(0)
        mse = tl / len(loader.dataset)
        rmse = np.sqrt(mse)
        print("\n--- Test Set Evaluation Results ---")
        print(f"Model: {self.model_name}")
        print(f"Test MSE: {mse:.6f}")
        print(f"Test RMSE: {rmse:.6f}")
        print("-----------------------------------")

    def export_predicted_data(self):
        if self.model is None: self.build_model(self.model_name or 'lstm')
        self.model.eval()
        
        preds = []
        with torch.no_grad():
            for i in range(0, len(self.X_test_full), self.batch_size):
                batch = self.X_test_full[i:i+self.batch_size].to(self.device)
                preds.append(self.model(batch).cpu())
        
        preds_np = torch.cat(preds, 0).numpy()
        
        # Inverse
        n, steps, feats = preds_np.shape
        s1 = self.scalers['sensor1']
        s2 = self.scalers['sensor2']
        
        inv_s1 = s1.inverse_transform(preds_np[:,:,0].reshape(-1,1)).flatten()
        inv_s2 = s2.inverse_transform(preds_np[:,:,1].reshape(-1,1)).flatten()
        inv_preds = np.stack([inv_s1, inv_s2], axis=-1).reshape(n, steps, feats)
        
        test_ds = self.test_raw[::self.downsample_factor]
        full_pre = np.full_like(test_ds, np.nan)
        
        for i in range(n):
            idx = i + self.input_steps
            if idx < len(full_pre): full_pre[idx,:] = inv_preds[i,0,:]
            
        tr_d = self.data_config['train_date']
        te_d = self.data_config['test_date']
        fdir = f"data/{self.model_name}_Train_{tr_d}/TestData_{te_d}"
        os.makedirs(fdir, exist_ok=True)
        np.savetxt(f"{fdir}/target_{te_d}_ds.txt", test_ds)
        np.savetxt(f"{fdir}/full_pre_{te_d}_ds.txt", full_pre)
        print(f"Saved target and predicted signals to {fdir}")
        
        return test_ds, full_pre
    
    # ==========================================
    # Forecast Function
    # ==========================================

    def forecast(self, input_10sec_s1_raw, input_10sec_s2_raw):
        if self.model is None:
            self.build_model(self.model_name or 'lstm')
            path = os.path.join("saved_models", f"best_{self.model_name}_model.pt")
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.model.eval()

        if self.data_config['filter_type'] == 'filt':
             sos = butter(4, [self.filter_low, self.filter_high], btype='band', output='sos', fs=self.fs)
             s1_filt = sosfiltfilt(sos, input_10sec_s1_raw, axis=0)
             s2_filt = sosfiltfilt(sos, input_10sec_s2_raw, axis=0)
             s1_ds = s1_filt[::self.downsample_factor]
             s2_ds = s2_filt[::self.downsample_factor]
        else:
             s1_ds = input_10sec_s1_raw[::self.downsample_factor]
             s2_ds = input_10sec_s2_raw[::self.downsample_factor]

        if 'sensor1' not in self.scalers:
            self.preprocess_data(plot_type=False)
            
        s1_scaled = self.scalers['sensor1'].transform(s1_ds.reshape(-1, 1))
        s2_scaled = self.scalers['sensor2'].transform(s2_ds.reshape(-1, 1))
        
        input_arr = np.hstack([s1_scaled, s2_scaled])
        
        with torch.no_grad():
            inp_tensor = torch.from_numpy(input_arr).float().unsqueeze(0).to(self.device)
            out_tensor = self.model(inp_tensor)
            out_arr = out_tensor.cpu().squeeze(0).numpy()
            
        pred_s1 = self.scalers['sensor1'].inverse_transform(out_arr[:, 0].reshape(-1, 1))
        pred_s2 = self.scalers['sensor2'].inverse_transform(out_arr[:, 1].reshape(-1, 1))
        
        return np.hstack([pred_s1, pred_s2])
    
    # ==========================================
    # ANALYSIS METHODS
    # ==========================================

    def analyze_raw_signal(self, date_str=None, threshold=0.99):
        """Analyzes input signal coherence to tune filters."""
        if date_str is None: date_str = self.data_config["test_date"]
        try:
            sig = self._load_data_file(date_str)
            raw1, raw2 = sig[:,0], sig[:,1]
        except:
            print("Skipping raw analysis (file not found).")
            return

        f, Cxy = coherence(raw1, raw2, fs=self.fs, nperseg=len(raw1)//5)
        
        # Plot Coherence
        plt.figure(figsize=(10, 6))
        plt.semilogy(f, Cxy)
        plt.title(f"Raw Input Coherence ({date_str})")
        plt.grid(True)
        os.makedirs("plots/performance", exist_ok=True)
        plt.savefig(f"plots/performance/coherence_{date_str}.png")
        plt.close()
        
        # Tune Filter
        valid = np.where(Cxy >= threshold)[0]
        if len(valid) > 0:
            f_start = max(0.001, f[valid[0]])
            f_end = f[valid[-1]]
            print(f"Updating Filter to High Coherence Band: {f_start:.4f}-{f_end:.4f} Hz")
            self.filter_low = f_start
            self.filter_high = f_end

    def _performance_metrics(self, low, high, f, asd_error, asd_target):
        """Helper to calculate spectral metrics."""
        mask2 = (f > low) * (f < high)
        f_ = f[mask2]
        asd_error_ = asd_error[mask2]
        asd_target_ = asd_target[mask2]
        
        # Avoid div by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            asd_ratio = asd_error_ / asd_target_
            
        # Trapz integration
        pse = np.sqrt(np.trapz(asd_ratio**2, f_))
        ms = 1 / np.min(asd_ratio) if len(asd_ratio) > 0 else 0
        rms_e = np.sqrt(np.trapz(asd_error_**2, f_))
        rms_t = np.sqrt(np.trapz(asd_target_**2, f_))
        rmsp = rms_e / rms_t if rms_t != 0 else 0

        return pse, ms, rms_e, rms_t, rmsp

    def _output_results(self, low, high, pse, ms, rms_e, rms_t, rmsp, sensor_index):
        """Helper to print/display results."""
        print(f"\n--- Performance Metrics (Sensor {sensor_index}) [{low:.2f}-{high:.2f} Hz] ---")
        
        # Text based output
        print(f'Percentage Spectrum Error: {pse*100:.4f}%')
        print(f'Maximum Suppression: {ms:.4f}')
        print(f'Residual RMSE: {rms_e:.4f}')
        print(f'Ground Motion RMSE: {rms_t:.4f}')
        print(f'RMS Percentage Error: {rmsp*100:.4f}%')

        # LaTeX display for Jupyter environments
        try:
            text1= r"$\Large\sqrt{\int_{low}^{high}\left(\frac{E(f)}{T(F)}\right)^2,df}\times100\% = $"
            text2= r"$\huge\frac{1}{min(\frac{E(f)}{T(f)})} = $"
            text3= r"$\Large RMS(E)=\sqrt{\int E(f)^2,df} = $"
            text4= r"$\Large RMS(T)=\sqrt{\int T(f)^2,df} = $"
            text5= r"$\Large\frac{RMS(E)}{RMS(T)}\times100\% = $"
            
            display(Latex(f'Percentage Spectrum Error = {text1} {pse*100:.4f}%'))
            display(Latex(f'Maximum Suppression = {text2} {ms:.4f}'))
            display(Latex(f'Residual RMSE = {text3} {rms_e:.4f}'))
            display(Latex(f'Ground Motion RMSE = {text4} {rms_t:.4f}'))
            display(Latex(f'RMS Percentage Error = {text5} {rmsp*100:.4f}%'))
        except:
            pass # Use standard print if display/Latex fails

    def analyze_prediction_performance(self):
        """
        Reads exported predicted results, calculates ASD, Plots comparisons,
        and computes metrics for the specific filter band.
        """
        print("\n--- Analyzing Prediction Performance ---")
        
        # 1. Load Data
        tr_d = self.data_config['train_date']
        te_d = self.data_config['test_date']
        # fdir = f"data/{self.model_name}_Train_{tr_d}/TestData_{te_d}"
        fdir = "data/combined_results_1105_02_ds"
        try:
            target = np.loadtxt(f"{fdir}/target_{te_d}_ds.txt")
            pred = np.loadtxt(f"{fdir}/full_pre_{te_d}_ds.txt")
        except FileNotFoundError:
            print("Prediction files not found. Run export_predicted_data() first.")
            return

        # Handle NaNs (skip initial window)
        valid_idx = ~np.isnan(pred[:,0])
        target = target[valid_idx]
        pred = pred[valid_idx]
        error = target - pred
        
        fs = self.target_fs
        nperseg = len(target) // 5

        # 2. Iterate per Sensor
        for i in range(2): # 2 Sensors
            t_s = target[:, i]
            p_s = pred[:, i]
            e_s = error[:, i]
            
            # ASD Calculations
            f, P_t = welch(t_s, fs=fs, nperseg=nperseg)
            _, P_p = welch(p_s, fs=fs, nperseg=nperseg)
            _, P_e = welch(e_s, fs=fs, nperseg=nperseg)
            
            asd_t = np.sqrt(P_t)
            asd_p = np.sqrt(P_p)
            asd_e = np.sqrt(P_e)
            
            # Plot 1: Prediction vs Target
            plt.figure(figsize=(10, 6))
            plt.loglog(f, asd_t, label='Target')
            plt.loglog(f, asd_p, label='Prediction')
            plt.title(f"Sensor {i+1}: Prediction vs Target ASD")
            plt.legend()
            plt.grid(True, which="both")
            plt.savefig(f"{fdir}/S{i+1}_pred_vs_target_asd.png")
            plt.close()
            
            # Plot 2: Error vs Target
            plt.figure(figsize=(10, 6))
            plt.loglog(f, asd_t, label='Target')
            plt.loglog(f, asd_e, label='Error')
            plt.title(f"Sensor {i+1}: Error vs Target ASD")
            plt.legend()
            plt.grid(True, which="both")
            plt.savefig(f"{fdir}/S{i+1}_error_vs_target_asd.png")
            plt.close()
            
            # Plot 3: All three
            plt.figure(figsize=(10, 6))
            plt.loglog(f, asd_t, label='Target')
            plt.loglog(f, asd_p, label='Prediction')
            plt.loglog(f, asd_e, label='Error')
            plt.title(f"Sensor {i+1}: Pred, Target, Error ASD")
            plt.legend()
            plt.grid(True, which="both")
            plt.savefig(f"{fdir}/S{i+1}_all_asd.png")
            plt.close()
            
            # 3. Metrics
            # Use the filter band found/set in config
            low, high = self.filter_low, self.filter_high
            
            # Calculate metrics using helper
            metrics = self._performance_metrics(low, high, f, asd_e, asd_t)
            
            # Output results
            self._output_results(low, high, *metrics, sensor_index=i+1)

if __name__ == "__main__":
    
    # Configuration
    DATA_CONF = {
        "path1": 'data/data_1105_00_06/s1_{date}.txt',
        "path2": 'data/data_1105_00_06/s2_{date}.txt',
        "train_date": '1105_04',
        "test_date": '1105_05',
        "test_skip_seconds": 20,
        "scaler_type": 'multi_scaler', 
        "filter_type" : 'filt',
        "filter_low": 0.01,
        "filter_high": 0.6
    }

    CONSTANTS = {
        "fs": 512, "target_fs": 64,
        "input_seconds": 10, "output_seconds": 1,
        "BATCH_SIZE": 64, "EPOCHS": 5,
        "PATIENCE": 5, "NUM_FEATURES": 2, "LEARNING_RATE": 1e-4
    }
    
    GRU_PARAMS = {"gru_units_layer1": 64, "dropout": 0.2, "gru_units_layer2": 64, "dense_units": 32}
    LSTM_PARAMS = {"lstm_units_layer1": 64, "dropout": 0.2, "lstm_units_layer2": 64, "dense_units": 32}

    pipeline = gmp(CONSTANTS, DATA_CONF, GRU_PARAMS, LSTM_PARAMS)

    # 1. Raw Analysis (Updates filter band)
    pipeline.analyze_raw_signal(DATA_CONF["test_date"], threshold=0.99)

    # 2. Train
    pipeline.preprocess_data()
    pipeline.build_model('lstm')
    pipeline.train_model()
    
    # 3. Export
    pipeline.export_predicted_data()
    
    # 4. Post-Prediction Performance Analysis
    pipeline.analyze_prediction_performance()

    # 5. Forecast Example
    # Example input data (replace with actual data)
    input_10sec_s1_raw = np.random.randn(5120, 1)  # 10 seconds of data at 512 Hz
    input_10sec_s2_raw = np.random.randn(5120, 1)  # 10 seconds of data at 512 Hz
    forecasted_data = pipeline.forecast(input_10sec_s1_raw, input_10sec_s2_raw)
    print("Forecasted Data Shape:", forecasted_data.shape)

    #use date for the exported model
    # 3 tables
    # comparison of different models, same model - different parameters, same model - different data sets
    # best performance model in the center of the table
    # heat map
    # y_from_u[i] = ham.step(u[i]) 
    # 10 datapoints for 1 trainable parameter #rule of thumb
    # 10 hours training for the final model
    # class model path function sent to the user
    # ask Dr. Kazim for the data points.