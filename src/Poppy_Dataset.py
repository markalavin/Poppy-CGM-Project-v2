import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class PoppyDataset(Dataset):
    def __init__(self, cgm_file, events_file, window_size=160, predict_ahead=30, until_date='2024-01-10'):
        self.window_size = window_size
        self.predict_ahead = predict_ahead

        # 1. Load CGM Data
        df_cgm = pd.read_csv(cgm_file, skiprows=1)

        # Explicitly parse the timestamp.
        # 'dayfirst=False' is usually the fix for US-style dates (MM/DD/YYYY)
        df_cgm['Timestamp'] = pd.to_datetime(df_cgm['Device Timestamp'], dayfirst=False, errors='coerce')

        # Drop rows where timestamp or glucose failed to parse
        df_cgm['Glucose'] = pd.to_numeric(df_cgm['Historic Glucose mg/dL'], errors='coerce')
        df_cgm = df_cgm.dropna(subset=['Timestamp', 'Glucose']).sort_values('Timestamp')

        # Debug print to see what the dates actually look like
        if not df_cgm.empty:
            print(f"📅 CGM Data Range: {df_cgm['Timestamp'].min()} to {df_cgm['Timestamp'].max()}")
        else:
            print("❌ ERROR: No data found in CGM file. Check column headers!")

        # 2. Resample to 5-minute intervals
        df_cgm = df_cgm.set_index('Timestamp')
        self.cgm = df_cgm[['Glucose']].resample('5min').mean().interpolate()

        # 3. Apply the Date Cutoff
        # This prevents training on recent CGM data that lacks corresponding event logs
        if until_date:
            cutoff = pd.to_datetime(until_date)
            self.cgm = self.cgm[self.cgm.index <= cutoff]
            print(f"✂️ Data truncated at {until_date}. Training on {len(self.cgm)} intervals.")

        # 4. Apply the "Smear" (Pre-calculating the Impulse Response)
        self.cgm['Metabolic_Pressure'] = 0.0
        df_events = pd.read_csv(events_file)
        df_events['timestamp'] = pd.to_datetime(df_events['timestamp'])

        # Filter events to match our cutoff as well
        if until_date:
            df_events = df_events[df_events['timestamp'] <= cutoff]

        for _, row in df_events.iterrows():
            ts = row['timestamp']
            etype = str(row['report_type']).lower()

            if ts in self.cgm.index.round('5min'):
                idx = self.cgm.index.get_indexer([ts], method='nearest')[0]

                # Apply the 160-minute curve (32 intervals of 5 mins)
                # This represents the observed 2h40m impulse response
                for i in range(32):
                    if idx + i < len(self.cgm):
                        impact = 15.0 * np.exp(-i / 8.0)
                        self.cgm.iloc[idx + i, self.cgm.columns.get_loc('Metabolic_Pressure')] += impact
        # 5. Initialize the Scaler and Scale Glucose
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.cgm['Glucose'] = self.scaler.fit_transform(self.cgm[['Glucose']].values)
        print(f"✅ Pre-processing complete. Data is ready for training.")

    def __len__(self):
        # Ensure we have enough room for the window and the prediction ahead
        # 30 mins ahead = 6 intervals of 5 mins
        return len(self.cgm) - self.window_size - 6

    def __getitem__(self, idx):
        # Slice the window for both Glucose and our Pre-baked Pressure
        gl_window = self.cgm['Glucose'].values[idx: idx + self.window_size]
        mp_window = self.cgm['Metabolic_Pressure'].values[idx: idx + self.window_size]

        # Stack them into a [2, 160] tensor
        x = torch.tensor(np.stack([gl_window, mp_window]), dtype=torch.float32)

        # Target is the glucose value 30 mins (6 intervals) in the future
        y_val = self.cgm['Glucose'].values[idx + self.window_size + 6]
        y = torch.tensor(y_val, dtype=torch.float32)

        return x, y