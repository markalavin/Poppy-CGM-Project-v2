import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class PoppyDataset(Dataset):
    def __init__(self, cgm_file, events_file, window_size=160, predict_ahead=30):
        self.window_size = window_size
        self.predict_ahead = predict_ahead

        # --- 1. LOAD CGM DATA (Brute Force Discovery) ---
        try:
            with open(cgm_file, 'r') as f:
                lines = f.readlines()

            header_row_idx = 0
            for i, line in enumerate(lines[:30]):
                if line.count(',') > 10:
                    header_row_idx = i
                    break

            self.cgm = pd.read_csv(cgm_file, skiprows=header_row_idx)
            self.cgm.columns = self.cgm.columns.str.strip()

            # Flexible Mapping for Timestamps and Glucose
            ts_col = [c for c in self.cgm.columns if 'Timestamp' in c][0]
            gl_col = [c for c in self.cgm.columns if 'Glucose' in c and 'Historic' in c][0]

            self.cgm['Timestamp'] = pd.to_datetime(self.cgm[ts_col])
            self.cgm['Glucose'] = pd.to_numeric(self.cgm[gl_col], errors='coerce')

            print(f"✅ Headers found at row {header_row_idx}")
        except Exception as e:
            print(f"❌ CGM Load Error: {e}")
            raise

        # --- 2. LOAD REPORTS DATA ---
        try:
            self.events = pd.read_csv(events_file)
            self.events.columns = self.events.columns.str.strip()
            # Ensure the reports timestamp is also converted
            self.events['Timestamp'] = pd.to_datetime(self.events['timestamp'])
        except Exception as e:
            print(f"❌ Reports Load Error: {e}")
            raise

        # --- 3. PROCESSING & ALIGNMENT (Speed Demon) ---
        self.cgm = self.cgm.dropna(subset=['Glucose']).sort_values('Timestamp')

        # Filter to last 60 days to keep the P1 fast
        latest_date = self.cgm['Timestamp'].max()
        start_date = latest_date - pd.Timedelta(days=60)
        self.cgm = self.cgm[self.cgm['Timestamp'] > start_date]

        print(f"⏳ Resampling {len(self.cgm)} rows into 1-minute grid...")

        self.cgm = self.cgm.set_index('Timestamp')
        self.cgm = self.cgm[['Glucose']].resample('1min').mean().interpolate(method='linear')

        # Initialize Karo column
        self.cgm['Karo_Impulse'] = 0.0

        # Map Karo events
        karo_mask = self.events['report_type'].str.contains('Karo', case=False, na=False)
        karo_events = self.events[karo_mask]

        mapped_count = 0
        for _, row in karo_events.iterrows():
            try:
                # Check if event falls within our 60-day window
                if row['Timestamp'] >= self.cgm.index.min() and row['Timestamp'] <= self.cgm.index.max():
                    idx = self.cgm.index.get_indexer([row['Timestamp']], method='nearest')[0]
                    val = float(row['report_measure']) if not pd.isna(row['report_measure']) else 1.0
                    self.cgm.iloc[idx, self.cgm.columns.get_loc('Karo_Impulse')] = val
                    mapped_count += 1
            except:
                continue

        print(f"✅ Successfully mapped {mapped_count} Karo events.")

    def __len__(self):
        return len(self.cgm) - self.window_size - self.predict_ahead

    def __getitem__(self, idx):
        window = self.cgm.iloc[idx: idx + self.window_size]

        glucose_tensor = torch.tensor(window['Glucose'].values, dtype=torch.float32)
        impulse_tensor = torch.tensor(window['Karo_Impulse'].values, dtype=torch.float32)

        # Shape: [2, 160]
        x = torch.stack([glucose_tensor, impulse_tensor])

        y_val = self.cgm.iloc[idx + self.window_size + self.predict_ahead]['Glucose']
        y = torch.tensor(y_val, dtype=torch.float32)

        return x, y