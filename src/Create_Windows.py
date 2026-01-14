import torch
import numpy as np
import pandas as pd

print(f"Torch CUDA version: {torch.version.cuda}")
# print(f"System CUDA version: {torch.utils.cpp_extension.CUDA_HOME}")

# Take "df" the dataframe containing the automatically-processed glucose
# data
def create_windows(df, window_size=48, lead_time=12):
    """
    df: Resampled dataframe with a DatetimeIndex (5-min frequency)
    window_size: 48 (4 hours of 5-min samples)
    lead_time: 12 (1 hour ahead prediction)
    """

    # Ensure the 'Timestamp' column is actually a datetime object
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.set_index('Timestamp')

    # 1. Add Cyclical Time Features (Sin/Cos)
    # We use fractional hours (e.g., 14.5 for 2:30 PM) for a smooth curve
    hour_val = df.index.hour + df.index.minute / 60.0
    df['hour_sin'] = np.sin(2 * np.pi * hour_val / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour_val / 24)

    # 2. Select columns for the input vector
    feature_cols = ['Historic Glucose mg/dL', 'insulin', 'meal', 'minimeal', 'karo', 'hour_sin', 'hour_cos']
    features = df[feature_cols].values
    target = df['Historic Glucose mg/dL'].values

    X_list, y_list = [], []

    # 3. Slide the window with Gap-Aware logic
    for i in range(len(df) - window_size - lead_time):

        # Verify this window represents exactly 4 hours of clock time
        start_time = df.index[i]
        end_time = df.index[i + window_size - 1]

        # If there's a jump in the index (a gap), skip this window
        if (end_time - start_time) == pd.Timedelta(minutes=(window_size - 1) * 5):
            window = features[i: i + window_size]
            label = target[i + window_size + lead_time]

            # Final safety check: discard if there are missing values (NaNs)
            if not np.isnan(window).any() and not np.isnan(label):
                X_list.append(window)
                y_list.append(label)

    # 4. Convert to GPU Tensors
    X_tensor = torch.tensor(np.array(X_list), dtype=torch.float32).cuda()
    y_tensor = torch.tensor(np.array(y_list), dtype=torch.float32).cuda()

    return X_tensor, y_tensor