import torch
import numpy as np
import pandas as pd
from Application_Parameters import INPUT_SAMPLES, PREDICTION_SAMPLES


def create_windows(df):
    """
    df: Resampled dataframe with a DatetimeIndex (5-min frequency)
    Uses INPUT_SAMPLES (72) for history and PREDICTION_SAMPLES (24) for future.
    """

    # Ensure the 'Timestamp' column is actually a datetime object
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.set_index('Timestamp')

    # 1. Add Cyclical Time Features (Sin/Cos)
    hour_val = df.index.hour + df.index.minute / 60.0
    df['hour_sin'] = np.sin(2 * np.pi * hour_val / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour_val / 24)

    # 2. Select columns for the input vector
    feature_cols = ['Historic Glucose mg/dL', 'insulin', 'meal', 'minimeal', 'karo', 'hour_sin', 'hour_cos']
    features = df[feature_cols].values
    target = df['Historic Glucose mg/dL'].values

    X_list, y_list = [], []

    # 3. Slide the window with Gap-Aware logic
    # Stop point ensures room for both the input window and the prediction window
    stop_point = len(df) - INPUT_SAMPLES - PREDICTION_SAMPLES

    for i in range(stop_point):
        # Verify the history window represents exact clock time
        start_time = df.index[i]
        end_time = df.index[i + INPUT_SAMPLES - 1]

        # Check for gaps in the input window
        if (end_time - start_time) == pd.Timedelta(minutes=(INPUT_SAMPLES - 1) * 5):
            window = features[i: i + INPUT_SAMPLES]

            # The Label: Grab the NEXT 24 samples (2 hours)
            label = target[i + INPUT_SAMPLES: i + INPUT_SAMPLES + PREDICTION_SAMPLES]

            # Final safety check: discard if there are missing values (NaNs)
            if not np.isnan(window).any() and not np.isnan(label).any():
                X_list.append(window)
                y_list.append(label)

    # 4. Convert to GPU Tensors
    X_tensor = torch.tensor(np.array(X_list), dtype=torch.float32).cuda()
    # y_tensor shape will now be [Number of Windows, 24]
    y_tensor = torch.tensor(np.array(y_list), dtype=torch.float32).cuda()

    return X_tensor, y_tensor