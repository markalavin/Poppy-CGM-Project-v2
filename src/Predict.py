import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from Model_Architecture import PoppyLSTMModel
from Application_Parameters import INPUT_SAMPLES, PREDICTION_SAMPLES, CGM_MIN, CGM_MAX, CGM_RANGE
from Logging import log_prediction
from datetime import datetime
from Get_Prediction_Data import get_glucose_data_from_api, get_recent_records, parse_time_input
import glob
import os

# 1. MODEL ARCHITECTURECGM
# PyTorch needs this class definition to understand how to load the weights
class PoppyLSTMModel(nn.Module):
    def __init__(self, input_size=7, hidden_size=64, num_layers=2, output_size=12):
        super(PoppyLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Take the output of the last time step
        return out


# 2. HELPER: DATA PREPARATION
def merge_records_to_context(glucose_df, records_df):
    merged_df = glucose_df.copy()

    # Add Cyclic Time Features
    minutes_into_day = merged_df['time'].dt.hour * 60 + merged_df['time'].dt.minute
    merged_df['sin_time'] = np.sin(2 * np.pi * minutes_into_day / 1440)
    merged_df['cos_time'] = np.cos(2 * np.pi * minutes_into_day / 1440)

    if records_df is not None and not records_df.empty:
        pivoted = records_df.pivot_table(index='time', columns='record_type',
                                         values='record_amount', aggfunc='sum').reset_index()
        merged_df = pd.merge_asof(merged_df.sort_values('time'), pivoted.sort_values('time'),
                                  on='time', direction='backward', tolerance=pd.Timedelta("5m"))

    # Force-create missing columns and fill NaNs
    for col in ['insulin', 'meal', 'minimeal', 'karo']:
        if col not in merged_df.columns:
            merged_df[col] = 0.0
    return merged_df.fillna(0.0)


def prepare_tensor(df):
    features = ['glucose', 'insulin', 'meal', 'minimeal', 'karo', 'sin_time', 'cos_time']
    df_norm = df.copy()
    # Apply your custom scaling
    df_norm['glucose'] = (df_norm['glucose'] - 50.0) / 350.0
    df_norm['insulin'] = df_norm['insulin'] / 12.0
    df_norm['karo'] = df_norm['karo'] / 2.0
    df_norm['minimeal'] = df_norm['minimeal'] / 2.0
    df_norm['meal'] = df_norm['meal'] / 50.0

    data_array = df_norm[features].values.astype(np.float32)
    return torch.tensor(data_array).unsqueeze(0)


# 3. MAIN EXECUTION
import glob
import os


# 3. MAIN EXECUTION
def Predict(prediction_time=None, device_arg=None):
    print("--- Starting Predict() ---")
    if device_arg:
        device = torch.device(device_arg)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Step 1: Using device: {device}")

    # Initialize model architecture
    model = PoppyLSTMModel(input_size=7, output_size=PREDICTION_SAMPLES).to(device)

    try:
        # Step 2: Automatically find the LATEST model file
        # This looks for any file starting with 'poppy_model_' in the src/ folder
        search_pattern = os.path.join("src", "poppy_model_*.pth*")
        list_of_files = glob.glob(search_pattern)

        if not list_of_files:
            # Fallback check if the user is running from inside /src already
            list_of_files = glob.glob("poppy_model_*.pth*")

        if list_of_files:
            # Get the file with the most recent creation time
            latest_model = max(list_of_files, key=os.path.getctime)
            print(f"Step 2: Automatically loading newest model: {latest_model}")

            checkpoint = torch.load(latest_model, map_location=device)
            model.load_state_dict(checkpoint)
            model.eval()
            print("Model weights loaded successfully.")
        else:
            print("STOP: No files matching 'poppy_model_*.pth' found in src/ folder.")
            return

    except Exception as e:
        print(f"An error occurred during model loading: {type(e).__name__}")
        print(f"Error message: {e}")
        return
    # Data Acquisition
    print("Step 3: Fetching glucose data from API...")
    current_time = pd.Timestamp.now()
    glucose_df = get_glucose_data_from_api(email='markalavin@gmail.com', password='my$Poppy$Dog1')

    print(f"Step 4: API returned {len(glucose_df)} rows of glucose data.")
    if len(glucose_df) < INPUT_SAMPLES:
        print(f"STOP: Not enough data points (Need {INPUT_SAMPLES}, got {len(glucose_df)}).")
        return

    print("Step 5: Fetching recent records (insulin/meals)...")
    records_df = get_recent_records(current_time)

    print("Step 6: Merging data streams...")
    merged_df = merge_records_to_context(glucose_df, records_df)

    print("Step 7: Preparing input tensor...")
    input_tensor = prepare_tensor(merged_df).to(device)

    print("Step 8: Running Inference...")
    with torch.no_grad():
        prediction = model(input_tensor)

    print("Step 9: De-normalizing results...")
    final_forecast = (prediction.cpu().numpy() * 350.0) + 50.0

    # Save this prediction in the log file for later comparison with "actuals":
    log_prediction(
        current_glucose=merged_df['glucose'].iloc[-1],
        forecast_array=final_forecast.flatten() )
    #    insulin=insulin_input,  # These come from your user prompts
    #    carbs=meal_input



    print(f"\nSUCCESS! Poppy's 120-Minute Forecast (5-min intervals):")
    # This will now print all (24) predicted points
    print(final_forecast.flatten().round(1))

    plot_forecast( merged_df, final_forecast.flatten() )



import matplotlib.pyplot as plt


def plot_forecast(recent_df, forecast_array):
    plt.figure(figsize=(12, 6))

    # 1. Plot the Actual History
    plt.plot(recent_df['time'], recent_df['glucose'],
             label='Actual History (Past 6h)', color='blue', marker='o', markersize=3)

    # 2. Setup Forecast Times
    last_time = recent_df['time'].iloc[-1]
    forecast_times = [last_time + pd.Timedelta(minutes=5 * (i + 1)) for i in range(len(forecast_array))]

    # 3. Plot the Forecast
    plt.plot(forecast_times, forecast_array,
             label='LSTM Forecast (Next 2h)', color='red', linestyle='--', marker='s', markersize=3)

    # 4. Add Canine-Specific Thresholds
    plt.axhline(y=CGM_MIN, color='green', linestyle=':', label=f'Low Target ({CGM_MIN})')
    plt.axhline(y=CGM_MAX, color='orange', linestyle=':', label=f'High Target ({CGM_MAX})')

    # NEW: Create a dynamic, timestamped title
    formatted_time = last_time.strftime('%Y-%m-%d %H:%M')
    plt.title(f"Poppy's Glucose Forecast\nLast Reading: {formatted_time}")

    # Formatting
    plt.xlabel("Time")
    plt.ylabel("Glucose (mg/dL)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Optional: Save the plot automatically
    # plt.savefig(f"forecast_{last_time.strftime('%Y%m%d_%H%M')}.png")

    plt.show()

if __name__ == "__main__":
    Predict()