import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from datetime import datetime, timedelta

from Utilities import time_spec
from Validate import validate_prediction
from Logging import log_prediction
from Get_Prediction_Data import get_glucose_data_from_api, get_recent_records
from Application_Parameters import INPUT_SAMPLES, PREDICTION_SAMPLES, CGM_MIN, CGM_MAX
from Model_Architecture import PoppyLSTMModel


def get_autoregressive_forecast(model, input_seq, horizon):
    model.eval()
    current_window = input_seq.clone()
    forecasts = []

    with torch.no_grad():
        for i in range(horizon):
            prediction = model(current_window)
            val = prediction.item()
            forecasts.append(val)

            # Shift window: Template from the last row
            last_row = current_window[:, -1:, :].clone()
            last_row[0, 0, 0] = val  # Update Glucose

            # Slide: [1, 35, 7] + [1, 1, 7]
            current_window = torch.cat((current_window[:, 1:, :], last_row), dim=1)
    return forecasts

# 1. <deleted>
# 2. DATA HELPERS
def merge_records_to_context(glucose_df, records_df):
    merged_df = glucose_df.copy()
    minutes_into_day = merged_df['time'].dt.hour * 60 + merged_df['time'].dt.minute
    merged_df['sin_time'] = np.sin(2 * np.pi * minutes_into_day / 1440)
    merged_df['cos_time'] = np.cos(2 * np.pi * minutes_into_day / 1440)

    if records_df is not None and not records_df.empty:
        pivoted = records_df.pivot_table(index='time', columns='record_type',
                                         values='record_amount', aggfunc='sum').reset_index()
        merged_df = pd.merge_asof(merged_df.sort_values('time'), pivoted.sort_values('time'),
                                  on='time', direction='backward', tolerance=pd.Timedelta("5m"))

    for col in ['insulin', 'meal', 'minimeal', 'karo']:
        if col not in merged_df.columns:
            merged_df[col] = 0.0
    return merged_df.fillna(0.0)


def prepare_tensor(df):
    features = ['glucose', 'insulin', 'meal', 'minimeal', 'karo', 'sin_time', 'cos_time']
    df_norm = df.copy()
    df_norm['glucose'] = (df_norm['glucose'] - 50.0) / 350.0
    df_norm['insulin'] = df_norm['insulin'] / 12.0
    df_norm['karo'] = df_norm['karo'] / 2.0
    df_norm['minimeal'] = df_norm['minimeal'] / 2.0
    df_norm['meal'] = df_norm['meal'] / 50.0
    data_array = df_norm[features].values.astype(np.float32)
    return torch.tensor(data_array).unsqueeze(0)


# 3. MAIN PREDICT FUNCTION
def Predict(prediction_time=None, device_arg=None, validate=False):
    print("--- Starting Predict() ---")
    device = torch.device(device_arg if device_arg else ('cuda' if torch.cuda.is_available() else 'cpu'))

    # Validation Guardrails
    if validate:
        p_time = pd.to_datetime(prediction_time)
        if p_time < datetime.now() - timedelta(hours=12):
            print("🛑 Error: Cannot validate. API only provides the last 12 hours.")
            return
        if p_time > datetime.now() - timedelta(hours=2):
            print("🛑 Error: Validation requires 2 hours of 'actual' data. Too soon!")
            return

    # Load Model
    model = PoppyLSTMModel(input_size=7, hidden_size=64, num_layers=2).to(device)
    # Force the output layer to 1 to match our 'Physiology' training
    model.fc = torch.nn.Linear(64, 1)
    search_pattern = os.path.join("src", "poppy_model_*.pth*")
    list_of_files = glob.glob(search_pattern) or glob.glob("poppy_model_*.pth*")

    if not list_of_files:
        print("🛑 No model file found.")
        return

    latest_model = max(list_of_files, key=os.path.getctime)
    print(f"Loading weights from: {latest_model}")

    # map_location='cpu' forces the GPU weights to be moved to system RAM
    model.load_state_dict(torch.load(latest_model, map_location=torch.device('cpu')))

    # One final push to make sure EVERY layer (including convolutions) is on CPU
    model.to(torch.device('cpu'))
    model.eval()

    # Data Fetching
    # 1. Establish the "Anchor" - use prediction_time if provided, else use current time
    anchor_time = pd.Timestamp(prediction_time) if prediction_time else pd.Timestamp.now()

    # 2. Fetch glucose data and filter it so we don't "see the future"
    print(f"Step 3: Fetching glucose data relative to {anchor_time}...")
    glucose_df = get_glucose_data_from_api(email='markalavin@gmail.com', password='my$Poppy$Dog1')

    # Filter glucose_df to only include readings UP TO the anchor_time
    # We keep 96 samples (8 hours) for the plot's blue line,
    # but we will still only feed 36 to the model later.
    glucose_df = glucose_df[glucose_df['time'] <= anchor_time].tail(96)

    print(f"Step 5: Fetching recent records leading up to {anchor_time}...")
    # Pass anchor_time here so the console prompt asks about the correct 6-hour window
    records_df = get_recent_records(anchor_time)

    # Step 6: Merging
    print("Step 6: Merging data streams...")
    merged_df = merge_records_to_context(glucose_df, records_df)

    # Step 7: Preparing input tensor
    # Ensure this line is NOT indented inside an 'if' or 'try' block
    # unless Step 8 is also inside that same block.
    print("Step 7: Preparing input tensor...")

    # We use .tail(INPUT_SAMPLES) to get the exact window needed
    input_tensor = prepare_tensor(merged_df.tail(INPUT_SAMPLES)).cpu()

    # Step 8: Running Inference (Autoregressive)
    print(f"Step 8: Generating {PREDICTION_SAMPLES}-step forecast...")

    # 1. Run the loop FIRST to get the scaled values
    forecast_scaled = get_autoregressive_forecast(model, input_tensor, PREDICTION_SAMPLES)

    # 2. THEN convert those values to mg/dL
    forecast_array = (np.array(forecast_scaled) * 350.0) + 50.0
    # 1. Log the results (Passing the whole 2-hour array)
    log_prediction(current_glucose=merged_df['glucose'].iloc[-1], forecast_array=forecast_array)

    # 2. Validation (Checking the full 2-hour forecast against ground truth)
    actuals_results = None
    if validate:
        print("Step 9: Validating 2-hour forecast against actual CGM data...")
        actuals_results = validate_prediction(prediction_time, forecast_array,
                                              email='markalavin@gmail.com',
                                              password='my$Poppy$Dog1')

    # 3. Final Visualization
    plot_forecast(merged_df, forecast_array, model, actuals_results)

def plot_forecast(recent_df, forecast_array, model, actuals_results=None):
    plt.figure(figsize=(12, 6))
    last_time = recent_df['time'].iloc[-1]
    forecast_times = [last_time + pd.Timedelta(minutes=5 * (i + 1)) for i in range(len(forecast_array))]

    plt.plot(recent_df['time'], recent_df['glucose'], label='Actual History', color='blue', marker='o', markersize=3)
    plt.plot(forecast_times, forecast_array, label='LSTM Forecast', color='red', linestyle='--', marker='s',
             markersize=3)

    if actuals_results:
        plt.plot(actuals_results['times'], actuals_results['actuals'],
                 label='Actual CGM (Ground Truth)', color='green', linestyle='-', marker='x', markersize=4)
        plt.title(f"Poppy's Forecast vs Actuals\nRMSE: {actuals_results['rmse']:.2f} mg/dL")
    else:
        plt.title(f"Poppy's Glucose Forecast\nReading at: {last_time.strftime('%H:%M')}")

    plt.axhline(y=CGM_MIN, color='green', linestyle=':', alpha=0.5)
    plt.axhline(y=CGM_MAX, color='orange', linestyle=':', alpha=0.5)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Temporary Weight Inspector
    # Access the weight of the insulin_conv layer specifically
    with torch.no_grad():
        insulin_weights = model.phys_layer.insulin_conv.weight.cpu().numpy().flatten()
        print(f"\nLEARNED INSULIN CURVE (First 24): {insulin_weights[:24].round(3).tolist()}")
    plt.show()

    # --- NEW: Text Visualization for Gemini ---
    print("\n" + "=" * 30)
    print("DATA SUMMARY FOR VALIDATION")
    print("=" * 30)

    # 1. Blue: Last 10 points of history (to see the entry trend)
    history_vals = recent_df['glucose'].tail(10).values.round(1)
    print(f"BLUE (History - Last 10): {history_vals.tolist()}")

    # 2. Red: The 120-minute forecast
    print(f"RED  (Forecast):           {forecast_array.round(1).tolist()}")

    # 3. Green: The actuals (if available)
    if actuals_results is not None:
        actual_vals = actuals_results['actuals'].round(1)
        print(f"GREEN (Actuals):            {actual_vals.tolist()}")

        # Calculate point-by-point error for context
        min_len = min(len(forecast_array), len(actual_vals))
        errors = (forecast_array[:min_len] - actual_vals[:min_len]).round(1)
        print(f"ERROR (Red minus Green):   {errors.tolist()}")

    print("=" * 30 + "\n")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", type=str)
    parser.add_argument("--device", type=str)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    final_time = time_spec(args.time)
    Predict(prediction_time=final_time, device_arg=args.device, validate=args.validate )