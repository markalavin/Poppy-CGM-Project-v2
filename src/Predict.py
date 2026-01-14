import torch
import numpy as np
import pandas as pd
from Model_Architecture import PoppyLSTM
from pylibrelinkup import PyLibreLinkUp     # API to Libre 3 history

# Returns Tensor (X) representing the merger and normalization of the CGM
# data from "cgm_source" consisting of "context_samples" time intervals
# and the Records (food, insulin, etc.) from "user_records".  That tensor
# is then used to predict future Glucose levels:

import torch
import numpy as np
import pandas as pd
from pylibrelinkup import PyLibreLinkUp


def get_latest_context(email, password, user_inputs=None, context_samples=48):
    """
    Connects to LibreLinkUp, fetches the latest 12h graph,
    and prepares a normalized 48-sample tensor for the model.
    """
    # 1. AUTHENTICATION & FETCH
    client = PyLibreLinkUp(email=email, password=password)
    client.authenticate()

    patient_list = client.get_patients()
    if not patient_list:
        raise Exception("Still no patients found. Double-check you accepted the invite in the app.")

    poppy = patient_list[0]
    measurements = client.graph(patient_identifier=poppy)  # Returns ~last 12 hours

    # 2. CONVERT TO DATAFRAME & CALCULATE TIME FEATURES
    data = []
    for m in measurements:
        ts = pd.to_datetime(m.timestamp)
        # Convert time of day to radians (24 hours = 2*pi)
        seconds_in_day = ts.hour * 3600 + ts.minute * 60 + ts.second
        day_radians = 2 * np.pi * (seconds_in_day / 86400.0)

        data.append({
            'glucose': m.value,
            'insulin': 0.0, 'meal': 0.0, 'minimeal': 0.0, 'karo': 0.0,
            'sin_time': np.sin(day_radians),
            'cos_time': np.cos(day_radians)
        })

    df = pd.DataFrame(data).tail(context_samples).copy()

    # 3. MERGE USER RECORDS (Inject entries into the very last row)
    if user_inputs:
        idx = df.index[-1]
        for key, val in user_inputs.items():
            if key in df.columns: df.at[idx, key] = val

    # 4. NORMALIZATION (Must match your training script exactly)
    df['glucose'] = (df['glucose'] - 50.0) / 350.0
    df['insulin'] = df['insulin'] / 12.0
    df[['meal', 'minimeal', 'karo']] = df[['meal', 'minimeal', 'karo']] / 50.0

    # 5. TENSOR CONVERSION [Batch, TimeSteps, Features]
    features = ['glucose', 'insulin', 'meal', 'minimeal', 'karo', 'sin_time', 'cos_time']
    context_tensor = torch.tensor(df[features].values).float().unsqueeze(0)

    return context_tensor


def run_recurrent_inference(model, context_tensor, steps=12):
    """
    Takes a [1, 48, 7] context and predicts 'steps' into the future.
    """
    model.eval()
    predictions = []
    current_window = context_tensor.clone()  # [1, 48, 7]

    with torch.no_grad():
        for _ in range(steps):
            # 1. Predict the next 5-minute glucose value
            # The model returns a normalized value (e.g., 0.3)
            pred_scaled = model(current_window)
            predictions.append(pred_scaled.item())

            # 2. Prepare the 'Future Row' for the next step
            # We must update the Sine/Cosine features for the new time point
            new_row = current_window[:, -1:, :].clone()  # Copy the last row
            new_row[0, 0, 0] = pred_scaled  # Set predicted glucose

            # Reset Insulin/Carbs to 0 for the 'future' steps
            # (Unless you want to 'what-if' a meal here)
            new_row[0, 0, 1:5] = 0

            # Update Time Features (Shift 5 minutes forward)
            # This prevents the model from getting 'stuck' in the current time
            new_row = update_time_features(new_row)

            # 3. Slide the Window
            # Drop the oldest sample (index 0) and append the new prediction
            current_window = torch.cat((current_window[:, 1:, :], new_row), dim=1)

    return predictions


def update_time_features(row_tensor):
    """
    Helper to advance the Sine/Cosine features by 5 minutes (1/288th of a day).
    """
    # Current Sine/Cosine values
    sin_val = row_tensor[0, 0, 5].item()
    cos_val = row_tensor[0, 0, 6].item()

    # Calculate current angle in radians
    current_angle = np.arctan2(sin_val, cos_val)

    # Advance by 5 minutes (2*pi / 288 steps per day)
    next_angle = current_angle + (2 * np.pi / 288.0)

    row_tensor[0, 0, 5] = np.sin(next_angle)
    row_tensor[0, 0, 6] = np.cos(next_angle)
    return row_tensor

def display_forecast(predictions_scaled):
    """
    DISPLAY/UI:
    Converts decimals back to mg/dL and prints/plots the results.
    """
    print("\n--- Poppy's 60-Minute Forecast ---")
    for i, p in enumerate(predictions_scaled):
        mgdl = (p * 350) + 50
        time_ahead = (i + 1) * 5
        print(f"+{time_ahead} min: {mgdl:.1f} mg/dL")

if __name__ == "__main__":
    email = "markalavin@gmail.com"
    password = "my$Poppy$Dog1"
    context_tensor = get_latest_context(email, password, user_inputs=None, context_samples=48)

    # 1. Initialize the architecture (must match your training setup)
    model = PoppyLSTM(input_size=7)

    # 2. Load the weights from your saved file
    # map_location ensures it works even if you move between CPU and GPU
    model.load_state_dict(torch.load("poppy_model_best.pth", map_location=torch.device('cpu')))

    # 3. Set to evaluation mode
    # This is critical to turn off dropout/batch norm layers for prediction
    model.eval()

    print("Model successfully loaded and ready for testing!")

    predictions = run_recurrent_inference(model, context_tensor, steps=12)

    # Assuming 'results' is your list of 12 normalized predictions
    print("\n--- Poppy's 60-Minute Forecast ---")
    for i, pred in enumerate( predictions ):
        # Reverse the Min-Max scaling
        mgdl = (pred * ( 400 - 50 ) ) + 50

        time_ahead = (i + 1) * 5
        print(f"+{time_ahead} min: {mgdl:.1f} mg/dL")