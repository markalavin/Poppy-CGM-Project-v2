# Compare Predicted glucose levels against subsequent "actuals"
#################################################################################################

import pandas as pd
import numpy as np
from datetime import timedelta
from pylibrelinkup import PyLibreLinkUp


def get_recent_actuals(email, password):
    try:
        # 1. Initialize with email and password
        client = PyLibreLinkUp(email, password)

        # 2. Use .authenticate() instead of .login()
        client.authenticate()

        # 3. Get the patient list
        patients = client.get_patients()
        if not patients:
            print("❌ No patients found. Check if you have an active Connection.")
            return None

        patient = patients[0]
        # 4. Fetch the graph (last 12 hours)
        recent_data = client.graph(patient)

        return pd.DataFrame([{
            'time': pd.to_datetime(m.timestamp),
            'glucose': m.value
        } for m in recent_data])

    except Exception as e:
        print(f"❌ API Error: {e}")
        return None

def validate_prediction(prediction_time_str, predicted_values, email, password):
    df_actuals = get_recent_actuals(email, password)
    if df_actuals is None or df_actuals.empty:
        return None

    start_time = pd.to_datetime(prediction_time_str)
    end_time = start_time + timedelta(hours=2)

    # Extract actuals for the 2-hour window
    actual_segment = df_actuals[(df_actuals['time'] > start_time) & (df_actuals['time'] <= end_time)].copy()

    if len(actual_segment) < 5:
        print("🛑 Not enough actual data points found in the API's 12-hour window.")
        return None

    # Align lengths for RMSE
    min_len = min(len(actual_segment), len(predicted_values))
    actuals = actual_segment['glucose'].values[:min_len]
    preds = predicted_values[:min_len]

    rmse = np.sqrt(np.mean((preds - actuals) ** 2))
    print(f"✅ Validation Complete. RMSE: {rmse:.2f} mg/dL")

    return {'times': actual_segment['time'].values[:min_len], 'actuals': actuals, 'rmse': rmse}