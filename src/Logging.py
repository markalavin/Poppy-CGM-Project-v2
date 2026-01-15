import csv
import os
from datetime import datetime
from Application_Parameters import APP_VERSION, PREDICTION_SAMPLES, PREDICTION_HISTORY_FILE


def log_prediction(current_glucose, forecast_array, insulin=None, carbs=None):
    """
    Logs a single prediction event to the CSV history.
    """
    file_exists = os.path.isfile(PREDICTION_HISTORY_FILE)

    # 1. Determine the next Serial Index
    serial_index = 1
    if file_exists:
        with open(PREDICTION_HISTORY_FILE, 'r') as f:
            # Efficiently get the last line to find the last index
            last_line = f.readlines()[-1]
            if not last_line.startswith('serial'):  # Check if it's just the header
                serial_index = int(last_line.split(',')[0]) + 1

    # 2. Build the Header (if file is new)
    header = [
        'serial', 'version', 'timestamp', 'current_glucose', 'insulin_units', 'carbs_grams'
    ]
    # Add 24 columns for the forecast: +5min, +10min, etc.
    header += [f'plus_{5 * (i + 1)}min' for i in range(PREDICTION_SAMPLES)]

    # 3. Prepare the Data Row
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    row = [serial_index, APP_VERSION, timestamp, current_glucose, insulin, carbs]
    row += list(forecast_array)

    # 4. Append to CSV
    with open(PREDICTION_HISTORY_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)

    print(f"✅ Prediction #{serial_index} (v{APP_VERSION}) logged to CSV.")