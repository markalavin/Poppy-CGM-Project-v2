import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
import warnings

# Suppress the Dtype and Timestamp warnings for a cleaner console
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def load_cgm_data(file_path):
    if not os.path.exists(file_path):
        return None

    header_idx = 0
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if "Device Timestamp" in line:
                header_idx = i
                break

    df = pd.read_csv(file_path, skiprows=header_idx)
    df.columns = df.columns.str.strip()

    # Explicitly handle MM-DD-YYYY format for consistency
    df['Timestamp'] = pd.to_datetime(df['Device Timestamp'], format='%m-%d-%Y %I:%M %p', errors='coerce')

    # Merge Historic and Scan glucose
    df['Glucose_Val'] = pd.to_numeric(df['Historic Glucose mg/dL'], errors='coerce')
    df['Glucose_Val'] = df['Glucose_Val'].fillna(pd.to_numeric(df['Scan Glucose mg/dL'], errors='coerce'))

    return df.dropna(subset=['Timestamp', 'Glucose_Val'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("start")
    parser.add_argument("end")
    args = parser.parse_args()

    start_dt = pd.to_datetime(args.start)
    end_dt = pd.to_datetime(args.end)

    src = os.path.join('..', 'data', 'raw_LibreView_export.csv')
    df = load_cgm_data(src)

    mask = (df['Timestamp'] >= start_dt) & (df['Timestamp'] <= end_dt)
    view = df.loc[mask].sort_values('Timestamp')

    plt.figure(figsize=(14, 7))
    plt.plot(view['Timestamp'], view['Glucose_Val'], color='#1f77b4', linewidth=2, label='Glucose (mg/dL)')

    # --- MARK CRITICAL EVENT: 9:10 PM ---
    event_time = pd.to_datetime("2026-01-18 21:10")
    if start_dt <= event_time <= end_dt:
        # Draw a vertical line for the intervention
        plt.axvline(x=event_time, color='red', linestyle='--', alpha=0.7)
        plt.text(event_time, view['Glucose_Val'].max(), ' Karo + Minimeal',
                 color='red', fontweight='bold', verticalalignment='bottom')

        # Highlight the "Recovery Zone" (next 2 hours)
        recovery_end = event_time + pd.Timedelta(hours=2)
        plt.axvspan(event_time, recovery_end, color='green', alpha=0.1, label='Recovery Phase')

    plt.title(f"Poppy's Hypo Recovery: Jan 18th Intervention")
    plt.ylabel("Glucose (mg/dL)")
    plt.xlabel("Time")
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()