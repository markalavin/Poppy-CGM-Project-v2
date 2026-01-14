from Merge_Poppy_Data import merge_poppy_data
from Process_CGM_Data import processLibreViewCSV
from Process_Report_Data import processLibreReportsCSV
from Create_Windows import create_windows
import pandas as pd
import numpy as np

# Returns the X and y tensors that represent the training data in the merger of the dataframes
# "poppy_cgm_df" and "poppy_report_df"

def construct_input_tensors():

    poppy_cgm_csv = r"..\data\Poppy CGM.csv"
    poppy_cgm_df = processLibreViewCSV( poppy_cgm_csv )
    poppy_report_csv = r"..\data\Poppy Reports.csv"
    poppy_report_df = processLibreReportsCSV( poppy_report_csv )
    poppy_data_df = merge_poppy_data(poppy_cgm_df, poppy_report_df)

    print( "poppy_data_df:\n", poppy_data_df )

    # Assuming poppy_data_df is your merged dataframe
    glucose_col = 'Historic Glucose mg/dL'

    # 1. Get the core statistics
    stats = poppy_data_df[glucose_col].agg(['min', 'max', 'mean', 'std'])

    print("--- Poppy's Glucose Statistics ---")
    print(f"Minimum Reading: {stats['min']:.1f} mg/dL")
    print(f"Maximum Reading: {stats['max']:.1f} mg/dL")
    print(f"Average:         {stats['mean']:.1f} mg/dL")
    print(f"Std Dev:         {stats['std']:.1f}")

    # 2. Check for "Out of Bounds" readings relative to the Libre 3 specs
    low_count = (poppy_data_df[glucose_col] <= 50).sum()
    high_count = (poppy_data_df[glucose_col] >= 400).sum()

    print(f"\nReadings at/below 50 (Low Floor):  {low_count}")
    print(f"Readings at/above 400 (High Ceiling): {high_count}")

    X, y = create_windows(poppy_data_df, window_size=48, lead_time=12)

    return X, y


# Normalize the feature values in tensors "X" and "y" so that they all lie in the
# range 0 - 1 and return the normalized tensors:

def normalize_tensors(X, y):
    # Create a clone so we don't modify the originals by accident
    X_norm = X.clone()
    y_norm = y.clone()

    # Column 0 is Glucose: (Value - Min) / (Max - Min)
    X_norm[:, :, 0] = (X_norm[:, :, 0] - 50.0) / (400.0 - 50.0)

    # Column 1 is Insulin: Value / Max_Dose
    X_norm[:, :, 1] = X_norm[:, :, 1] / 12.0

    # Normalize y (Target Glucose) using the SAME glucose scale
    y_norm = (y_norm - 50.0) / (400.0 - 50.0)

    return X_norm, y_norm

def check_input_tensors( X, y ):

    # 1. Check the Shapes
    print(f"Total windows: {X.shape[0]}")
    print(f"Each window shape: {X.shape[1]} samples x {X.shape[2]} features")

    # 2. Inspect the first window (Window 0)
    # We move it back to CPU and convert to a NumPy array for easy reading
    first_window = X[0].cpu().numpy()
    first_target = y[0].cpu().numpy()

    # 3. Create a small temporary DataFrame to look at the first window
    feature_names = ['Glucose', 'Insulin', 'Meal', 'Minimeal', 'Karo', 'Sin_T', 'Cos_T']
    window_df = pd.DataFrame(first_window, columns=feature_names)

    print("\n--- check_input_tensors: FIRST WINDOW (First 5 rows) ---")
    print(window_df.head())

    print("\n--- check_input_tensors: LAST WINDOW (Last 5 rows) ---")
    print(window_df.tail())

    print(f"\ncheck_input_tensors: Target Glucose (1 hour after window ends): {first_target}")

if __name__ == "__main__":
    X, y = construct_input_tensors()
    check_input_tensors(X, y)
