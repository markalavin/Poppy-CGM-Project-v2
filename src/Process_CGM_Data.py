# Process CGM Data.py:  Handle CSVs of Glucose Level and Reports
##############################################################################################################

import pandas as pd

# Return Pandas dataframe corresponding to CSV file provided by LibreView:
# Columns are named by the header record:
def processLibreViewCSV( csv_filename ):
    print( "csv_filename", csv_filename )
    df = pd.read_csv( csv_filename, header = 1 )   # Skip 0-th line
    print( f"There are {df.shape} rows, cols in {csv_filename}")
    df = df[ df['Record Type'] == 0 ].copy()
    print( f"After filtering, there are {df.shape} rows, cols in {csv_filename}")
    print( df[ 0 : 10 ] )
    # 1. Rename 'Device Timestamp' to 'Timestamp' so the rest of your code works
    df.rename(columns={'Device Timestamp': 'Timestamp'}, inplace=True)

    # 2. Convert that column from "Text" to "Actual Date/Time" objects
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%m-%d-%Y %I:%M %p')

    # Filter for Historic Glucose (Record Type 0)
    df = df[df['Record Type'] == 0].copy()

    # Drop rows where glucose is missing (NaN) to prevent plotting errors
    df.dropna(subset=['Historic Glucose mg/dL'], inplace=True)
    return df

# Return Pandas dataframe corresponding to CSV file of meal and injection reports.
# The file consists of lines with the following format:
#     timestamp
#     report_type  { MEAL, KARO, MINIMEAL, INSULIN }
#     report_measure { e.g., 9 units, usual_meal, Karo syrup, etc. }

# Per suggestion of Google Gemini, represent the hand-recorded "reports"
# (e.g., food, insulin) by a separate CSV file with these headers:
#    timestamp,report_type,report_measure
#    2024-05-20 07:00:00,INSULIN,9
#    2024-05-20 07:05:00,MEAL,1.0
#    2024-05-20 10:30:00,KARO,1.0
#    2024-05-20 16:00:00,MINIMEAL,0.5

#define INSULIN_MEASURE 9
#define MEAL_MEASURE    1.0
#define KARO_MEASURE    1.0
#define MINI_MEAL_MEASURE 0.5

def processLibreReportsCSV( csv_filename ):
    # Load the manual notebook data
    df = pd.read_csv(csv_filename)

    # Ensure the timestamp is in the correct format
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Sort by time to ensure merge_asof works correctly later
    df = df.sort_values('timestamp')

    return df

def mergeCSVs( df1, df2 ):
    pass


import matplotlib.pyplot as plt


def plot_glucose(df):
    # Set the 'Timestamp' as the index so it appears on the x-axis
    df.set_index('Timestamp')['Historic Glucose mg/dL'].plot(figsize=(12, 6))

    plt.title("Poppy's Glucose Curve")
    plt.ylabel("Glucose (mg/dL)")
    plt.xlabel("Time")
    plt.grid(True)
    # plt.show()

def histogramDataGaps( df ):
    # 1. Sort by time and calculate the difference between each row
    gaps = df['Timestamp'].sort_values().diff()

    # 2. Convert those timedeltas into a number (minutes is usually best for CGM)
    gap_minutes = gaps.dt.total_seconds() / 60

    # 2.5  Filter out the gap_minutes that have value 5, which indicates no real gap
    # Only look at gaps larger than 20 minutes
    real_gaps = gap_minutes[gap_minutes > 5]

    plt.figure(figsize=(10, 6))
    real_gaps.hist(bins=30, color='orange', edgecolor='black')
    plt.title("Distribution of Significant Data Gaps (>5 mins)")
    plt.xlabel("Gap Duration (Minutes)")
    plt.ylabel("Frequency")


if __name__ == "__main__":
    df = processLibreViewCSV( r"C:\Users\marka\PycharmProjects\Poppy CGM Project v1\venv\Data\Poppy CGM.csv" )
    plot_glucose( df )
    histogramDataGaps( df )
    plt.show()
