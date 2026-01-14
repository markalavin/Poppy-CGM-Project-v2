import pandas as pd

def processLibreReportsCSV(csv_filename):
    # Load your new CSV
    df = pd.read_csv(csv_filename)

    # Convert text timestamps to actual Python datetime objects
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Pivot the data: This turns 'INSULIN' and 'MEAL' into column headers
    # aggfunc='sum' ensures that if you logged two things at once, they stay together
    df_pivoted = df.pivot_table(
        index='timestamp',
        columns='report_type',
        values='report_measure',
        aggfunc='sum'
    ).fillna(0)  # Fill empty spots with 0

    # Reset index to bring 'timestamp' back as a normal column
    return df_pivoted.reset_index()


if __name__ == "__main__":
    # Update the path to wherever your file is
    report_path = r"C:\Users\marka\PycharmProjects\Poppy CGM Project v1\venv\Data\Poppy Reports.csv"
    reports_df = processLibreReportsCSV(report_path)

    print("Successfully flattened reports!")
    print(f"Dimensions: {reports_df.shape}")
    print( reports_df )

    is_ordered = reports_df['timestamp'].is_monotonic_increasing
    print(f"Is the data in order? {is_ordered}")