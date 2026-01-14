from Process_CGM_Data import processLibreViewCSV
from Process_Report_Data import processLibreReportsCSV
import pandas as pd
import numpy as np

# Return a single dataframe that represents the merging of the CGM data in dataframes
# "cgm_df" and "reports_df"

def merge_poppy_data(cgm_df, reports_df):
    # 1. Ensure both are sorted (merge_asof requires this)
    cgm_df = cgm_df.sort_values('Timestamp')
    reports_df = reports_df.sort_values('timestamp')

    # 2. Perform the 'as-of' merge
    # This aligns the report to the closest CGM timestamp
    merged = pd.merge_asof(
        cgm_df,
        reports_df,
        left_on='Timestamp',
        right_on='timestamp',
        direction='nearest',
        tolerance=pd.Timedelta('8 minutes')
    )

    # 3. Cleanup
    # Drop the extra timestamp column from the reports
    merged.drop(columns=['timestamp'], inplace=True)

    # Fill gaps with 0 (for all the times Poppy didn't eat/get insulin)
    # We only fill the report columns, not the Glucose column!
    report_cols = reports_df.columns.drop('timestamp')
    merged[report_cols] = merged[report_cols].fillna(0)
    return merged

if __name__ == "__main__":

    poppy_cgm_csv = r"C:\Users\marka\PycharmProjects\Poppy CGM Project v1\Data\Poppy CGM.csv"
    poppy_cgm_df = processLibreViewCSV( poppy_cgm_csv )
    poppy_report_csv = r"C:\Users\marka\PycharmProjects\Poppy CGM Project v1\Data\Poppy Reports.csv"
    poppy_report_df = processLibreReportsCSV( poppy_report_csv )

    poppy_data_df = merge_poppy_data(poppy_cgm_df, poppy_report_df)
    print( poppy_data_df )