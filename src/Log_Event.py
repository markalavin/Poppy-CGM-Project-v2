import pandas as pd
import os
from datetime import datetime, timedelta


def validate_input(prompt, date_format):
    """Repeatedly prompts user until input matches the required format or is empty."""
    while True:
        user_input = input(prompt)
        if not user_input:
            return None
        try:
            # Enforces strict format (e.g., 01 instead of 1)
            datetime.strptime(user_input, date_format)
            return user_input
        except ValueError:
            print(f"  [!] Format Error. Please use exactly {date_format}")


def log_event():
    # Path to your Record data
    file_path = os.path.join('..', 'data', 'Poppy Reports.csv')

    print("\n--- Poppy Event Logger & Auditor ---")

    # --- 1. AUDIT PHASE ---
    if os.path.exists(file_path):
        try:
            df_audit = pd.read_csv(file_path)
            # Find the first column (should be Timestamp)
            ts_col = df_audit.columns[0]
            df_audit[ts_col] = pd.to_datetime(df_audit[ts_col])
            last_entry = df_audit[ts_col].max()

            gap = datetime.now() - last_entry
            if gap > timedelta(hours=12):
                print(f"[!] AUDIT ALERT: It has been {gap.total_seconds() // 3600:.1f} hours since the last record.")
            else:
                print(f"Coverage Status: OK. Last record was {gap.total_seconds() // 60:.0f} mins ago.")
        except Exception as e:
            print(f"Audit Note: New file or header mismatch ({e})")

    # --- 2. INPUT PHASE ---
    print("\n1. Meal | 2. Insulin | 3. Karo | 4. Minimeal")
    choice = input("Select event type (1-4): ")
    event_map = {'1': 'Meal', '2': 'Insulin', '3': 'Karo', '4': 'Minimeal'}
    event_name = event_map.get(choice)

    if not event_name:
        print("Exiting.")
        return

    dose = input(f"Enter {event_name} amount (default 1.0): ") or "1.0"

    # Enforcing strict formatting to avoid "2026-1-18" errors
    time_input = validate_input("Enter time (HH:MM, e.g., 09:10) or Enter for 'Now': ", "%H:%M")
    date_input = validate_input("Enter date (YYYY-MM-DD, e.g., 2026-01-18) or Enter for 'Today': ", "%Y-%m-%d")

    now = datetime.now()
    final_date = date_input if date_input else now.strftime('%Y-%m-%d')
    final_time = time_input if time_input else now.strftime('%H:%M')
    timestamp = f"{final_date} {final_time}:00"

    # --- 3. WRITE PHASE (with CRLF safety) ---
    new_line = f"{timestamp},{event_name},{dose}\n"
    file_exists = os.path.isfile(file_path)

    with open(file_path, 'a+') as f:
        # Check if we need to add a newline to the existing file first
        if file_exists:
            f.seek(0, os.SEEK_END)
            if f.tell() > 0:
                f.seek(f.tell() - 1)
                if f.read(1) != '\n':
                    f.write('\n')
        else:
            # Write header if file doesn't exist
            f.write("Timestamp,Event,Value\n")

        f.write(new_line)

    print(f"\n[SUCCESS] Recorded {event_name} ({dose}) at {timestamp}")


if __name__ == "__main__":
    log_event()