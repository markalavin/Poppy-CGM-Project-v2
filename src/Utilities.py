# Utility functions used throughout Poppy-CGM-Project:
################################################################################################

from datetime import datetime, timedelta

def time_spec(input_str):
    """
    Converts a string into a standard YYYY-MM-DD HH:MM format.
    Supports:
      - '15' or '15m' (15 minutes ago)
      - '2h' (2 hours ago)
      - 'YYYY-MM-DD HH:MM' (Direct timestamp)
    """
    if not input_str:
        return None

    input_str = input_str.strip().lower()
    now = datetime.now()

    try:
        # Handle "hours ago" (e.g., "2h")
        if input_str.endswith('h'):
            hours = int(input_str.rstrip('h'))
            return (now - timedelta(hours=hours)).strftime('%Y-%m-%d %H:%M')

        # Handle "minutes ago" (e.g., "15" or "15m")
        # Strips 'm' if present, then checks if the rest is a number
        minutes_str = input_str.rstrip('m')
        if minutes_str.isdigit():
            minutes = int(minutes_str)
            return (now - timedelta(minutes=minutes)).strftime('%Y-%m-%d %H:%M')

        # Default to assuming it's a timestamp
        return input_str
    except ValueError:
        print(f"Warning: Could not parse time '{input_str}'. Using raw input.")
        return input_str