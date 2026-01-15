# Application_Parameters.py

# Application version for logging and tracking
APP_VERSION = "1.0.2"  # Increment this when you change architecture or training logic

DATA_INTERVAL_MINUTES = 5

# Input: 6 hours of history
INPUT_WINDOW_HOURS = 6
INPUT_SAMPLES = (INPUT_WINDOW_HOURS * 60) // DATA_INTERVAL_MINUTES # 72

# Output: 2 hours of prediction
PREDICTION_WINDOW_HOURS = 2
PREDICTION_SAMPLES = (PREDICTION_WINDOW_HOURS * 60) // DATA_INTERVAL_MINUTES # 24

# Model Architecture
HIDDEN_SIZE = 64
NUM_LAYERS = 2

# Canine-specific glucose targets
LOW_GLUCOSE_THRESHOLD = 150
HIGH_GLUCOSE_THRESHOLD = 250

# Parameters specifying range of readings from Libre 3:
GLUCOSE_LOWER_LIMIT = 50
GLUCOSE_UPPER_LIMIT = 400

# Parameters for gathering Prediction histories
PREDICTION_HISTORY_FILE = '../data/prediction_history.csv'
