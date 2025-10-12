import pickle
from datetime import datetime
import os

# Create logs folder outside backend if it doesn't exist
LOGS_PATH = "../../logs"
os.makedirs(LOGS_PATH, exist_ok=True)

# Define separate log files
PREPROCESS_LOG_FILE = os.path.join(LOGS_PATH, "preprocessing.log")
DATA_COLLECTION_LOG_FILE = os.path.join(LOGS_PATH, "data_collection.log")

def log_message(message, log_file=None):
    """Prints message and writes to the specified log file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    print(full_message)
    
    # Choose which log file to write
    if log_file is None:
        log_file = PREPROCESS_LOG_FILE  # default
    with open(log_file, "a") as f:
        f.write(full_message + "\n")

def save_pickle(obj, path):
    """Save object as pickle and log it"""
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    log_message(f"Saved pickle file: {path}")