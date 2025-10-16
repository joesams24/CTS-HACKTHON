import os
from scripts.utils.helpers import setup_logging
from lstm_model import train_lstm
from xgb_model import train_xgb

SCRIPT_DIR = os.path.dirname(__file__)
RAW_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../data/processed"))
STATIC_FILE = os.path.join(RAW_DIR, "static_features.csv")
LOG_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../logs"))
os.makedirs(LOG_DIR, exist_ok=True)
logger = setup_logging(LOG_DIR, "train_pipeline")

WINDOWS = [30, 60, 90]

for window in WINDOWS:
    logger.info(f"=== Starting {window}-day pipeline ===")
    seq_file = os.path.join(RAW_DIR, f"sequences_{window}.csv")
    lstm_model_file = os.path.join(SCRIPT_DIR, f"../../models/lstm/lstm_{window}.pth")
    xgb_model_file = os.path.join(SCRIPT_DIR, f"../../models/xgb/xgb_{window}.joblib")
    target_file = os.path.join(RAW_DIR, f"target_{window}.csv")  # prepare targets

    emb_file = train_lstm(seq_file, lstm_model_file, logger=logger)
    train_xgb(STATIC_FILE, emb_file, target_file, xgb_model_file, logger=logger)

logger.info("All pipelines completed successfully!")
