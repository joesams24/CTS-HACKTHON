import os
import logging
from datetime import datetime

def setup_logging(log_dir: str, name: str) -> logging.Logger:
    """
    Set up a logging configuration with both file and console handlers.
    Logs will be stored in the given `log_dir`.
    Automatically resolves absolute paths to avoid nested directories.
    """
    # Resolve absolute path relative to project root
    base_dir = os.path.dirname(os.path.abspath(__file__))  # project root
    log_dir = os.path.abspath(os.path.join(base_dir, log_dir))
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)

    # Clear existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")

    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def calculate_risk_tier(score: float) -> str:
    """
    Convert a numeric risk score (0-1) into categorical risk tier.
    Ensures distribution across all 5 tiers.
    """
    if score >= 0.80:
        return "Very High"
    elif score >= 0.60:
        return "High"
    elif score >= 0.40:
        return "Moderate"
    elif score >= 0.20:
        return "Low"
    else:
        return "Minimal"
