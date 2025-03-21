import logging
import datetime
import os
import json
from src.constants import LOG_DIR, CONFIG_JSON

def setup_logger():
    try:
        with open(CONFIG_JSON, "r") as f:
            config = json.load(f)
        log_level = getattr(logging, config.get("logging", {}).get("level", "INFO"))
    except Exception:
        log_level = logging.INFO

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    log_filename = os.path.join(LOG_DIR, f"app_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    return logger
