import logging
import sys
from datetime import datetime
from pathlib import Path
from src.config import get_settings

def setup_logger(name: str) -> logging.Logger:
    settings = get_settings()
    logger = logging.getLogger(name)

    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler: best-effort — silently skipped in Lambda/read-only filesystems.
    try:
        settings.logs_dir.mkdir(parents=True, exist_ok=True)
        log_filename = f"app_{datetime.now().strftime('%Y%m%d')}.log"
        log_path = settings.logs_dir / log_filename
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception:
        pass

    return logger
