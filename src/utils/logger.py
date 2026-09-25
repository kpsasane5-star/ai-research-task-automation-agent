import logging
import sys
from rich.logging import RichHandler

def setup_logger(name: str = "agent", level: int = logging.INFO) -> logging.Logger:
    """Set up structured logger using Rich or standard stream handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        try:
            handler = RichHandler(rich_tracebacks=True, show_time=True, show_path=False)
        except Exception:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")
            handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = setup_logger()
