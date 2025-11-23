import logging
import os


def get_logger():
    log_level = os.getenv("level", "INFO")

    # Configure logging based on environment variable
    logging.basicConfig(level=log_level)

    return logging.getLogger(__name__)
