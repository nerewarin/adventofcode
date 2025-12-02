import logging
import os


def get_logger():
    log_level = os.getenv("level", "INFO")

    # Configure logging based on environment variable
    logging.basicConfig(level=log_level)

    return logging.getLogger(__name__)


class MessageOnlyFilter(logging.Filter):
    """Ensures this logger prints only the final message and nothing else."""

    def filter(self, record):
        record.msg = str(record.msg)
        record.args = ()
        return True


def get_message_only_logger(name: str = "grid_printer") -> logging.Logger:
    # Take effective level from root (or from your main logger if you prefer)
    root_level = logging.getLogger().getEffectiveLevel()

    logger = logging.getLogger(name)
    logger.setLevel(root_level)
    logger.propagate = False  # do NOT use root handlers/format

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(root_level)
        handler.setFormatter(logging.Formatter("%(message)s"))  # <-- only message
        logger.addHandler(handler)

    return logger
