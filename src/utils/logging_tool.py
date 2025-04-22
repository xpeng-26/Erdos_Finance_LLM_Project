import logging


def initialize_logger(log_path, log_file):
    """Initialize logger with file handler and formatter.

    Args:
        log_path (str): Path to the log directory
        log_file (str): Name of the log file

    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create log file
    log_file = f"{log_path}/{log_file}"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    return logger
