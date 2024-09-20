import logging
import sys

import bodo


def setup_service_logger(name: str = "default-logger", level: int = logging.INFO):
    """Setup service logger that will display logs to stdout

    :param name: Name of service logger
    :param level: Log level
    :return: Service logger object
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    # Create a formatter and set it for the handler
    formatter = logging.Formatter("%(asctime)s [%(levelname)s]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def get_service_logger(name: str = "default-logger"):
    """Get server logger

    :param name: Name of service logger
    :return: Service logger object
    """
    return logging.getLogger(name)


def setup_bodo_logger(file_name: str):
    """Setup Bodo logger that outputs to stderr and to file

    :param file_name: If filename is provided output will be put also in file
    :return: Bodo logger object
    """
    bodo.set_verbose_level(2)
    bodo_logger = bodo.user_logging.get_current_bodo_verbose_logger()

    if not file_name:
        return

    file_handler = logging.FileHandler(file_name, mode="w")
    bodo_logger.addHandler(file_handler)
    return bodo_logger
