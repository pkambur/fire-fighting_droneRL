import logging
from logging.handlers import RotatingFileHandler

from utils.logging_files import program_logs


# logger = logging.basicConfig(filename=program_logs, level=logging.INFO,
#                              format='%(asctime)s, %(message)s', filemode='w')


def setup_logger(logger_name='logger', log_file=program_logs, level=logging.INFO,
                 max_bytes=10 * 1024 * 1024, file_mode='w'):
    """
    :param file_mode:
    :param logger_name:
    :param log_file:
    :param level:
    :param max_bytes:
    :return:
    """
    logger = logging.getLogger(logger_name)
    if logger.hasHandlers():
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    file_handler = RotatingFileHandler(log_file, maxBytes=max_bytes, mode=file_mode)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # console_handler = logging.StreamHandler()
    # console_handler.setFormatter(formatter)
    # logger.addHandler(console_handler)

    return logger
