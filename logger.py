import logging
import datetime


def setup_logger(logger_name, log_file=None, level=logging.INFO):
    if log_file is None:
        current_datetime = datetime.datetime.now()
        log_file = current_datetime.strftime("%Y-%m-%d_%H-%M-%S.log")
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)


def get_logger(name):
    return logging.getLogger(name)
