import logging
import sys


def setup_basic_logger(name, root_level=logging.WARN, name_level=logging.INFO):
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
    if not logging.getLogger().handlers:
        logging.getLogger().setLevel(root_level)
        logging.getLogger().addHandler(handler)
    if not logging.getLogger(name).handlers:
        logging.getLogger(name).setLevel(name_level)
        logging.getLogger(name).propagate = False
        logging.getLogger(name).addHandler(handler)


def setup_advanced_logger(name, root_level=logging.INFO, name_level=logging.INFO):
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        '%(levelname)s|%(asctime)s - %(funcName)s:%(lineno)d - %(message)s', '%H:%M:%S'))
    if not logging.getLogger().handlers:
        logging.getLogger().setLevel(root_level)
        logging.getLogger().addHandler(handler)
    if not logging.getLogger(name).handlers:
        logging.getLogger(name).setLevel(name_level)
        logging.getLogger(name).propagate = False
        logging.getLogger(name).addHandler(handler)