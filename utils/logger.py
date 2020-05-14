import logging

from utils.simple_logging import setup_basic_logger


class Logger(object):
    """Adds class property logger for doing class-level logging."""
    @property
    def logger(self):
        """Property for accessing logger."""
        name = self.__class__.__name__
        setup_basic_logger(
            name,
            root_level=getattr(self, 'LOGGER_ROOT_LEVEL', logging.ERROR),
            name_level=getattr(self, 'LOGGER_NAME_LEVEL', logging.INFO)
        )
        return logging.getLogger(name)
