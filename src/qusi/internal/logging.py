import datetime
import logging
import sys

logger_initialized = False


def create_default_formatter() -> logging.Formatter:
    formatter = logging.Formatter("qusi [{asctime} {levelname} {name}] {message}", style="{")
    return formatter


def set_up_default_logger():
    global logger_initialized  # noqa PLW0603 : TODO: Probably a bad hack. Consider further.
    if not logger_initialized:
        formatter = create_default_formatter()
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger = logging.getLogger("qusi")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        sys.excepthook = excepthook
        logger_initialized = True


def excepthook(exc_type, exc_value, exc_traceback):
    logger = logging.getLogger("qusi")
    logger.critical(f"Uncaught exception at {datetime.datetime.now().astimezone()}:")
    logger.handlers[0].flush()
    sys.__excepthook__(exc_type, exc_value, exc_traceback)
