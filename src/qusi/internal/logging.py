from __future__ import annotations

import datetime
import logging
import re
import sys

import stringcase

logger_initialized = False


def create_default_formatter() -> logging.Formatter:
    formatter = logging.Formatter('qusi [{asctime} {levelname} {name}] {message}', style='{')
    return formatter


def set_up_default_logger():
    global logger_initialized  # noqa PLW0603 : TODO: Probably a bad hack. Consider further.
    if not logger_initialized:
        formatter = create_default_formatter()
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger = logging.getLogger('qusi')
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        sys.excepthook = excepthook
        logger_initialized = True


def excepthook(exc_type, exc_value, exc_traceback):
    logger = logging.getLogger('qusi')
    logger.critical(f'Uncaught exception at {datetime.datetime.now().astimezone()}:')
    logger.handlers[0].flush()
    sys.__excepthook__(exc_type, exc_value, exc_traceback)


def get_metric_name(metric_function):
    metric_name = type(metric_function).__name__
    return metric_name


def camel_case_acronyms(string: str) -> str:
    def camel_case_single_acronym(string: str | None) -> str:
        if string is None:
            return ''
        return stringcase.capitalcase(string.lower())

    return re.sub(
        r'([A-Z]{2,})([A-Z][a-z])|([A-Z]{2,})',
        lambda match: ''.join(map(camel_case_single_acronym, [match.group(1), match.group(2), match.group(3)])),
        string
    )
