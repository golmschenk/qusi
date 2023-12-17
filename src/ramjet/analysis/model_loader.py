"""
Code for loading trained models.
"""
import re
import datetime
from pathlib import Path
from typing import Union


def get_latest_log_directory(logs_directory: Union[Path, str]) -> Path:
    """
    Gets the most recent log directory in a root logs directory based on the datetime in the log name.

    :param logs_directory: The root logs directory containing the individual log directories.
    :return: The string path to the most recent log directory.
    """
    logs_directory = Path(logs_directory)
    log_directories = [path for path in Path(logs_directory).glob('*') if path.is_dir()]
    latest_log_directory = None
    latest_log_datetime = datetime.datetime.min
    for log_directory in log_directories:
        match = re.search(r'(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})', str(log_directory))
        if match is not None:
            log_datetime = datetime.datetime.strptime(match.group(1), '%Y-%m-%d-%H-%M-%S')
            if log_datetime > latest_log_datetime:
                latest_log_directory = log_directory
                latest_log_datetime = log_datetime
    if latest_log_directory is None:
        raise FileNotFoundError(f'No logs with datetime names found in {logs_directory}')
    else:
        return latest_log_directory
