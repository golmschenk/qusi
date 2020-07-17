"""
Code to clean up the log directory.
"""
import re
import datetime
import shutil
from pathlib import Path

from tensorflow.python.summary.summary_iterator import summary_iterator


class LogCleaner:
    """
    A class to clean up the log directory.
    """
    @staticmethod
    def delete_old_empty_logs(logs_directory: Path, timedelta: datetime.timedelta = datetime.timedelta(days=1)):
        """
        Removes logs which are more than 24 hours old, and contain less than 2 epochs worth of data.
        This delete cases that crashed immediately, but doesn't delete ones that just started running.

        :param logs_directory: The root logs directory containing folder.
        :param timedelta: The time frame to consider an old file.
        """
        logs_directory = Path(logs_directory)
        log_directories = [path for path in Path(logs_directory).glob('*') if path.is_dir()]
        for log_directory in log_directories:
            match = re.search(r'(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})', str(log_directory))
            log_datetime = datetime.datetime.strptime(match.group(1), '%Y-%m-%d-%H-%M-%S')
            if log_datetime > (datetime.datetime.now() - timedelta):
                continue
            event_paths = [path for path in Path(log_directory).glob('**/events.out.tfevents.*')]
            keep_event_file = False
            for event_path in event_paths:
                for summary in summary_iterator(str(event_path)):
                    if summary.step > 0:
                        keep_event_file = True
            if not keep_event_file:
                shutil.rmtree(log_directory)


if __name__ == '__main__':
    LogCleaner.delete_old_empty_logs(Path('logs'))
