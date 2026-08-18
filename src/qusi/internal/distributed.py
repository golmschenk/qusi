import functools
import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, TextIO, Callable, ParamSpec, TypeVar

logger = logging.getLogger(__name__)

P = ParamSpec('P')
R = TypeVar('R')


def ensure_torchrun_environment_variables() -> None:
    """
    Add fake torchrun environment variables if they are not already active.
    """
    if 'RANK' not in os.environ:
        # The script was not called with `torchrun` and environment variables need to be set manually.
        os.environ['RANK'] = str(0)
        os.environ['GROUP_RANK'] = str(0)
        os.environ['LOCAL_RANK'] = str(0)
        os.environ['WORLD_SIZE'] = str(1)
        os.environ['LOCAL_WORLD_SIZE'] = str(1)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '35728'
        os.environ['TORCHELASTIC_RESTART_COUNT'] = str(0)


def distributed_logging(decorated_function: Callable[P, R]) -> Callable[P, R]:
    """
    Enables distributed logging. Each torch worker will log to a separate file. Library extensions also log to this
    file.

    :param decorated_function: The function that should be logged in a distributed way.
    :return: The decorated function.
    """
    if 'RANK' not in os.environ:
        return decorated_function
    if 'QUSI_SESSION_DIRECTORY' not in os.environ:
        logging.warning('Distributed logging session directory environment variable `QUSI_SESSION_DIRECTORY` is not ' +
                        'set. Logs from all ranks will be displayed in stdout.')
        return decorated_function
    session_directory = Path(os.environ['QUSI_SESSION_DIRECTORY'])
    session_directory.mkdir(parents=True, exist_ok=True)
    rank_logging_path = session_directory.joinpath(
        f'rank_{os.environ["RANK"]}_group_rank_{os.environ["GROUP_RANK"]}_local_rank_{os.environ["LOCAL_RANK"]}.log')

    @functools.wraps(decorated_function)
    def redirect_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        """
        The wrapper that applies the redirect.

        :param args: The args passed through to the decorated function.
        :param kwargs: The kwargs passed through to the decorated function.
        :return: The wrapper function.
        """
        with redirected_output(rank_logging_path):
            return decorated_function(*args, **kwargs)

    return redirect_wrapper


@contextmanager
def redirected_output(path: Path) -> Generator[None, None, None]:
    """
    Redirect stdout and stderr to a path. Applies to extension libraries as well.

    :param path: The path to redirect to.
    :return: The redirect context manager.
    """
    stdout_file_descriptor = sys.stdout.fileno()
    stderr_file_descriptor = sys.stderr.fileno()
    original_stdout_file_descriptor = os.dup(stdout_file_descriptor)
    original_stderr_file_descriptor = os.dup(stderr_file_descriptor)
    try:
        with path.open('a') as file:
            sys.stdout.flush()
            os.dup2(file.fileno(), stdout_file_descriptor)
            sys.stderr.flush()
            os.dup2(file.fileno(), stderr_file_descriptor)
            try:
                yield
            finally:
                sys.stdout.flush()
                os.dup2(original_stdout_file_descriptor, stdout_file_descriptor)
                sys.stderr.flush()
                os.dup2(original_stderr_file_descriptor, stderr_file_descriptor)
    finally:
        os.close(original_stdout_file_descriptor)
        os.close(original_stderr_file_descriptor)