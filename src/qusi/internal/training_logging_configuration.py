from __future__ import annotations

import datetime
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TrainingLoggingConfiguration:
    """
    Configuration settings for the logging of a training session.

    :ivar session_directory: Path to the session directory.
    :ivar additional_log_dictionary: The dictionary of additional values to log.
    :ivar wandb_project: The wandb project to log to.
    :ivar wandb_entity: The wandb entity to log to.
    """

    session_directory: Path
    additional_log_dictionary: dict[str, Any]
    wandb_project: str | None
    wandb_entity: str | None

    @classmethod
    def new(
            cls,
            *,
            session_directory: Path | None = None,
            additional_log_dictionary: dict[str, Any] | None = None,
            wandb_project: str | None = None,
            wandb_entity: str | None = None,
    ):
        """
        Creates a `TrainingLoggingConfiguration`.

        :param session_directory: Path to the session directory.
        :param additional_log_dictionary: The dictionary of additional values to log.
        :param wandb_project: The wandb project to log to.
        :param wandb_entity: The wandb entity to log to.
        :return: The `TrainingLoggingConfiguration`.
        """
        if additional_log_dictionary is None:
            additional_log_dictionary = {}
        session_directory_environment_variable = os.environ.get('QUSI_SESSION_DIRECTORY')
        if session_directory_environment_variable is not None and session_directory is not None:
            raise ValueError(f'Passing a `session_directory` is not allowed when the environment variable '
                             f'`QUSI_SESSION_DIRECTORY` is set.')
        if session_directory is None:
            session_directory = session_directory_environment_variable
        if session_directory is None:
            datetime_string = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            session_directory = Path(f'sessions/{datetime_string}')
        if isinstance(session_directory, str):
            session_directory = Path(session_directory)
        return cls(
            session_directory=session_directory,
            additional_log_dictionary=additional_log_dictionary,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
        )
