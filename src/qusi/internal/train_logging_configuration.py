from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TrainLoggingConfiguration:
    """
    Configuration settings for the logging of a train session.

    :ivar wandb_project: The wandb project to log to.
    :ivar wandb_entity: The wandb entity to log to.
    :ivar additional_log_dictionary: The dictionary of additional values to log.
    """

    wandb_project: str | None
    wandb_entity: str | None
    additional_log_dictionary: dict[str, Any]

    @classmethod
    def new(
            cls,
            *,
            wandb_project: str | None = None,
            wandb_entity: str | None = None,
            additional_log_dictionary: dict[str, Any] | None = None,
    ):
        """
        Creates a `TrainLoggingConfiguration`.
        
        :param wandb_project: The wandb project to log to.
        :param wandb_entity: The wandb entity to log to.
        :param additional_log_dictionary: The dictionary of additional values to log.
        :return: The `TrainLoggingConfiguration`.
        """
        if additional_log_dictionary is None:
            additional_log_dictionary = {}
        return cls(
            wandb_project=wandb_project, wandb_entity=wandb_entity, additional_log_dictionary=additional_log_dictionary
        )
