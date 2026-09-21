from dataclasses import dataclass

import torch


@dataclass
class TrainingSystemConfiguration:
    """
    Configuration settings for the system of a training session.

    :ivar preprocessing_processes_per_train_process: The number of processes that are started to preprocess the data
        per train process. The train session will create this many processes for each of the train data and the
        validation data.
    :ivar accelerator: The accelerator to run the NN on.
    :ivar distributed_backend: The distributed backend to use.
    """

    preprocessing_processes_per_train_process: int
    accelerator: str
    distributed_backend: torch.distributed.Backend

    @classmethod
    def new(
            cls,
            *,
            data_workers_per_train_process: int = 10,
            accelerator: str = 'auto',
            distributed_backend: torch.distributed.Backend = torch.distributed.Backend.GLOO,
    ):
        """
        Creates a `TrainingSystemConfiguration`.

        :param data_workers_per_train_process: The number of processes that are started to preprocess the data
            per train process. The train session will create this many processes for both the train data and the
            validation data.
        :param accelerator: A string identifying the Lightning accelerator to use.
        :return: The `TrainingSystemConfiguration`.
        """
        return cls(
            preprocessing_processes_per_train_process=data_workers_per_train_process,
            accelerator=accelerator,
            distributed_backend=distributed_backend,
        )
