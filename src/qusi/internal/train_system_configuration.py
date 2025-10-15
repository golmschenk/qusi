from dataclasses import dataclass


@dataclass
class TrainSystemConfiguration:
    """
    Configuration settings for the system of a train session.

    :ivar preprocessing_processes_per_train_process: The number of processes that are started to preprocess the data
        per train process. The train session will create this many processes for each the train data and the validation
        data.
    """

    preprocessing_processes_per_train_process: int
    accelerator: str

    @classmethod
    def new(
            cls,
            *,
            preprocessing_processes_per_train_process: int = 10,
            accelerator: str = 'auto',
    ):
        """
        Creates a `TrainSystemConfiguration`.

        :param preprocessing_processes_per_train_process: The number of processes that are started to preprocess the data
            per train process. The train session will create this many processes for each the train data and the validation
            data.
        :param accelerator: A string identifying the Lightning accelerator to use.
        :return: The `TrainSystemConfiguration`.
        """
        return cls(
            preprocessing_processes_per_train_process=preprocessing_processes_per_train_process,
            accelerator=accelerator,
        )
