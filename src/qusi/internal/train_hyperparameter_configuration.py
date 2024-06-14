from dataclasses import dataclass


@dataclass
class TrainHyperparameterConfiguration:
    """
    Hyperparameter configuration settings for a train session.

    :ivar cycles: The number of cycles to run. Cycles consist of one set of training steps and one set of validation
                  steps. They can be seen as analogous to epochs. However, as qusi datasets are often
                  infinite or have different length sub-collections, there is not always the exact equivalent of an
                  epoch, so cycles are used instead.
    :ivar train_steps_per_cycle: The number of training steps per cycle.
    :ivar validation_steps_per_cycle: The number of validation steps per cycle.
    :ivar batch_size: The size of the batch for each train process. Each training step will use a number of observations
        equal to this value multiplied by the number of train processes.
    :ivar norm_based_gradient_clip: The norm based gradient clipping value.
    """

    cycles: int
    train_steps_per_cycle: int
    validation_steps_per_cycle: int
    batch_size: int
    norm_based_gradient_clip: float

    @classmethod
    def new(
            cls,
            *,
            cycles: int = 5000,
            train_steps_per_cycle: int = 100,
            validation_steps_per_cycle: int = 10,
            batch_size: int = 100,
            norm_based_gradient_clip: float = 1.0,
    ):
        """
        Creates a new `TrainHyperparameterConfiguration`.

        :param cycles: The number of cycles to run. Cycles consist of one set of training steps and one set of validation
                  steps. They can be seen as analogous to epochs. However, as qusi datasets are often
                  infinite or have different length sub-collections, there is not always the exact equivalent of an
                  epoch, so cycles are used instead.
        :param train_steps_per_cycle: The number of training steps per cycle.
        :param validation_steps_per_cycle: The number of validation steps per cycle.
        :param batch_size: The size of the batch for each train process. Each training step will use a number of observations
            equal to this value multiplied by the number of train processes.
        :param norm_based_gradient_clip: The norm based gradient clipping value.
        :return: The hyperparameter configuration.
        """
        return cls(
            cycles=cycles,
            train_steps_per_cycle=train_steps_per_cycle,
            validation_steps_per_cycle=validation_steps_per_cycle,
            batch_size=batch_size,
            norm_based_gradient_clip=norm_based_gradient_clip,
        )
