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
    :ivar learning_rate: The learning rate.
    :ivar optimizer_epsilon: The epsilon to be used by the optimizer.
    :ivar weight_decay: The weight decay of the optimizer.
    :ivar norm_based_gradient_clip: The norm based gradient clipping value.
    """

    cycles: int
    train_steps_per_cycle: int
    validation_steps_per_cycle: int
    batch_size: int
    learning_rate: float
    optimizer_epsilon: float
    weight_decay: float
    norm_based_gradient_clip: float

    @classmethod
    def new(
            cls,
            *,
            cycles: int = 5000,
            train_steps_per_cycle: int = 100,
            validation_steps_per_cycle: int = 10,
            batch_size: int = 100,
            learning_rate: float = 1e-4,
            optimizer_epsilon: float = 1e-7,
            weight_decay: float = 0.0001,
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
        :param learning_rate: The learning rate.
        :param optimizer_epsilon: The epsilon to be used by the optimizer.
        :param weight_decay: The weight decay of the optimizer.
        :param norm_based_gradient_clip: The norm based gradient clipping value.
        :return: The hyperparameter configuration.
        """
        return cls(
            learning_rate=learning_rate,
            optimizer_epsilon=optimizer_epsilon,
            weight_decay=weight_decay,
            batch_size=batch_size,
            cycles=cycles,
            train_steps_per_cycle=train_steps_per_cycle,
            validation_steps_per_cycle=validation_steps_per_cycle,
            norm_based_gradient_clip=norm_based_gradient_clip,
        )
