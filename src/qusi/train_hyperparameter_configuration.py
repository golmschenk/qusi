from dataclasses import dataclass


@dataclass
class TrainHyperparameterConfiguration:
    """
    Hyperparameter configuration settings for a train session.

    :ivar batch_size: The size of the batch for each train process. Each training step will use a number of examples
        equal to this value multiplied by the number of train processes.
    :ivar cycles: The number of train cycles to run.
    """
    learning_rate: float
    optimizer_epsilon: float
    weight_decay: float
    batch_size: int
    cycles: int
    train_steps_per_cycle: int
    validation_steps_per_cycle: int
    norm_based_gradient_clip: float

    @classmethod
    def new(cls,
            learning_rate: float = 1e-4,
            optimizer_epsilon: float = 1e-7,
            weight_decay: float = 0.0001,
            batch_size: int = 100,
            train_steps_per_cycle: int = 100,
            validation_steps_per_cycle: int = 10,
            cycles: int = 5000,
            norm_based_gradient_clip: float = 1.0):
        return cls(learning_rate=learning_rate,
                   optimizer_epsilon=optimizer_epsilon,
                   weight_decay=weight_decay,
                   batch_size=batch_size,
                   cycles=cycles,
                   train_steps_per_cycle=train_steps_per_cycle,
                   validation_steps_per_cycle=validation_steps_per_cycle,
                   norm_based_gradient_clip=norm_based_gradient_clip)
