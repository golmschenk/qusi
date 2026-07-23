import warnings
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
    :ivar global_batch_size: global_batch_size: Each training step will use a number of observations
            equal to this value. If this is set, local_batch_size must be None.
    :ivar local_batch_size: The size of the batch for each train process. Each training step will use a
            number of observations equal to this value multiplied by the number of train processes. If this is set,
            global_batch_size must be None.
    :ivar norm_based_gradient_clip: The norm based gradient clipping value.
    """

    cycles: int
    train_steps_per_cycle: int
    validation_steps_per_cycle: int
    norm_based_gradient_clip: float
    global_batch_size: int | None = None
    local_batch_size: int | None = None

    def __post_init__(self):
        if ((self.global_batch_size is None and self.local_batch_size is None) or
                (self.global_batch_size is not None and self.local_batch_size is not None)):
            raise ValueError('Exactly one of `global_batch_size` or `local_batch_size` must be set. '
                             f'Got global_batch_size={self.global_batch_size} and '
                             f'local_batch_size={self.local_batch_size}.')

    @classmethod
    def new(
            cls,
            *,
            cycles: int = 5000,
            train_steps_per_cycle: int = 100,
            validation_steps_per_cycle: int = 10,
            global_batch_size: int | None = 100,
            local_batch_size: int | None = None,
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
        :param global_batch_size: Each training step will use a number of observations
            equal to this value. If this is set, local_batch_size must be None.
        :param local_batch_size: The size of the batch for each train process. Each training step will use a
            number of observations equal to this value multiplied by the number of train processes. If this is set,
            global_batch_size must be None.
        :param norm_based_gradient_clip: The norm based gradient clipping value.
        :return: The hyperparameter configuration.
        """
        return cls(
            cycles=cycles,
            train_steps_per_cycle=train_steps_per_cycle,
            validation_steps_per_cycle=validation_steps_per_cycle,
            global_batch_size=global_batch_size,
            local_batch_size=local_batch_size,
            norm_based_gradient_clip=norm_based_gradient_clip,
        )

    def update_batch_sizes_based_on_world_size(self, world_size: int):
        if self.global_batch_size is not None:
            unrounded_local_batch_size = self.global_batch_size / world_size
            if not unrounded_local_batch_size.is_integer():
                warnings.warn(f'The global batch size of `{self.global_batch_size}` is not '
                              f'divisible by the world size of `{world_size}`. Rounding to determine local '
                              f'batch size.')
            self.local_batch_size = round(unrounded_local_batch_size)
            if self.local_batch_size == 0:
                self.local_batch_size = 1
        elif self.local_batch_size is not None:
            self.global_batch_size = (self.local_batch_size * world_size)
        else:
            raise ValueError('The hyperparameter configuration requires that either `global_batch_size` or '
                             '`local_batch_size` be set.')
