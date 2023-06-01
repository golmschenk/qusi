from typing import List

from torch.utils.data import DataLoader

from qusi.light_curve_dataset import LightCurveDataset, contains_injected_dataset, interleave_iterables_infinitely


class TrainSession:
    def __init__(self,
                 train_datasets: List[LightCurveDataset],
                 validation_datasets: List[LightCurveDataset],
                 batch_size: int,
                 train_steps_per_epoch: int | None = None,
                 validation_steps_per_epoch: int | None = None,
                 ):
        self.train_datasets: List[LightCurveDataset] = train_datasets
        self.validation_datasets: List[LightCurveDataset] = validation_datasets
        self.batch_size: int = batch_size
        self.train_steps_per_epoch: int = train_steps_per_epoch
        self.validation_steps_per_epoch: int = validation_steps_per_epoch

    @classmethod
    def new(cls,
            train_datasets: LightCurveDataset | List[LightCurveDataset],
            validation_datasets: LightCurveDataset | List[LightCurveDataset],
            batch_size: int,
            train_steps_per_epoch: int,
            validation_steps_per_epoch: int,
            ):
        if not isinstance(train_datasets, list):
            train_datasets = [train_datasets]
        train_datasets: List[LightCurveDataset] = train_datasets
        if not isinstance(validation_datasets, list):
            validation_datasets = [validation_datasets]
        validation_datasets: List[LightCurveDataset] = validation_datasets
        instance = cls(train_datasets=train_datasets,
                       validation_datasets=validation_datasets,
                       batch_size=batch_size,
                       train_steps_per_epoch=train_steps_per_epoch,
                       validation_steps_per_epoch=validation_steps_per_epoch)
        return instance

    def run(self):
        train_dataset = interleave_iterables_infinitely(*self.train_datasets)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size)
        for batch_index, (light_curve, target) in enumerate(train_dataloader):
            print(batch_index)
    # TODO: Create training loop.


