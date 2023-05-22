from typing import List

from qusi.light_curve_dataset import LightCurveDataset


class TrainSession:
    def __init__(self,
                 train_datasets: LightCurveDataset | List[LightCurveDataset],
                 validation_datasets: LightCurveDataset | List[LightCurveDataset]):
        if not isinstance(train_datasets, list):
            train_datasets = [train_datasets]
        self.train_datasets: List[LightCurveDataset] = train_datasets
        if not isinstance(validation_datasets, list):
            validation_datasets = [validation_datasets]
        self.validation_datasets: List[LightCurveDataset] = validation_datasets

    def run(self):
        pass
