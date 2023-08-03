from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
from torch.nn import Module
from torch.utils.data import DataLoader

from qusi.light_curve_dataset import LightCurveDataset, contains_injected_dataset, \
    interleave_infinite_iterators, InterleavedDataset, ConcatenatedIterableDataset
from qusi.single_dense_layer_model import SingleDenseLayerBinaryClassificationModel


@dataclass
class InferSession:
    infer_datasets: List[LightCurveDataset]
    model: Module
    batch_size: int

    @classmethod
    def new(cls,
            infer_datasets: LightCurveDataset | List[LightCurveDataset],
            model: Module,
            batch_size: int,
            ):
        if not isinstance(infer_datasets, list):
            infer_datasets = [infer_datasets]
        instance = cls(infer_datasets=infer_datasets,
                       model=model,
                       batch_size=batch_size,
                       )
        return instance

    def run(self):
        with torch.no_grad():
            sessions_directory = Path('sessions')
            session_directory = sessions_directory.joinpath(f'session_2023_07_25_14_59_51')
            infer_dataset = ConcatenatedIterableDataset.new(*self.infer_datasets)
            infer_dataloader = DataLoader(infer_dataset, batch_size=self.batch_size)
            model_path = session_directory.joinpath('latest_model.pth')
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
            predictions = infer_epoch(dataloader=infer_dataloader, model_=self.model)
        return predictions


def infer_epoch(dataloader, model_):
    batch_predictions = []
    for batch, (X, y) in enumerate(dataloader):
        y = y.to(torch.float32)
        X = X.to(torch.float32)
        pred = model_(X).to('cpu')
        batch_predictions.append(pred)
    predictions = torch.cat(batch_predictions, dim=0)
    return predictions
