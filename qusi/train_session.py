from datetime import datetime
from pathlib import Path
from typing import List

import torch
from torch.nn import BCELoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from qusi.light_curve_dataset import LightCurveDataset, contains_injected_dataset, \
    interleave_infinite_iterators, InterleavedDataset
from qusi.single_dense_layer_model import SingleDenseLayerBinaryClassificationModel


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
        current_datetime = datetime.now()
        datetime_string = current_datetime.strftime("%Y_%m_%d_%H_%M_%S")
        sessions_directory = Path('sessions')
        sessions_directory.mkdir(exist_ok=True)
        session_directory = sessions_directory.joinpath(f'session_{datetime_string}')
        session_directory.mkdir(exist_ok=True)
        train_dataset = InterleavedDataset.new(*self.train_datasets)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size)
        validation_dataloaders: List[DataLoader] = []
        for validation_dataset in self.validation_datasets:
            validation_dataloaders.append(DataLoader(validation_dataset, batch_size=self.batch_size))
        model = SingleDenseLayerBinaryClassificationModel(input_size=100)
        loss_function = BCELoss()
        optimizer = Adam(model.parameters())
        for epoch_index in range(7):
            train_epoch(dataloader=train_dataloader, model_=model, loss_fn=loss_function, optimizer=optimizer, steps=self.train_steps_per_epoch)
            for validation_dataloader in validation_dataloaders:
                validation_epoch(dataloader=validation_dataloader, model_=model, loss_fn=loss_function, steps=self.validation_steps_per_epoch)
        torch.save(model.state_dict(), session_directory.joinpath('latest_model.pth'))


def train_epoch(dataloader, model_, loss_fn, optimizer, steps):
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        # TODO: The conversion to float32 probably shouldn't be here, but the default collate_fn seems to be converting
        #  to float64. Probably should override the default collate.
        y = y.to(torch.float32)
        X = X.to(torch.float32)
        pred = model_(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{steps:>5d}]")
        if batch >= steps + 1:
            break

def validation_epoch(dataloader, model_, loss_fn, steps):
    validation_loss, correct = 0, 0

    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            y = y.to(torch.float32)
            X = X.to(torch.float32)
            pred = model_(X)
            validation_loss += loss_fn(pred, y).item()
            correct += (torch.round(pred) == y).type(torch.float32).sum().item()
            if batch >= steps + 1:
                break

    validation_loss /= steps
    correct /= steps * 103
    print(f"Validation Error: \nAvg loss: {validation_loss:>8f} \n")
