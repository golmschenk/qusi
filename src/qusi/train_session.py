from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import torch
from torch.nn import BCELoss, Module
from torch.optim import Adam
from torch.utils.data import DataLoader

from qusi.light_curve_dataset import LightCurveDataset, InterleavedDataset
from qusi.single_dense_layer_model import SingleDenseLayerBinaryClassificationModel


@dataclass
class TrainSession:
    train_datasets: List[LightCurveDataset]
    validation_datasets: List[LightCurveDataset]
    model: Module
    batch_size: int
    train_steps_per_cycle: int
    validation_steps_per_cycle: int

    @classmethod
    def new(cls, train_datasets: LightCurveDataset | List[LightCurveDataset],
            validation_datasets: LightCurveDataset | List[LightCurveDataset], model: Module, batch_size: int,
            train_steps_per_cycle: int, validation_steps_per_cycle: int):
        if not isinstance(train_datasets, list):
            train_datasets = [train_datasets]
        train_datasets: List[LightCurveDataset] = train_datasets
        if not isinstance(validation_datasets, list):
            validation_datasets = [validation_datasets]
        validation_datasets: List[LightCurveDataset] = validation_datasets
        instance = cls(train_datasets=train_datasets,
                       validation_datasets=validation_datasets,
                       model=model,
                       batch_size=batch_size,
                       train_steps_per_cycle=train_steps_per_cycle,
                       validation_steps_per_cycle=validation_steps_per_cycle)
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
        loss_function = BCELoss()
        optimizer = Adam(self.model.parameters())
        for cycle_index in range(7):
            train_phase(dataloader=train_dataloader, model_=self.model, loss_fn=loss_function, optimizer=optimizer,
                        steps=self.train_steps_per_cycle)
            for validation_dataloader in validation_dataloaders:
                validation_phase(dataloader=validation_dataloader, model_=self.model, loss_fn=loss_function,
                                 steps=self.validation_steps_per_cycle)
        torch.save(self.model.state_dict(), session_directory.joinpath('latest_model.pth'))


def train_phase(dataloader, model_, loss_fn, optimizer, steps):
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

        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{steps * len(X):>5d}]")
        if batch >= steps + 1:
            break


def validation_phase(dataloader, model_, loss_fn, steps):
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
    correct /= steps * dataloader.batch_size
    print(f"Validation Error: \nAvg loss: {validation_loss:>8f} \n")
