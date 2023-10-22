from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import torch
from bokeh.io import show
from bokeh.models import Div, Column
from torch import Tensor
from torch.nn import BCELoss, Module
from torch.optim import Adam
from torch.utils.data import DataLoader
from bokeh.plotting import figure as Figure

from qusi.light_curve_dataset import LightCurveDataset, InterleavedDataset


@dataclass
class TrainSession:
    train_datasets: List[LightCurveDataset]
    validation_datasets: List[LightCurveDataset]
    model: Module
    batch_size: int
    cycles: int
    train_steps_per_cycle: int
    validation_steps_per_cycle: int

    @classmethod
    def new(cls, train_datasets: LightCurveDataset | List[LightCurveDataset],
            validation_datasets: LightCurveDataset | List[LightCurveDataset], model: Module, batch_size: int,
            cycles: int, train_steps_per_cycle: int, validation_steps_per_cycle: int):
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
                       cycles=cycles,
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
        torch.multiprocessing.set_start_method('spawn')
        debug = True
        if debug:
            workers_per_dataloader = 0
            prefetch_factor = None
            persistent_workers = False
        else:
            workers_per_dataloader = 10
            prefetch_factor = 10
            persistent_workers = True
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, pin_memory=True,
                                      persistent_workers=persistent_workers, prefetch_factor=prefetch_factor, num_workers=workers_per_dataloader)
        validation_dataloaders: List[DataLoader] = []
        for validation_dataset in self.validation_datasets:
            validation_dataloaders.append(DataLoader(validation_dataset, batch_size=self.batch_size, pin_memory=True,
                                                     persistent_workers=persistent_workers, prefetch_factor=prefetch_factor, num_workers=workers_per_dataloader))
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        self.model = self.model.to(device)
        loss_function = BCELoss().to(device)
        optimizer = Adam(self.model.parameters())
        for cycle_index in range(self.cycles):
            train_phase(dataloader=train_dataloader, model_=self.model, loss_fn=loss_function, optimizer=optimizer,
                        steps=self.train_steps_per_cycle, device=device)
            for validation_dataloader in validation_dataloaders:
                validation_phase(dataloader=validation_dataloader, model_=self.model, loss_fn=loss_function,
                                 steps=self.validation_steps_per_cycle, device=device)
        torch.save(self.model.state_dict(), session_directory.joinpath('latest_model.pth'))


def train_phase(dataloader, model_, loss_fn, optimizer, steps, device):
    model_.train()
    for batch_index, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        # TODO: The conversion to float32 probably shouldn't be here, but the default collate_fn seems to be converting
        #  to float64. Probably should override the default collate.
        y = y.to(torch.float32).to(device)
        X = X.to(torch.float32).to(device)
        pred = model_(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_index % 10 == 0:
            loss, current = loss.to('cpu').item(), (batch_index + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{steps * len(X):>5d}]", flush=True)
        if batch_index >= steps:
            break


def validation_phase(dataloader, model_, loss_fn, steps, device):
    model_.eval()
    validation_loss, correct = 0, 0

    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            y = y.to(torch.float32).to(device)
            X = X.to(torch.float32).to(device)
            pred = model_(X)
            validation_loss += loss_fn(pred, y).to('cpu').item()
            correct += (torch.round(pred) == y).type(torch.float32).sum().to('cpu').item()
            if batch >= steps + 1:
                break

    validation_loss /= steps
    correct /= steps * dataloader.batch_size
    print(f"Validation Error: \nAvg loss: {validation_loss:>8f} \n")

