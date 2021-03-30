"""
Code for a logging agent to wandb.
"""
from __future__ import annotations

import queue
import uuid
from abc import ABC, abstractmethod

import wandb
from typing import Optional, Dict
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tensorflow import keras
from pathos.helpers import mp as multiprocess

from ramjet.photometric_database.light_curve import LightCurve


class ExampleRequest:
    pass


class ExampleRequestQueue:
    def __init__(self, name: Optional[str]):
        if name is None:
            name = str(uuid.uuid4())
        self.name: str = name
        manager = multiprocess.Manager()
        self.queue: multiprocess.Queue = manager.Queue()


class WandbLoggable(ABC):
    @abstractmethod
    def log(self, summary_name: str):
        raise NotImplementedError

    def log_figure(self, summary_name, figure):
        if summary_name in wandb.run.summary.keys():
            wandb.run.summary.update({summary_name: wandb.Plotly(figure)})
        else:
            wandb.log({summary_name: wandb.Plotly(figure)})


class WandbLoggableLightCurve(WandbLoggable):
    def __init__(self, light_curve_name: str, light_curve: LightCurve):
        super().__init__()
        self.light_curve_name: str = light_curve_name
        self.light_curve: LightCurve = light_curve

    def log(self, summary_name: str):
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=self.light_curve.times,
                                    y=self.light_curve.fluxes,
                                    mode='lines+markers'))
        figure.update_layout(title=self.light_curve_name,
                             margin={'l': 0, 'r': 0, 'b': 0, 't': 0})
        self.log_figure(summary_name, figure)


class WandbLoggableInjection(WandbLoggable):
    def __init__(self):
        super().__init__()
        self.injectee_name:  Optional[str] = None
        self.injectee_light_curve: Optional[LightCurve] = None
        self.injectable_name:  Optional[str] = None
        self.injectable_light_curve:  Optional[LightCurve] = None
        self.injected_light_curve:  Optional[LightCurve] = None
        self.injectable_light_curve: Optional[LightCurve] = None
        self.injected_light_curve: Optional[LightCurve] = None
        self.aligned_injectable_light_curve: Optional[LightCurve] = None
        self.aligned_injectee_light_curve: Optional[LightCurve] = None
        self.aligned_injected_light_curve: Optional[LightCurve] = None

    @classmethod
    def new(cls) -> WandbLoggableInjection:
        return cls()

    def log(self, summary_name: str):
        figure = make_subplots(rows=3, cols=1)
        figure.add_trace(go.Scatter(x=self.injectee_light_curve.times,
                                    y=self.injectee_light_curve.fluxes,
                                    mode='lines+markers', name="injectee"), row=1, col=1)
        figure.add_trace(go.Scatter(x=self.injectable_light_curve.times,
                                    y=self.injectable_light_curve.fluxes,
                                    mode='lines+markers', name="injectable"), row=2, col=1)
        figure.add_trace(go.Scatter(x=self.injected_light_curve.times,
                                    y=self.injected_light_curve.fluxes,
                                    mode='lines+markers', name="injected"), row=3, col=1)
        figure.update_layout(title_text=f"{self.injectable_name} injected into "
                                        f"{self.injectee_name}",
                             margin={'l': 0, 'r': 0, 'b': 0, 't': 0})
        self.log_figure(summary_name, figure)
        aligned_figure = make_subplots(rows=2, cols=1, shared_xaxes=True)
        aligned_figure.add_trace(go.Scatter(x=self.aligned_injectee_light_curve.times,
                                    y=self.aligned_injectee_light_curve.fluxes,
                                    mode='lines+markers', name="injectee"), row=1, col=1)
        aligned_figure.add_trace(go.Scatter(x=self.aligned_injectable_light_curve.times,
                                    y=self.aligned_injectable_light_curve.fluxes,
                                    mode='lines+markers', name="injectable"), row=2, col=1)
        aligned_figure.add_trace(go.Scatter(x=self.aligned_injected_light_curve.times,
                                    y=self.aligned_injected_light_curve.fluxes,
                                    mode='lines+markers', name="injected"), row=1, col=1)
        aligned_figure.update_layout(title_text=f"Aligned {self.injectable_name} injected into "
                                                f"{self.injectee_name}",
                                     margin={'l': 0, 'r': 0, 'b': 0, 't': 0})
        aligned_summary_name = summary_name + 'aligned'
        self.log_figure(aligned_summary_name, aligned_figure)


class WandbLogger:
    """
    A class to log to wandb.
    """
    loggable_types = [LightCurve]

    def __init__(self):
        manager = multiprocess.Manager()
        self.lock = manager.Lock()
        self.request_queues: Dict[str, multiprocess.Queue] = {}
        self.example_queues: Dict[str, multiprocess.Queue] = {}

    @classmethod
    def new(cls) -> Optional[WandbLogger]:
        wandb.init(project='qusi', sync_tensorboard=True)
        return cls()

    def process_queue(self):
        for example_queue_name, example_queue in self.example_queues.items():
            while True:
                try:
                    queue_item = example_queue.get(block=False)
                    pass
                    if isinstance(queue_item, WandbLoggable):
                        queue_item.log(example_queue_name)
                    else:
                        raise ValueError(f"{queue_item} is not a handled logger type.")
                except queue.Empty:
                    break

    def request_examples(self):
        for request_queue_name, request_queue in self.request_queues.items():
            request_queue.put(ExampleRequest())

    def create_callback(self) -> WandbLoggerCallback:
        return WandbLoggerCallback(self)

    def create_request_queue_for_collection(self, name: str) -> multiprocess.Queue:
        assert name not in self.request_queues.keys()
        manager = multiprocess.Manager()
        queue_ = manager.Queue()
        self.request_queues[name] = queue_
        return queue_

    def create_example_queue_for_collection(self, name: str) -> multiprocess.Queue:
        assert name not in self.example_queues.keys()
        manager = multiprocess.Manager()
        queue_ = manager.Queue()
        self.example_queues[name] = queue_
        return queue_

    def should_produce_example(self, request_queue: multiprocess.Queue) -> bool:
        try:
            request_queue.get(block=False)
            return True
        except queue.Empty:
            return False

    def submit_loggable(self, example_queue: multiprocess.Queue, loggable: WandbLoggable):
        while True:  # This loop should not be required, but it seems there is a bug in multiprocessing.
            try:     # https://stackoverflow.com/q/29277150/1191087
                example_queue.put(loggable)
                break
            except TypeError:
                continue


class WandbLoggerCallback(keras.callbacks.Callback):
    def __init__(self, logger: WandbLogger):
        super().__init__()
        self.logger = logger

    def on_train_begin(self, logs=None):
        self.logger.process_queue()

    def on_epoch_begin(self, epoch, logs=None):
        self.logger.request_examples()

    def on_epoch_end(self, epoch, logs=None):
        self.logger.process_queue()
