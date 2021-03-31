"""
Code for a logging agent to wandb.
"""
from __future__ import annotations

import queue
from abc import ABC, abstractmethod

import plotly
import wandb
from typing import Optional, Dict
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tensorflow import keras
from pathos.helpers import mp as multiprocess

from ramjet.photometric_database.light_curve import LightCurve


class ExampleRequest:
    """
    A representation of a request for an example.
    """
    pass


class WandbLoggable(ABC):
    """
    An object which is loggable to wandb.
    """

    @abstractmethod
    def log(self, summary_name: str) -> None:
        """
        Logs self to wandb.

        :param summary_name: The name of the summary to use on wandb.
        """
        raise NotImplementedError

    def log_latest_figure(self, summary_name: str, figure: plotly.graph_objects.Figure) -> None:
        """
        Logs a figure to wandb, keeping only the most recent one.

        :param summary_name: The name of the summary to use on wandb.
        :param figure: The figure to be logged.
        :return:
        """
        if summary_name in wandb.run.summary.keys():
            wandb.run.summary.update({summary_name: wandb.Plotly(figure)})
        else:
            wandb.log({summary_name: wandb.Plotly(figure)})


class WandbLoggableLightCurve(WandbLoggable):
    """
    A wandb loggable light curve.
    """
    def __init__(self, light_curve_name: str, light_curve: LightCurve):
        super().__init__()
        self.light_curve_name: str = light_curve_name
        self.light_curve: LightCurve = light_curve

    def log(self, summary_name: str) -> None:
        """
        Logs self to wandb.

        :param summary_name: The name of the summary to use on wandb.
        """
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=self.light_curve.times,
                                    y=self.light_curve.fluxes,
                                    mode='lines+markers'))
        figure.update_layout(title=self.light_curve_name,
                             margin={'l': 0, 'r': 0, 'b': 0, 't': 0})
        self.log_latest_figure(summary_name, figure)


class WandbLoggableInjection(WandbLoggable):
    """
    A wandb loggable containing logging data for injecting a signal into a light curve.
    """
    def __init__(self):
        super().__init__()
        self.injectee_name: Optional[str] = None
        self.injectee_light_curve: Optional[LightCurve] = None
        self.injectable_name: Optional[str] = None
        self.injectable_light_curve: Optional[LightCurve] = None
        self.injected_light_curve: Optional[LightCurve] = None
        self.injectable_light_curve: Optional[LightCurve] = None
        self.injected_light_curve: Optional[LightCurve] = None
        self.aligned_injectable_light_curve: Optional[LightCurve] = None
        self.aligned_injectee_light_curve: Optional[LightCurve] = None
        self.aligned_injected_light_curve: Optional[LightCurve] = None

    def log(self, summary_name: str):
        """
        Logs self to wandb.

        :param summary_name: The name of the summary to use on wandb.
        """
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
        self.log_latest_figure(summary_name, figure)
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
        self.log_latest_figure(aligned_summary_name, aligned_figure)


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
    def new(cls) -> WandbLogger:
        """
        Creates a new logger.

        :return: The logger.
        """
        wandb.init(project='qusi', sync_tensorboard=True)
        return cls()

    def process_py_mapper_example_queues(self) -> None:
        """
        Processes the example queues, logging the items to wandb.
        """
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

    def request_examples_from_py_mapper_processes(self) -> None:
        """
        Sends requests for examples to the other processes.
        """
        for request_queue_name, request_queue in self.request_queues.items():
            request_queue.put(ExampleRequest())

    def create_callback(self) -> WandbLoggerCallback:
        """
        Creates a callback for the fit loop to call the logger methods.

        :return: The callback.
        """
        return WandbLoggerCallback(self)

    def create_request_queue_for_collection(self, name: str) -> multiprocess.Queue:
        """
        Creates a queue to send requests for examples on.

        :param name: The name of the queue.
        :return: The queue.
        """
        assert name not in self.request_queues.keys()
        manager = multiprocess.Manager()
        queue_ = manager.Queue()
        self.request_queues[name] = queue_
        return queue_

    def create_example_queue_for_collection(self, name: str) -> multiprocess.Queue:
        """
        Creates a queue to receive examples on.

        :param name: The name of the queue.
        :return: The queue.
        """
        assert name not in self.example_queues.keys()
        manager = multiprocess.Manager()
        queue_ = manager.Queue()
        self.example_queues[name] = queue_
        return queue_

    def should_produce_example(self, request_queue: multiprocess.Queue) -> bool:
        """
        Checks a request queue to see if an example has been requested.

        :param request_queue: The request queue.
        :return: Whether an example was requested on the queue.
        """
        try:
            request_queue.get(block=False)
            return True
        except queue.Empty:
            return False

    @staticmethod
    def submit_loggable(example_queue: multiprocess.Queue, loggable: WandbLoggable) -> None:
        """
        Submits a loggable to a request queue to be logged by the main process.

        :param example_queue: The queue to submit the loggable to.
        :param loggable: The loggable to be submitted.
        """
        while True:  # This loop should not be required, but it seems there is a bug in multiprocessing.
            try:  # https://stackoverflow.com/q/29277150/1191087
                example_queue.put(loggable)
                break
            except TypeError:
                continue


class WandbLoggerCallback(keras.callbacks.Callback):
    """
    A callback for the training loop to call to utilize the wandb logger.
    """
    def __init__(self, logger: WandbLogger):
        super().__init__()
        self.logger = logger

    def on_epoch_begin(self, epoch, logs=None):
        """Called at the beginning of an epoch."""
        self.logger.request_examples_from_py_mapper_processes()

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of an epoch."""
        self.logger.process_py_mapper_example_queues()
