"""
Code for a logging agent to wandb.
"""
from __future__ import annotations

import math
import queue
from abc import ABC, abstractmethod
from pathlib import Path

import plotly
import wandb
from typing import Optional, Dict
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    def log(self, summary_name: str, epoch: int) -> None:
        """
        Logs self to wandb.

        :param summary_name: The name of the summary to use on wandb.
        :param epoch: The current epoch to log with.
        """
        raise NotImplementedError

    @staticmethod
    def log_figure(summary_name: str, figure: plotly.graph_objects.Figure, epoch: int) -> None:
        """
        Logs a figure to wandb.

        :param summary_name: The name of the summary to use on wandb.
        :param figure: The figure to be logged.
        :param epoch: The current epoch to log with.
        """
        wandb.log({summary_name: wandb.Plotly(figure)}, step=epoch)


class WandbLoggableLightCurve(WandbLoggable):
    """
    A wandb loggable light curve.
    """
    def __init__(self, light_curve_name: str, light_curve: LightCurve):
        super().__init__()
        self.light_curve_name: str = light_curve_name
        self.light_curve: LightCurve = light_curve

    def log(self, summary_name: str, epoch: int) -> None:
        """
        Logs self to wandb.

        :param summary_name: The name of the summary to use on wandb.
        :param epoch: The current epoch to log with.
        """
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=self.light_curve.times,
                                    y=self.light_curve.fluxes,
                                    mode='lines+markers'))
        figure.update_layout(title=self.light_curve_name,
                             margin={'l': 0, 'r': 0, 'b': 0, 't': 0})
        self.log_figure(summary_name, figure, epoch)


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

    def log(self, summary_name: str, epoch: int):
        """
        Logs self to wandb.

        :param summary_name: The name of the summary to use on wandb.
        :param epoch: The current epoch to log with.
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
        self.log_figure(summary_name, figure, epoch)
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
        self.log_figure(aligned_summary_name, aligned_figure, epoch)


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
    def new(cls, entity: Optional[str] = None, project: Optional[str] = None) -> WandbLogger:
        """
        Creates a new logger.

        :return: The logger.
        """
        wandb.init(entity=entity, project=project, settings=wandb.Settings(start_method='fork'))
        return cls()

    def process_py_mapper_example_queues(self, epoch: int) -> None:
        """
        Processes the example queues, logging the items to wandb.
        """
        for example_queue_name, example_queue in self.example_queues.items():
            while True:
                try:
                    queue_item = example_queue.get(block=False)
                    pass
                    if isinstance(queue_item, WandbLoggable):
                        queue_item.log(example_queue_name, epoch)
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

    @staticmethod
    def should_produce_example(request_queue: multiprocess.Queue) -> bool:
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

    @staticmethod
    def is_power(number: int, base: int) -> bool:
        """
        Checks if number is a power of the given base.

        :param number: The number.
        :param base: The base.
        :return: Whether the number is a power of the base.
        """
        if base in {0, 1}:
            return number == base
        power = int(math.log(number, base) + 0.5)
        return base ** power == number
