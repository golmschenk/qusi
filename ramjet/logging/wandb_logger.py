"""
Code for a logging agent to wandb.
"""
from __future__ import annotations

import queue
import uuid
import wandb
from typing import Optional, List
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


class WandbLoggable:
    def __init__(self, identifier: str):
        self.identifier: str = identifier

class WandbLoggableLightCurve(WandbLoggable):
    def __init__(self, identifier: str, light_curve_name: str, light_curve: LightCurve):
        super().__init__(identifier)
        self.light_curve_name: str = light_curve_name
        self.light_curve: LightCurve = light_curve

class WandbLoggableInjection(WandbLoggable):
    def __init__(self, identifier: str):
        super().__init__(identifier)
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
    def new(cls, identifier) -> WandbLoggableInjection:
        return cls(identifier)


class WandbLogger:
    """
    A class to log to wandb.
    """
    loggable_types = [LightCurve]

    def __init__(self):
        manager = multiprocess.Manager()
        self.queue = manager.Queue()
        self.request_queues: List[multiprocess.Queue] = []

    @classmethod
    def new(cls) -> Optional[WandbLogger]:
        wandb.init(project='qusi', sync_tensorboard=True)
        return cls()

    def process_queue(self):
        while True:
            try:
                queue_item = self.queue.get(block=False)
                if isinstance(queue_item, WandbLoggable):
                    self.log(queue_item)
                else:
                    raise ValueError(f"{queue_item} is not a handled logger type.")
            except queue.Empty:
                break

    def add_mapper_request_queue(self, mapper_publish_queue: ExampleRequestQueue):
        self.request_queues.append(mapper_publish_queue)

    def request_examples(self):
        for request_queue in self.request_queues:
            request_queue.queue.put(ExampleRequest())

    def log_light_curve(self, loggable_light_curve: WandbLoggableLightCurve):
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=loggable_light_curve.light_curve.times,
                                    y=loggable_light_curve.light_curve.fluxes,
                                    mode='lines+markers'))
        figure.update_layout(title=loggable_light_curve.light_curve_name,
                             margin={'l': 0, 'r': 0, 'b': 0, 't': 0})
        if loggable_light_curve.identifier in wandb.run.summary.keys():
            wandb.run.summary.update({loggable_light_curve.identifier: wandb.Plotly(figure)})
        else:
            wandb.log({loggable_light_curve.identifier: wandb.Plotly(figure)})

    def log_injection(self, loggable_injection: WandbLoggableInjection):
        figure = make_subplots(rows=3, cols=1)
        figure.add_trace(go.Scatter(x=loggable_injection.injectee_light_curve.times,
                                    y=loggable_injection.injectee_light_curve.fluxes,
                                    mode='lines+markers', name="injectee"), row=1, col=1)
        figure.add_trace(go.Scatter(x=loggable_injection.injectable_light_curve.times,
                                    y=loggable_injection.injectable_light_curve.fluxes,
                                    mode='lines+markers', name="injectable"), row=2, col=1)
        figure.add_trace(go.Scatter(x=loggable_injection.injected_light_curve.times,
                                    y=loggable_injection.injected_light_curve.fluxes,
                                    mode='lines+markers', name="injected"), row=3, col=1)
        figure.update_layout(title_text=f"{loggable_injection.injectable_name} injected into "
                                        f"{loggable_injection.injectee_name}",
                             margin={'l': 0, 'r': 0, 'b': 0, 't': 0})
        if loggable_injection.identifier in wandb.run.summary.keys():
            wandb.run.summary.update({loggable_injection.identifier: wandb.Plotly(figure)})
        else:
            wandb.log({loggable_injection.identifier: wandb.Plotly(figure)}, commit=False)
        aligned_figure = make_subplots(rows=2, cols=1, shared_xaxes=True)
        aligned_figure.add_trace(go.Scatter(x=loggable_injection.aligned_injectee_light_curve.times,
                                    y=loggable_injection.aligned_injectee_light_curve.fluxes,
                                    mode='lines+markers', name="injectee"), row=1, col=1)
        aligned_figure.add_trace(go.Scatter(x=loggable_injection.aligned_injectable_light_curve.times,
                                    y=loggable_injection.aligned_injectable_light_curve.fluxes,
                                    mode='lines+markers', name="injectable"), row=2, col=1)
        aligned_figure.add_trace(go.Scatter(x=loggable_injection.aligned_injected_light_curve.times,
                                    y=loggable_injection.aligned_injected_light_curve.fluxes,
                                    mode='lines+markers', name="injected"), row=1, col=1)
        aligned_figure.update_layout(title_text=f"Aligned {loggable_injection.injectable_name} injected into "
                                                f"{loggable_injection.injectee_name}",
                                     margin={'l': 0, 'r': 0, 'b': 0, 't': 0})
        aligned_identifier = loggable_injection.identifier + 'aligned'
        if aligned_identifier in wandb.run.summary.keys():
            wandb.run.summary.update({aligned_identifier: wandb.Plotly(aligned_figure)})
        else:
            wandb.log({aligned_identifier: wandb.Plotly(aligned_figure)}, commit=False)

    def log(self, wandb_loggable):
        if isinstance(wandb_loggable, WandbLoggableLightCurve):
            self.log_light_curve(wandb_loggable)
        elif isinstance(wandb_loggable, WandbLoggableInjection):
            self.log_injection(wandb_loggable)
        else:
            raise ValueError(f"{wandb_loggable} does not contain a known loggable type.")

    def create_callback(self) -> WandbLoggerCallback:
        return WandbLoggerCallback(self)

    def create_request_queue_for_collection(self, name: str) -> ExampleRequestQueue:
        example_request_queue = ExampleRequestQueue(name)
        self.request_queues.append(example_request_queue)
        return example_request_queue

    def should_produce_example(self, request_queue: ExampleRequestQueue) -> bool:
        try:
            request_queue.queue.get(block=False)
            return True
        except queue.Empty:
            return False

    def submit_loggable(self, loggable: WandbLoggable):
        self.queue.put(loggable)


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
