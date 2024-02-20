"""
An abstract class allowing for any number and combination of standard and injectable/injectee light curve collections.
"""
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Callable

import numpy as np
import numpy.typing as npt

from ramjet.photometric_database.light_curve import LightCurve
from ramjet.photometric_database.light_curve_database import LightCurveDatabase
from ramjet.photometric_database.light_curve_dataset_manipulations import (
    BaselineFluxEstimationMethod,
    OutOfBoundsInjectionHandlingMethod,
    inject_signal_into_light_curve_with_intermediates,
)

if TYPE_CHECKING:
    from ramjet.logging.wandb_logger import WandbLoggableInjection, WandbLogger
    from ramjet.photometric_database.light_curve_collection import LightCurveCollection


def inject_signal_into_light_curve(
    light_curve_times: npt.NDArray[np.float64],
    light_curve_fluxes: npt.NDArray[np.float64],
    signal_times: npt.NDArray[np.float64],
    signal_magnifications: npt.NDArray[np.float64],
    out_of_bounds_injection_handling_method: OutOfBoundsInjectionHandlingMethod = (
        OutOfBoundsInjectionHandlingMethod.ERROR
    ),
    baseline_flux_estimation_method: BaselineFluxEstimationMethod = BaselineFluxEstimationMethod.MEDIAN,
) -> npt.NDArray[np.float64]:
    """
    Injects a synthetic magnification signal into real light curve fluxes.

    :param light_curve_times: The times of the flux observations of the light curve.
    :param light_curve_fluxes: The fluxes of the light curve to be injected into.
    :param signal_times: The times of the synthetic magnifications.
    :param signal_magnifications: The synthetic magnifications to inject.
    :param out_of_bounds_injection_handling_method: The method to use to handle out of bounds injection.
    :param baseline_flux_estimation_method: The method to use to estimate the baseline flux of the light curve
                                            for scaling the signal magnifications.
    :return: The fluxes with the injected signal, the offset signal times, and the signal flux.
    """
    fluxes_with_injected_signal, _, _ = inject_signal_into_light_curve_with_intermediates(
        light_curve_times=light_curve_times,
        light_curve_fluxes=light_curve_fluxes,
        signal_times=signal_times,
        signal_magnifications=signal_magnifications,
        out_of_bounds_injection_handling_method=out_of_bounds_injection_handling_method,
        baseline_flux_estimation_method=baseline_flux_estimation_method,
    )
    return fluxes_with_injected_signal


class StandardAndInjectedLightCurveDatabase(LightCurveDatabase):
    """
    An abstract class allowing for any number and combination of standard and injectable/injectee light curve
    collections to be used for training.
    """

    def __init__(self):
        super().__init__()
        self.training_standard_light_curve_collections: list[LightCurveCollection] = []
        self.training_injectee_light_curve_collection: LightCurveCollection | None = None
        self.training_injectable_light_curve_collections: list[LightCurveCollection] = []
        self.validation_standard_light_curve_collections: list[LightCurveCollection] = []
        self.validation_injectee_light_curve_collection: LightCurveCollection | None = None
        self.validation_injectable_light_curve_collections: list[LightCurveCollection] = []
        self.inference_light_curve_collections: list[LightCurveCollection] = []
        self.shuffle_buffer_size = 10000
        self.number_of_label_values = 1
        self.number_of_auxiliary_values: int = 0
        self.out_of_bounds_injection_handling: OutOfBoundsInjectionHandlingMethod = (
            OutOfBoundsInjectionHandlingMethod.ERROR
        )
        self.baseline_flux_estimation_method = BaselineFluxEstimationMethod.MEDIAN
        self.logger: WandbLogger | None = None

    @property
    def number_of_input_channels(self) -> int:
        """
        Determines the number of input channels that should exist for this database.

        :return: The number of channels.
        """
        channels = 1
        if self.include_time_as_channel:
            channels += 1
        if self.include_flux_errors_as_channel:
            channels += 1
        return channels

    def add_logging_queues_to_map_function(self, preprocess_map_function: Callable, name: str | None) -> Callable:
        """
        Adds logging queues to the map functions.

        :param preprocess_map_function: The function to map.
        :param name: The name of the dataset.
        :return: The updated map function.
        """
        if self.logger is not None:
            preprocess_map_function = partial(
                preprocess_map_function,
                request_queue=self.logger.create_request_queue_for_collection(name),
                example_queue=self.logger.create_example_queue_for_collection(name),
            )
        return preprocess_map_function

    def inject_signal_into_light_curve(
        self,
        light_curve_fluxes: np.ndarray,
        light_curve_times: np.ndarray,
        signal_magnifications: np.ndarray,
        signal_times: np.ndarray,
        wandb_loggable_injection: WandbLoggableInjection | None = None,
    ) -> np.ndarray:
        """
        Injects a synthetic magnification signal into real light curve fluxes.

        :param light_curve_fluxes: The fluxes of the light curve to be injected into.
        :param light_curve_times: The times of the flux observations of the light curve.
        :param signal_magnifications: The synthetic magnifications to inject.
        :param signal_times: The times of the synthetic magnifications.
        :param wandb_loggable_injection: The object to log the injection process.
        :return: The fluxes with the injected signal.
        """
        out_of_bounds_injection_handling_method = self.out_of_bounds_injection_handling
        baseline_flux_estimation_method = self.baseline_flux_estimation_method
        (
            fluxes_with_injected_signal,
            offset_signal_times,
            signal_fluxes,
        ) = inject_signal_into_light_curve_with_intermediates(
            light_curve_times,
            light_curve_fluxes,
            signal_times,
            signal_magnifications,
            out_of_bounds_injection_handling_method,
            baseline_flux_estimation_method,
        )
        if wandb_loggable_injection is not None:
            wandb_loggable_injection.aligned_injectee_light_curve = LightCurve.from_times_and_fluxes(
                light_curve_times, light_curve_fluxes
            )
            wandb_loggable_injection.aligned_injectable_light_curve = LightCurve.from_times_and_fluxes(
                offset_signal_times, signal_fluxes
            )
            wandb_loggable_injection.aligned_injected_light_curve = LightCurve.from_times_and_fluxes(
                light_curve_times, fluxes_with_injected_signal
            )
        return fluxes_with_injected_signal


def expand_label_to_training_dimensions(label: int | list[int] | tuple[int] | np.ndarray) -> np.ndarray:
    """
    Expand the label to the appropriate dimensions for training.

    :param label: The label to convert.
    :return: The label with the correct dimensions.
    """
    if type(label) is not np.ndarray:
        if type(label) in [list, tuple]:
            label = np.array(label)
        else:
            label = np.array([label])
    return label
