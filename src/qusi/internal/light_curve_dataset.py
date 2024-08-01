from __future__ import annotations

import copy
import itertools
import math
import re
import shutil
import socket
from enum import Enum
from functools import partial
from pathlib import Path
from random import Random
from typing import TYPE_CHECKING, Any, Callable, TypeVar

import numpy as np
import numpy.typing as npt
import torch
from filelock import FileLock
from scipy import stats
from scipy.interpolate import interp1d
from torch import Tensor
from torch.utils.data import IterableDataset
from typing_extensions import Self

from qusi.internal.light_curve import (
    LightCurve,
    randomly_roll_light_curve,
    remove_nan_flux_data_points_from_light_curve,
)
from qusi.internal.light_curve_observation import (
    LightCurveObservation,
    randomly_roll_light_curve_observation,
    remove_nan_flux_data_points_from_light_curve_observation,
)
from qusi.internal.light_curve_transforms import (
    from_light_curve_observation_to_fluxes_array_and_label_array,
    pair_array_to_tensor, normalize_tensor_by_modified_z_score, make_uniform_length,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from qusi.internal.light_curve_collection import LightCurveObservationCollection


class OutOfBoundsInjectionHandlingMethod(Enum):
    """
    An enum of approaches for handling cases where the injectable signal is shorter than the injectee signal.
    """

    ERROR = "error"
    REPEAT_SIGNAL = "repeat_signal"
    RANDOM_INJECTION_LOCATION = "random_inject_location"


class LightCurveDataset(IterableDataset):
    """
    A dataset of light curves. Includes cases where light curves can be injected into one another.
    """

    def __init__(
            self,
            standard_light_curve_collections: list[LightCurveObservationCollection],
            *,
            injectee_light_curve_collections: list[LightCurveObservationCollection],
            injectable_light_curve_collections: list[LightCurveObservationCollection],
            post_injection_transform: Callable[[Any], Any],
    ):
        self.standard_light_curve_collections: list[LightCurveObservationCollection] = standard_light_curve_collections
        self.injectee_light_curve_collections: list[LightCurveObservationCollection] = injectee_light_curve_collections
        self.injectable_light_curve_collections: list[
            LightCurveObservationCollection] = injectable_light_curve_collections
        if len(self.standard_light_curve_collections) == 0 and len(self.injectee_light_curve_collections) == 0:
            error_message = (
                "Either the standard or injectee light curve collection lists must not be empty. "
                "Both were empty."
            )
            raise ValueError(error_message)
        self.post_injection_transform: Callable[[Any], Any] = post_injection_transform
        self.worker_randomizing_set: bool = False

    def __iter__(self):
        if not self.worker_randomizing_set:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                self.seed_random(worker_info.id)
            self.worker_randomizing_set = True
        base_light_curve_collection_iter_and_type_pairs: list[
            tuple[Iterator[Path], Callable[[Path], LightCurveObservation], LightCurveCollectionType]
        ] = []
        injectee_collections = copy.copy(self.injectee_light_curve_collections)
        for standard_collection in self.standard_light_curve_collections:
            if standard_collection in injectee_collections:
                base_light_curve_collection_iter_and_type_pairs.append(
                    (
                        loop_iter_function(standard_collection.path_iter),
                        standard_collection.observation_from_path,
                        LightCurveCollectionType.STANDARD_AND_INJECTEE,
                    )
                )
                injectee_collections.remove(standard_collection)
            else:
                base_light_curve_collection_iter_and_type_pairs.append(
                    (
                        loop_iter_function(standard_collection.path_iter),
                        standard_collection.observation_from_path,
                        LightCurveCollectionType.STANDARD,
                    )
                )
        for injectee_collection in injectee_collections:
            base_light_curve_collection_iter_and_type_pair = (
                loop_iter_function(injectee_collection.path_iter),
                injectee_collection.observation_from_path,
                LightCurveCollectionType.INJECTEE,
            )
            base_light_curve_collection_iter_and_type_pairs.append(base_light_curve_collection_iter_and_type_pair)
        injectable_light_curve_collection_iters: list[
            tuple[Iterator[Path], Callable[[Path], LightCurveObservation]]
        ] = []
        for injectable_collection in self.injectable_light_curve_collections:
            injectable_light_curve_collection_iter = loop_iter_function(injectable_collection.path_iter)
            injectable_light_curve_collection_iters.append(
                (injectable_light_curve_collection_iter, injectable_collection.observation_from_path))
        while True:
            for (
                    base_light_curve_collection_iter_and_type_pair
            ) in base_light_curve_collection_iter_and_type_pairs:
                (base_collection_iter, observation_from_path_function,
                 collection_type) = base_light_curve_collection_iter_and_type_pair
                if collection_type in [
                    LightCurveCollectionType.STANDARD,
                    LightCurveCollectionType.STANDARD_AND_INJECTEE,
                ]:
                    # TODO: Preprocessing step should be here. Or maybe that should all be on the light curve collection
                    #  as well? Or passed in somewhere else?
                    standard_path = next(base_collection_iter)
                    standard_light_curve = observation_from_path_function(standard_path)
                    try:
                        transformed_standard_light_curve = self.post_injection_transform(
                            standard_light_curve
                        )
                    except ValueError as error:
                        with Path('problem_light_curves.txt').open('a') as problem_files_list_file:
                            print(f'#############################', flush=True)
                            print(f'{standard_light_curve.path}', file=problem_files_list_file, flush=True)
                        continue
                    yield transformed_standard_light_curve
                if collection_type in [
                    LightCurveCollectionType.INJECTEE,
                    LightCurveCollectionType.STANDARD_AND_INJECTEE,
                ]:
                    for (injectable_light_curve_collection_iter,
                         injectable_observation_from_path_function) in injectable_light_curve_collection_iters:
                        injectable_light_path = next(
                            injectable_light_curve_collection_iter
                        )
                        injectable_light_curve = injectable_observation_from_path_function(injectable_light_path)
                        injectee_light_curve_path = next(base_collection_iter)
                        injectee_light_curve = observation_from_path_function(injectee_light_curve_path)
                        # TODO: Here's where the error occurs.
                        try:
                            injected_light_curve = inject_light_curve(
                                injectee_light_curve, injectable_light_curve
                            )
                        except ValueError as error:
                            with Path('problem_light_curves.txt').open('a') as problem_files_list_file:
                                print(f'#############################', flush=True)
                                print(f'{injectee_light_curve.path}', file=problem_files_list_file, flush=True)
                                print(f'{injectable_light_curve.path}', file=problem_files_list_file, flush=True)
                            continue
                        transformed_injected_light_curve = (
                            self.post_injection_transform(injected_light_curve)
                        )
                        yield transformed_injected_light_curve

    @classmethod
    def new(
            cls,
            standard_light_curve_collections: list[LightCurveObservationCollection] | None = None,
            *,
            injectee_light_curve_collections: list[LightCurveObservationCollection] | None = None,
            injectable_light_curve_collections: list[LightCurveObservationCollection] | None = None,
            post_injection_transform: Callable[[Any], Any] | None = None,
    ) -> Self:
        """
        Creates a new light curve dataset.

        :param standard_light_curve_collections: The light curve collections to be used without injection.
        :param injectee_light_curve_collections: The light curve collections that other light curves will be injected
                                                 into.
        :param injectable_light_curve_collections: The light curve collections that will be injected into other light
                                                   curves.
        :return: The light curve dataset.
        """
        if (
                standard_light_curve_collections is None
                and injectee_light_curve_collections is None
        ):
            error_message = (
                "Either the standard or injectee light curve collection lists must be specified. "
                "Both were `None`."
            )
            raise ValueError(error_message)
        if standard_light_curve_collections is None:
            standard_light_curve_collections = []
        if injectee_light_curve_collections is None:
            injectee_light_curve_collections = []
        if injectable_light_curve_collections is None:
            injectable_light_curve_collections = []
        if post_injection_transform is None:
            post_injection_transform = partial(
                default_light_curve_observation_post_injection_transform, length=3500
            )
        instance = cls(
            standard_light_curve_collections=standard_light_curve_collections,
            injectee_light_curve_collections=injectee_light_curve_collections,
            injectable_light_curve_collections=injectable_light_curve_collections,
            post_injection_transform=post_injection_transform,
        )
        return instance

    def seed_random(self, seed: int):
        for collection_group in [self.standard_light_curve_collections, self.injectee_light_curve_collections,
                                 self.injectable_light_curve_collections]:
            for collection in collection_group:
                collection.path_getter.random_number_generator = Random(seed)


def inject_light_curve(
        injectee_observation: LightCurveObservation,
        injectable_observation: LightCurveObservation,
        *,
        out_of_bounds_injection_handling_method=OutOfBoundsInjectionHandlingMethod.RANDOM_INJECTION_LOCATION,
) -> LightCurveObservation:
    (
        fluxes_with_injected_signal,
        injected_light_curve_times,
        _,
        _,
    ) = inject_signal_into_light_curve_with_intermediates(
        light_curve_times=injectee_observation.light_curve.times,
        light_curve_fluxes=injectee_observation.light_curve.fluxes,
        signal_times=injectable_observation.light_curve.times,
        signal_magnifications=injectable_observation.light_curve.fluxes,
        out_of_bounds_injection_handling_method=out_of_bounds_injection_handling_method,
        baseline_flux_estimation_method=BaselineFluxEstimationMethod.MEDIAN,
    )
    injected_light_curve = LightCurve.new(
        times=injected_light_curve_times,
        fluxes=fluxes_with_injected_signal,
    )
    # TODO: Quickly hacked in times with nans removed. Should be handled elsewhere.
    injected_observation = LightCurveObservation.new(
        light_curve=injected_light_curve, label=injectable_observation.label
    )
    return injected_observation


def is_injected_dataset(dataset: LightCurveDataset):
    return len(dataset.injectee_light_curve_collections) > 0


def contains_injected_dataset(datasets: list[LightCurveDataset]):
    return any(is_injected_dataset(dataset) for dataset in datasets)


def interleave_infinite_iterators(*infinite_iterators: Iterator):
    while True:
        for iterator in infinite_iterators:
            yield next(iterator)


T = TypeVar("T")


def loop_iter_function(iter_function: Callable[[], Iterable[T]]) -> Iterator[T]:
    while True:
        iterator = iter_function()
        yield from iterator


class ObservationType(Enum):
    STANDARD = "standard"
    INJECTEE = "injectee"


class LightCurveCollectionType(Enum):
    STANDARD = "standard"
    INJECTEE = "injectee"
    STANDARD_AND_INJECTEE = "standard_and_injectee"


class InterleavedDataset(IterableDataset):
    def __init__(self, *datasets: IterableDataset):
        self.datasets: tuple[IterableDataset, ...] = datasets

    @classmethod
    def new(cls, *datasets: IterableDataset):
        instance = cls(*datasets)
        return instance

    def __iter__(self):
        # noinspection PyTypeChecker
        dataset_iterators = list(map(iter, self.datasets))
        return interleave_infinite_iterators(*dataset_iterators)


class ConcatenatedIterableDataset(IterableDataset):
    def __init__(self, *datasets: IterableDataset):
        self.datasets: tuple[IterableDataset, ...] = datasets

    @classmethod
    def new(cls, *datasets: IterableDataset):
        instance = cls(*datasets)
        return instance

    def __iter__(self):
        for dataset in self.datasets:
            yield from dataset


class LimitedIterableDataset(IterableDataset):
    def __init__(self, dataset: IterableDataset, limit: int):
        self.dataset: IterableDataset = dataset
        self.limit: int = limit

    @classmethod
    def new(cls, dataset: IterableDataset, limit: int):
        instance = cls(dataset, limit)
        return instance

    def __iter__(self):
        count = 0
        for element in self.dataset:
            yield element
            count += 1
            if count >= self.limit:
                break


def identity_transform(x: LightCurveObservation) -> LightCurveObservation:
    return x


def default_light_curve_observation_post_injection_transform(
        x: LightCurveObservation,
        *,
        length: int,
        randomize: bool = True,
) -> (Tensor, Tensor):
    """
    The default light curve observation post injection transforms. A set of transforms that is expected to work well for
    a variety of use cases.

    :param x: The light curve observation to be transformed.
    :param length: The length to make all light curves.
    :param randomize: Whether to have randomization in the transforms.
    :return: The transformed light curve observation.
    """
    x = remove_nan_flux_data_points_from_light_curve_observation(x)
    if randomize:
        x = randomly_roll_light_curve_observation(x)
    x = from_light_curve_observation_to_fluxes_array_and_label_array(x)
    x = (make_uniform_length(x[0], length=length), x[1])  # Make the fluxes a uniform length.
    x = pair_array_to_tensor(x)
    x = (normalize_tensor_by_modified_z_score(x[0]), x[1])
    return x


def default_light_curve_post_injection_transform(
        x: LightCurve,
        *,
        length: int,
        randomize: bool = True,
) -> Tensor:
    """
    The default light curve post injection transforms. A set of transforms that is expected to work well for a variety
    of use cases.

    :param x: The light curve to be transformed.
    :param length: The length to make all light curves.
    :param randomize: Whether to have randomization in the transforms.
    :return: The transformed light curve.
    """
    x = remove_nan_flux_data_points_from_light_curve(x)
    if randomize:
        x = randomly_roll_light_curve(x)
    x = x.fluxes
    x = make_uniform_length(x, length=length)
    x = torch.tensor(x, dtype=torch.float32)
    x = normalize_tensor_by_modified_z_score(x)
    return x


class BaselineFluxEstimationMethod(Enum):
    """
    An enum of to designate the type of baseline flux estimation method to use during training.
    """

    MEDIAN = "median"
    MEDIAN_ABSOLUTE_DEVIATION = "median_absolute_deviation"


def inject_signal_into_light_curve_with_intermediates(
        light_curve_times: npt.NDArray[np.float64],
        light_curve_fluxes: npt.NDArray[np.float64],
        signal_times: npt.NDArray[np.float64],
        signal_magnifications: npt.NDArray[np.float64],
        out_of_bounds_injection_handling_method: OutOfBoundsInjectionHandlingMethod = (
                OutOfBoundsInjectionHandlingMethod.ERROR),
        baseline_flux_estimation_method: BaselineFluxEstimationMethod = BaselineFluxEstimationMethod.MEDIAN,
) -> (npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]):
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
    light_curve_nan_flux_indexes = np.isnan(light_curve_times) | np.isnan(light_curve_fluxes)
    light_curve_times = light_curve_times[~light_curve_nan_flux_indexes]
    light_curve_fluxes = light_curve_fluxes[~light_curve_nan_flux_indexes]
    signal_nan_flux_indexes = np.isnan(signal_times) | np.isnan(signal_magnifications)
    signal_times = signal_times[~signal_nan_flux_indexes]
    signal_magnifications = signal_magnifications[~signal_nan_flux_indexes]
    # TODO: Remove quick hack of removing nans and add a more proper handling.

    minimum_light_curve_time = np.nanmin(light_curve_times)
    relative_light_curve_times = light_curve_times - minimum_light_curve_time
    relative_signal_times = signal_times - np.nanmin(signal_times)
    signal_time_length = np.nanmax(relative_signal_times)
    light_curve_time_length = np.nanmax(relative_light_curve_times)
    time_length_difference = light_curve_time_length - signal_time_length
    signal_start_offset = (
                                  np.random.random() * time_length_difference
                          ) + minimum_light_curve_time
    offset_signal_times = relative_signal_times + signal_start_offset
    if (
            baseline_flux_estimation_method
            == BaselineFluxEstimationMethod.MEDIAN_ABSOLUTE_DEVIATION
    ):
        baseline_flux = stats.median_abs_deviation(light_curve_fluxes)
        baseline_to_median_absolute_deviation_ratio = (
            10  # Arbitrarily chosen to give a reasonable scale.
        )
        baseline_flux *= baseline_to_median_absolute_deviation_ratio
    else:
        baseline_flux = np.median(light_curve_fluxes)
    signal_fluxes = (signal_magnifications * baseline_flux) - baseline_flux
    if (
            out_of_bounds_injection_handling_method
            is OutOfBoundsInjectionHandlingMethod.RANDOM_INJECTION_LOCATION
    ):
        signal_flux_interpolator = interp1d(
            offset_signal_times, signal_fluxes, bounds_error=False, fill_value=0
        )
    elif (
            out_of_bounds_injection_handling_method
            is OutOfBoundsInjectionHandlingMethod.REPEAT_SIGNAL
            and time_length_difference > 0
    ):
        before_signal_gap = signal_start_offset - minimum_light_curve_time
        after_signal_gap = time_length_difference - before_signal_gap
        minimum_signal_time_step = np.min(np.diff(offset_signal_times))
        before_repeats_needed = math.ceil(
            before_signal_gap / (signal_time_length + minimum_signal_time_step)
        )
        after_repeats_needed = math.ceil(
            after_signal_gap / (signal_time_length + minimum_signal_time_step)
        )
        repeated_signal_fluxes = np.tile(
            signal_fluxes, before_repeats_needed + 1 + after_repeats_needed
        )
        repeated_signal_times = None
        for repeat_index in range(-before_repeats_needed, after_repeats_needed + 1):
            repeat_signal_start_offset = (
                                                 signal_time_length + minimum_signal_time_step
                                         ) * repeat_index
            if repeated_signal_times is None:
                repeated_signal_times = offset_signal_times + repeat_signal_start_offset
            else:
                repeat_index_signal_times = (
                        offset_signal_times + repeat_signal_start_offset
                )
                repeated_signal_times = np.concatenate(
                    [repeated_signal_times, repeat_index_signal_times]
                )
        signal_flux_interpolator = interp1d(
            repeated_signal_times, repeated_signal_fluxes, bounds_error=True
        )
    else:
        signal_flux_interpolator = interp1d(
            offset_signal_times, signal_fluxes, bounds_error=True
        )
    interpolated_signal_fluxes = signal_flux_interpolator(light_curve_times)
    fluxes_with_injected_signal = light_curve_fluxes + interpolated_signal_fluxes
    return fluxes_with_injected_signal, light_curve_times, offset_signal_times, signal_fluxes


def move_path_to_nvme(path: Path) -> Path:
    match = re.match(r"gpu\d{3}", socket.gethostname())
    if match is not None:
        nvme_path = Path("/lscratch/golmsche").joinpath(path)
        if not nvme_path.exists():
            nvme_path.parent.mkdir(exist_ok=True, parents=True)
            nvme_lock_path = nvme_path.parent.joinpath(nvme_path.name + ".lock")
            lock = FileLock(str(nvme_lock_path))
            with lock.acquire():
                if not nvme_path.exists():
                    nvme_tmp_path = nvme_path.parent.joinpath(nvme_path.name + ".tmp")
                    shutil.copy(path, nvme_tmp_path)
                    nvme_tmp_path.rename(nvme_path)
        return nvme_path
    return path
