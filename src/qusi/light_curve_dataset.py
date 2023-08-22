import copy
from enum import Enum
from functools import partial
from typing import List, Iterable, Self, Tuple, TypeVar, Iterator, Callable

from torch.utils.data import IterableDataset
from torchvision import transforms

from qusi.light_curve import LightCurve
from qusi.light_curve_collection import LabeledLightCurveCollection
from qusi.light_curve_observation import LightCurveObservation, remove_nan_flux_data_points_from_light_curve_observation
from qusi.light_curve_transforms import from_observation_to_fluxes_array_and_label_array, \
    pair_array_to_tensor
from ramjet.photometric_database.light_curve_database import make_times_and_fluxes_array_uniform_length, \
    make_times_and_label_array_uniform_length
from ramjet.photometric_database.light_curve_dataset_manipulations import OutOfBoundsInjectionHandlingMethod, \
    BaselineFluxEstimationMethod, inject_signal_into_light_curve_with_intermediates


class LightCurveDataset(IterableDataset):
    """
    A dataset of light curve data.
    """
    def __init__(self,
                 standard_light_curve_collections: List[LabeledLightCurveCollection],
                 injectee_light_curve_collections: List[LabeledLightCurveCollection],
                 injectable_light_curve_collections: List[LabeledLightCurveCollection]
                 ):
        self.standard_light_curve_collections: List[LabeledLightCurveCollection] = standard_light_curve_collections
        self.injectee_light_curve_collections: List[LabeledLightCurveCollection] = injectee_light_curve_collections
        self.injectable_light_curve_collections: List[LabeledLightCurveCollection] = injectable_light_curve_collections
        if len(self.standard_light_curve_collections) == 0 and len(self.injectee_light_curve_collections) == 0:
            raise ValueError('Either the standard or injectee light curve collection lists must not be empty. '
                             'Both were empty.')
        self.include_standard_in_injectee = False  # TODO: Should this be automatically detected?
        self.transforms = [
            remove_nan_flux_data_points_from_light_curve_observation,
            from_observation_to_fluxes_array_and_label_array,
            partial(make_times_and_label_array_uniform_length, length=1000),
            pair_array_to_tensor,
        ]  # TODO: remove hard coded and make available at multiple steps.

    @classmethod
    def new(cls,
            standard_light_curve_collections: List[LabeledLightCurveCollection] | None = None,
            injectee_light_curve_collections: List[LabeledLightCurveCollection] | None = None,
            injectable_light_curve_collections: List[LabeledLightCurveCollection] | None = None,
            ) -> Self:
        if standard_light_curve_collections is None and injectee_light_curve_collections is None:
            raise ValueError('Either the standard or injectee light curve collection lists must be specified. '
                             'Both were `None`.')
        if standard_light_curve_collections is None:
            standard_light_curve_collections = []
        if injectee_light_curve_collections is None:
            injectee_light_curve_collections = []
        if injectable_light_curve_collections is None:
            injectable_light_curve_collections = []
        instance = cls(standard_light_curve_collections=standard_light_curve_collections,
                       injectee_light_curve_collections=injectee_light_curve_collections,
                       injectable_light_curve_collections=injectable_light_curve_collections)
        return instance


    def __iter__(self):
        base_light_curve_collection_iter_and_type_pairs: \
            List[Tuple[Iterator[LightCurveObservation], LightCurveCollectionType]] = []
        injectee_collections = copy.copy(self.injectee_light_curve_collections)
        for standard_collection in self.standard_light_curve_collections:
            if standard_collection in injectee_collections:
                base_light_curve_collection_iter_and_type_pairs.append(
                    (loop_iter_function(standard_collection.observation_iter), LightCurveCollectionType.STANDARD_AND_INJECTEE))
                injectee_collections.remove(standard_collection)
            else:
                base_light_curve_collection_iter_and_type_pairs.append(
                    (loop_iter_function(standard_collection.observation_iter), LightCurveCollectionType.STANDARD))
        for injectee_collection in injectee_collections:
            base_light_curve_collection_iter_and_type_pairs.append(
                (loop_iter_function(injectee_collection.observation_iter), LightCurveCollectionType.INJECTEE))
        injectable_light_curve_collection_iters: List[Iterator[LightCurveObservation]] = []
        for injectable_collection in self.injectable_light_curve_collections:
            injectable_light_curve_collection_iters.append(loop_iter_function(injectable_collection.observation_iter))
        while True:
            for base_light_curve_collection_iter_and_type_pair in base_light_curve_collection_iter_and_type_pairs:
                base_collection_iter, collection_type = base_light_curve_collection_iter_and_type_pair
                if collection_type in [LightCurveCollectionType.STANDARD,
                                       LightCurveCollectionType.STANDARD_AND_INJECTEE]:
                    # TODO: Preprocessing step should be here. Or maybe that should all be on the light curve collection
                    #  as well? Or passed in somewhere else?
                    standard_light_curve = next(base_collection_iter)
                    transformed_standard_light_curve = self.apply_transforms(standard_light_curve)
                    yield transformed_standard_light_curve
                if collection_type in [LightCurveCollectionType.INJECTEE,
                                       LightCurveCollectionType.STANDARD_AND_INJECTEE]:
                    for injectable_light_curve_collection_iter in injectable_light_curve_collection_iters:
                        injectable_light_curve = next(injectable_light_curve_collection_iter)
                        injectee_light_curve = next(base_collection_iter)
                        injected_light_curve = inject_light_curve(injectee_light_curve, injectable_light_curve)
                        transformed_injected_light_curve = self.apply_transforms(injected_light_curve)
                        yield transformed_injected_light_curve


    def apply_transforms(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x


def inject_light_curve(injectee_observation: LightCurveObservation, injectable_observation: LightCurveObservation
                       ) -> LightCurveObservation:
    fluxes_with_injected_signal, _, _ = inject_signal_into_light_curve_with_intermediates(
        light_curve_times=injectee_observation.light_curve.times,
        light_curve_fluxes=injectee_observation.light_curve.fluxes,
        signal_times=injectable_observation.light_curve.times,
        signal_magnifications=injectable_observation.light_curve.fluxes,
        out_of_bounds_injection_handling_method=OutOfBoundsInjectionHandlingMethod.RANDOM_INJECTION_LOCATION,
        baseline_flux_estimation_method=BaselineFluxEstimationMethod.MEDIAN
    )
    injected_light_curve = LightCurve.new(times=injectee_observation.light_curve.times,
                                          fluxes=injectee_observation.light_curve.fluxes)
    injected_observation = LightCurveObservation.new(light_curve=injected_light_curve,
                                                     label=injectable_observation.label)
    return injected_observation


def is_injected_dataset(dataset: LightCurveDataset):
    return len(dataset.injectee_light_curve_collections) > 0


def contains_injected_dataset(datasets: List[LightCurveDataset]):
    for dataset in datasets:
        if is_injected_dataset(dataset):
            return True
    return False


def interleave_infinite_iterators(*infinite_iterators: Iterator):
    while True:
        for iterator in infinite_iterators:
            yield next(iterator)


T = TypeVar('T')


def loop_iter_function(iter_function: Callable[[], Iterable[T]]) -> Iterator[T]:
    while True:
        iterator = iter_function()
        for element in iterator:
            yield element


class ObservationType(Enum):
    STANDARD = 'standard'
    INJECTEE = 'injectee'


class LightCurveCollectionType(Enum):
    STANDARD = 'standard'
    INJECTEE = 'injectee'
    STANDARD_AND_INJECTEE = 'standard_and_injectee'


class InterleavedDataset(IterableDataset):
    def __init__(self, *datasets: IterableDataset):
        self.datasets: Tuple[IterableDataset] = datasets

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
        self.datasets: Tuple[IterableDataset] = datasets

    @classmethod
    def new(cls, *datasets: IterableDataset):
        instance = cls(*datasets)
        return instance

    def __iter__(self):
        for dataset in self.datasets:
            for element in dataset:
                yield element


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
