import itertools
from itertools import islice
from unittest.mock import Mock

from qusi.internal.light_curve_dataset import (
    contains_injected_dataset,
    interleave_infinite_iterators,
    is_injected_dataset,
)
from tests.iterable_mock import IterableMock


def test_is_injected_dataset():
    non_injected_dataset = Mock()
    non_injected_dataset.injectee_light_curve_collections = []

    assert not is_injected_dataset(non_injected_dataset)

    injected_dataset = Mock()
    injected_dataset.injectee_light_curve_collections = [Mock()]

    assert is_injected_dataset(injected_dataset)


def test_contains_injected_dataset():
    non_injected_dataset = Mock()
    non_injected_dataset.injectee_light_curve_collections = []
    injected_dataset = Mock()
    injected_dataset.injectee_light_curve_collections = [Mock()]

    datasets_without_injected = [
        non_injected_dataset,
        non_injected_dataset,
        non_injected_dataset,
    ]
    assert not contains_injected_dataset(datasets_without_injected)

    datasets_with_injected = [
        non_injected_dataset,
        injected_dataset,
        non_injected_dataset,
    ]
    assert contains_injected_dataset(datasets_with_injected)


def test_interleave_infinite_iterators():
    dataset0 = IterableMock()
    dataset0.__iter__.return_value = itertools.cycle(iter([1, 2, 3]))
    dataset1 = IterableMock()
    dataset1.__iter__.return_value = itertools.cycle(iter(["a", "b"]))
    combined_dataset = interleave_infinite_iterators(iter(dataset0), iter(dataset1))
    first_eight_elements = list(islice(combined_dataset, 8))
    assert first_eight_elements == [1, "a", 2, "b", 3, "a", 1, "b"]
