from unittest.mock import Mock

from qusi.light_curve_dataset import is_injected_dataset, contains_injected_dataset


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

    datasets_without_injected = [non_injected_dataset, non_injected_dataset, non_injected_dataset]
    assert not contains_injected_dataset(datasets_without_injected)

    datasets_with_injected = [non_injected_dataset, injected_dataset, non_injected_dataset]
    assert contains_injected_dataset(datasets_with_injected)

