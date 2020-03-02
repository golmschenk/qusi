"""Tests for the PyMapper class."""
import os
import time
import pytest
import numpy as np
import tensorflow as tf

from ramjet.photometric_database.py_mapper import PyMapper, map_py_function_to_dataset


class TestPyMapper:
    """Tests for the PyMapper class."""

    @pytest.fixture
    def dataset(self) -> tf.data.Dataset:
        """
        Sets up the dataset for use in a test.

        :return: The dataset.
        """
        return tf.data.Dataset.from_tensor_slices([0, 10, 20, 30])

    @pytest.mark.slow
    def test_py_map_runs_function_on_multiple_processes(self, dataset: tf.data.Dataset):
        py_mapper = PyMapper(sleep_and_get_pid, number_of_parallel_calls=4)
        map_dataset = py_mapper.map_to_dataset(dataset)
        batch_dataset = map_dataset.batch(batch_size=4)
        batch = next(iter(batch_dataset))
        batch_array = batch.numpy()
        unique_pids = set(batch_array)
        assert len(unique_pids) == 4

    def test_py_map_correctly_applies_map_function(self, dataset: tf.data.Dataset):
        py_mapper = PyMapper(add_one, number_of_parallel_calls=4)
        map_dataset = py_mapper.map_to_dataset(dataset)
        batch_dataset = map_dataset.batch(batch_size=4)
        batch = next(iter(batch_dataset))
        batch_array = batch.numpy()
        assert np.array_equal(batch_array, np.array([1, 11, 21, 31]))

    def test_py_map_correctly_applies_map_function_with_two_outputs(self, dataset: tf.data.Dataset):
        py_mapper = PyMapper(add_one_and_add_two, number_of_parallel_calls=4)
        map_dataset = py_mapper.map_to_dataset(dataset, output_types=[tf.float32, tf.float32])
        batch_dataset = map_dataset.batch(batch_size=4)
        batch = next(iter(batch_dataset))
        plus_one_array = batch[0].numpy()
        plus_two_array = batch[1].numpy()
        assert np.array_equal(plus_one_array, np.array([1, 11, 21, 31]))
        assert np.array_equal(plus_two_array, np.array([2, 12, 22, 32]))

    def test_py_map_correctly_applies_map_function_with_two_inputs(self, dataset: tf.data.Dataset):
        py_mapper = PyMapper(add_tensors, number_of_parallel_calls=4)
        zipped_dataset = tf.data.Dataset.zip((dataset, dataset))
        map_dataset = py_mapper.map_to_dataset(zipped_dataset)
        batch_dataset = map_dataset.batch(batch_size=4)
        batch = next(iter(batch_dataset))
        batch_array = batch.numpy()
        assert np.array_equal(batch_array, np.array([0, 20, 40, 60]))

    def test_single_function_wrapper(self, dataset):
        mapped_dataset = map_py_function_to_dataset(dataset=dataset, map_function=add_one, number_of_parallel_calls=4,
                                                    output_types=tf.float32)
        batch_dataset = mapped_dataset.batch(batch_size=4)
        batch = next(iter(batch_dataset))
        batch_array = batch.numpy()
        assert np.array_equal(batch_array, np.array([1, 11, 21, 31]))

    def test_py_map_can_be_applied_as_flat_map(self):
        dataset = tf.data.Dataset.from_tensor_slices([[[0, 0], [10, 10]], [[20, 20], [30, 30]]])
        py_mapper = PyMapper(add_one, number_of_parallel_calls=4)
        map_dataset = py_mapper.map_to_dataset(dataset, flat_map=True)
        batch_dataset = map_dataset.batch(batch_size=4)
        batch = next(iter(batch_dataset))
        batch_array = batch.numpy()
        assert np.array_equal(batch_array, np.array([[1, 1], [11, 11], [21, 21], [31, 31]]))

    def test_flat_map_can_be_used_from_single_function_wrapper(self):
        dataset = tf.data.Dataset.from_tensor_slices([[[0, 0], [10, 10]], [[20, 20], [30, 30]]])
        mapped_dataset = map_py_function_to_dataset(dataset=dataset, map_function=add_one, number_of_parallel_calls=4,
                                                    output_types=tf.float32, flat_map=True)
        batch_dataset = mapped_dataset.batch(batch_size=4)
        batch = next(iter(batch_dataset))
        batch_array = batch.numpy()
        assert np.array_equal(batch_array, np.array([[1, 1], [11, 11], [21, 21], [31, 31]]))

    def test_py_map_returns_specified_shape(self):
        dataset = tf.data.Dataset.from_tensor_slices([[0, 0, 0], [1, 1, 1]])
        py_mapper = PyMapper(add_one, number_of_parallel_calls=4)
        map_dataset = py_mapper.map_to_dataset(dataset, output_shapes=(3,))
        assert map_dataset.element_spec.shape == (3,)

    def test_py_map_returns_specified_shape_for_multiple_elements(self):
        dataset0 = tf.data.Dataset.from_tensor_slices([[0, 0, 0], [1, 1, 1]])
        dataset1 = tf.data.Dataset.from_tensor_slices([[2, 2], [3, 3]])
        zipped_dataset = tf.data.Dataset.zip((dataset0, dataset1))
        py_mapper = PyMapper(add_one, number_of_parallel_calls=4)
        map_dataset = py_mapper.map_to_dataset(zipped_dataset, output_types=[tf.float32, tf.float32],
                                               output_shapes=[(3,), (2,)])
        assert map_dataset.element_spec[0].shape == (3,)
        assert map_dataset.element_spec[1].shape == (2,)

    def test_py_map_returns_specified_shape_when_shape_passed_in_wrapper(self):
        dataset = tf.data.Dataset.from_tensor_slices([[0, 0, 0], [1, 1, 1]])
        map_dataset = map_py_function_to_dataset(dataset=dataset, map_function=add_one, number_of_parallel_calls=4,
                                                 output_shapes=(3,))
        assert map_dataset.element_spec.shape == (3,)

    def test_flat_map_with_output_shapes_are_applied_in_the_correct_order(self):
        dataset = tf.data.Dataset.from_tensor_slices([[[0, 0], [10, 10]], [[20, 20], [30, 30]]])
        mapped_dataset = map_py_function_to_dataset(dataset=dataset, map_function=add_one, number_of_parallel_calls=4,
                                                    output_types=tf.float32, flat_map=True, output_shapes=(2,))
        batch_dataset = mapped_dataset.batch(batch_size=4)
        batch = next(iter(batch_dataset))
        batch_array = batch.numpy()
        assert np.array_equal(batch_array, np.array([[1, 1], [11, 11], [21, 21], [31, 31]]))


def sleep_and_get_pid(element_tensor: tf.Tensor) -> int:
    """
    A simple sleep and get pid function to test multiprocessing.

    :param element_tensor: Input value.
    :return: The pid of the process that ran this function.
    """
    time.sleep(0.1)
    return os.getpid()


# noinspection PyPackageRequirements
def add_tensors(element_tensor0: tf.Tensor, element_tensor1: tf.Tensor) -> float:
    """
    Adds two elements together.

    :param element_tensor0: First input value.
    :param element_tensor1: Second input value.
    :return: The added inputs.
    """
    element0 = element_tensor0.numpy()
    element1 = element_tensor1.numpy()
    return element0 + element1


def add_one(element_tensor: tf.Tensor) -> float:
    """
    Adds 1

    :param element_tensor: Input value.
    :return: Input plus 1.
    """
    element = element_tensor.numpy()
    return element + 1


def add_one_and_add_two(element_tensor: tf.Tensor) -> (float, float):
    """
    Adds 1 and adds 2 returning both.

    :param element_tensor: Input value.
    :return: Input plus 1 and input plus 2.
    """
    element = element_tensor.numpy()
    return element + 1, element + 2
