"""
Code for TensorFlow's `Dataset` class which allows for multiprocessing in CPU map functions.
"""
import multiprocessing
from typing import Callable, Union, List
import signal
import tensorflow as tf


class PyMapper:
    """
    A class which allows for mapping a py_function to a TensorFlow dataset in parallel on CPU.
    """
    def __init__(self, map_function: Callable, number_of_parallel_calls: int):
        self.map_function = map_function
        self.number_of_parallel_calls = number_of_parallel_calls
        self.pool = multiprocessing.Pool(self.number_of_parallel_calls, self.pool_worker_initializer)

    @staticmethod
    def pool_worker_initializer():
        """
        Used to initialize each worker process.
        """
        # Corrects bug where worker instances catch and throw away keyboard interrupts.
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    def send_to_map_pool(self, *example_elements):
        """
        Sends the tensor element to the pool for processing.

        :param example_elements: The elements list to be processed by the pool. That is, each example_elements
                                 is the contents of a single example in the dataset. Often this may be a single element.
        :return: The output of the map function on the element.
        """
        import pydevd
        pydevd.settrace(suspend=False)
        result = self.pool.apply_async(self.map_function, example_elements)
        mapped_element = result.get()
        return mapped_element

    def map_to_dataset(self, dataset: tf.data.Dataset,
                       output_types: Union[List[tf.dtypes.DType], tf.dtypes.DType] = tf.float32,
                       flat_map: bool = False):
        """
        Maps the map function to the passed dataset.

        :param dataset: The dataset to apply the map function to.
        :param output_types: The TensorFlow output types of the function to convert to.
        :param flat_map: Determines whether to flatten the first level of the output, similar to TensorFlow's
                         `flat_map`. Note, the `output_types` should be the shape of the unflattened output.
        :return: The mapped dataset.
        """
        def map_py_function(*args):
            """A py_function wrapper for the map function."""
            return tf.py_function(self.send_to_map_pool, args, output_types)

        mapped_dataset = dataset.map(map_py_function, self.number_of_parallel_calls)
        if flat_map:
            return mapped_dataset.flat_map(lambda elements: tf.data.Dataset.from_tensor_slices(elements))
        else:
            return mapped_dataset


def map_py_function_to_dataset(dataset: tf.data.Dataset, map_function: Callable, number_of_parallel_calls: int,
                               output_types: Union[List[tf.dtypes.DType], tf.dtypes.DType] = tf.float32,
                               flat_map: bool = False) -> tf.data.Dataset:
    """
    A one line wrapper to allow mapping a parallel py function to a dataset.

    :param dataset: The dataset whose elements the mapping function will be applied to.
    :param map_function: The function to map to the dataset.
    :param number_of_parallel_calls: The number of parallel calls of the mapping function.
    :param output_types: The TensorFlow output types of the function to convert to.
    :param flat_map: Determines whether to flatten the first level of the output, similar to TensorFlow's
                     `flat_map`. Note, the `output_types` should be the shape of the unflattened output.
    :return: The mapped dataset.
    """
    py_mapper = PyMapper(map_function=map_function, number_of_parallel_calls=number_of_parallel_calls)
    mapped_dataset = py_mapper.map_to_dataset(dataset=dataset, output_types=output_types, flat_map=flat_map)
    return mapped_dataset
