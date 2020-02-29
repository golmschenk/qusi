"""
Code for TensorFlow's `Dataset` class which allows for multiprocessing in CPU map functions.
"""
import multiprocessing
from typing import Callable, Union, List, Tuple
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
        result = self.pool.apply_async(self.map_function, example_elements)
        mapped_element = result.get()
        return mapped_element

    def map_to_dataset(self, dataset: tf.data.Dataset,
                       output_types: Union[List[tf.dtypes.DType], tf.dtypes.DType] = tf.float32,
                       output_shapes: Union[List[Tuple[int, ...]], Tuple[int, ...]] = None,
                       flat_map: bool = False):
        """
        Maps the map function to the passed dataset.

        :param dataset: The dataset to apply the map function to.
        :param output_types: The TensorFlow output types of the function to convert to.
        :param output_shapes: The shape of the outputs of the dataset.
        :param flat_map: Determines whether to flatten the first level of the output, similar to TensorFlow's
                         `flat_map`. Note, the `output_types` should be the shape of the unflattened output.
        :return: The mapped dataset.
        """
        def map_py_function(*args):
            """A py_function wrapper for the map function."""
            py_function = tf.py_function(self.send_to_map_pool, args, output_types)
            return py_function

        def flat_map_function(*args):
            """A method to flatten the first dimension of datasets, including zipped ones."""
            if len(args) == 1:
                return tf.data.Dataset.from_tensor_slices(args[0])
            else:
                return tf.data.Dataset.zip(tuple(tf.data.Dataset.from_tensor_slices(arg) for arg in args))

        def set_shape_function(*args):
            """A method to shape a dataset's output."""
            # TensorFlow doesn't like iterating over the Tensor. This is a work around. There's probably a better
            # solution.
            if len(args) == 1:
                args[0].set_shape(output_shapes)
            elif len(args) == 2:
                args[0].set_shape(output_shapes[0])
                args[1].set_shape(output_shapes[1])
            elif len(args) == 4:
                args[0].set_shape(output_shapes[0])
                args[1].set_shape(output_shapes[1])
                args[2].set_shape(output_shapes[2])
                args[3].set_shape(output_shapes[3])
            else:
                raise NotImplementedError

            if len(args) == 1:
                return args[0]
            else:
                return args

        mapped_dataset = dataset.map(map_py_function, self.number_of_parallel_calls)
        if output_shapes is not None:
            assert isinstance(output_types, tf.DType) or len(output_types) == len(output_shapes)
            mapped_dataset = mapped_dataset.map(set_shape_function)
        if flat_map:
            return mapped_dataset.flat_map(flat_map_function)
        else:
            return mapped_dataset


def map_py_function_to_dataset(dataset: tf.data.Dataset, map_function: Callable, number_of_parallel_calls: int,
                               output_types: Union[Tuple[tf.dtypes.DType, ...], tf.dtypes.DType] = tf.float32,
                               output_shapes: Union[List[Tuple[int, ...]], Tuple[int, ...]] = None,
                               flat_map: bool = False) -> tf.data.Dataset:
    """
    A one line wrapper to allow mapping a parallel py function to a dataset.

    :param dataset: The dataset whose elements the mapping function will be applied to.
    :param map_function: The function to map to the dataset.
    :param number_of_parallel_calls: The number of parallel calls of the mapping function.
    :param output_types: The TensorFlow output types of the function to convert to.
    :param output_shapes: The shape to set the outputs to clarify from Python to TensorFlow.
    :param flat_map: Determines whether to flatten the first level of the output, similar to TensorFlow's
                     `flat_map`. Note, the `output_types` should be the shape of the un-flattened output.
    :return: The mapped dataset.
    """
    py_mapper = PyMapper(map_function=map_function, number_of_parallel_calls=number_of_parallel_calls)
    mapped_dataset = py_mapper.map_to_dataset(dataset=dataset, output_types=output_types, output_shapes=output_shapes,
                                              flat_map=flat_map)
    return mapped_dataset
