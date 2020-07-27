"""
Boilerplate code for running trials.
"""

import io
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path


def infer(model: tf.keras.Model, dataset: tf.data.Dataset, infer_results_path: Path,
          number_of_top_predictions_to_keep: int = None):
    """
    Performs inference of a model on a dataset saving the results to a file.

    :param model: The model to infer with.
    :param dataset: The dataset to infer on.
    :param infer_results_path: The path to save the resulting predictions to.
    :param number_of_top_predictions_to_keep: The number of top results to keep. None will save all results.
    """
    columns = ['Lightcurve path', 'Prediction']
    dtypes = [str, int]
    predictions_data_frame = pd.read_csv(io.StringIO(''), names=columns, dtype=dict(zip(columns, dtypes)))
    examples_count = 0
    for batch_index, (paths, examples) in enumerate(dataset):
        predictions = model(examples, training=False)
        batch_predictions = pd.DataFrame({'Lightcurve path': paths.numpy().astype(str),
                                          'Prediction': np.squeeze(predictions, axis=1)})
        examples_count += batch_predictions.shape[0]
        predictions_data_frame = pd.concat([predictions_data_frame, batch_predictions])
        print(f'{examples_count} examples inferred on.', flush=True)
        if number_of_top_predictions_to_keep is not None and batch_index % 100 == 0:
            predictions_data_frame = save_results(predictions_data_frame, infer_results_path,
                                                  number_of_top_predictions_to_keep)
    save_results(predictions_data_frame, infer_results_path, number_of_top_predictions_to_keep)


def save_results(predictions_data_frame: pd.DataFrame, infer_results_path: Path,
                 number_of_top_predictions_to_keep: int = None):
    """
    Saves a predictions data frame to a file.

    :param predictions_data_frame: The data frame of predictions to save.
    :param infer_results_path: The path to save the resulting predictions to.
    :param number_of_top_predictions_to_keep: The number of top results to keep. None will save all results.
    :return:
    """
    predictions_data_frame = predictions_data_frame.sort_values('Prediction', ascending=False)
    if number_of_top_predictions_to_keep is not None:
        predictions_data_frame = predictions_data_frame.head(number_of_top_predictions_to_keep)
    predictions_data_frame = predictions_data_frame.reset_index(drop=True)
    predictions_data_frame.to_csv(infer_results_path, index_label='Index')
    return predictions_data_frame
