"""
Boilerplate code for running trials.
"""
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
    confidences_data_frame = None
    examples_count = 0
    for batch_index, (paths, examples) in enumerate(dataset):
        confidences = model(examples, training=False)
        if confidences.shape[1] == 1:
            batch_confidences_data_frame = pd.DataFrame({'light_curve_path': paths.numpy().astype(str),
                                                         'confidence': np.squeeze(confidences, axis=1)})
        else:
            batch_confidences_data_frame = pd.DataFrame({'light_curve_path': paths.numpy().astype(str)})
            for label_index in range(confidences.shape[1]):
                batch_confidences_data_frame[f'label_{label_index}_confidence'] = confidences[:, label_index]
        examples_count += batch_confidences_data_frame.shape[0]
        if confidences_data_frame is None:
            confidences_data_frame = batch_confidences_data_frame
        else:
            confidences_data_frame = pd.concat([confidences_data_frame, batch_confidences_data_frame])
        print(f'{examples_count} examples inferred on.', flush=True)
        if number_of_top_predictions_to_keep is not None and batch_index % 100 == 0:
            confidences_data_frame = save_results(confidences_data_frame, infer_results_path,
                                                  number_of_top_predictions_to_keep)
    save_results(confidences_data_frame, infer_results_path, number_of_top_predictions_to_keep)


def save_results(confidences_data_frame: pd.DataFrame, infer_results_path: Path,
                 number_of_top_predictions_to_keep: int = None):
    """
    Saves a predictions data frame to a file.

    :param confidences_data_frame: The data frame of predictions to save.
    :param infer_results_path: The path to save the resulting predictions to.
    :param number_of_top_predictions_to_keep: The number of top results to keep. None will save all results.
    :return: The updated data frame.
    """
    try:
        confidences_data_frame = confidences_data_frame.sort_values('confidence', ascending=False)
    except KeyError:
        confidences_data_frame = confidences_data_frame.sort_values('label_0_confidence', ascending=False)
    if number_of_top_predictions_to_keep is not None:
        confidences_data_frame = confidences_data_frame.head(number_of_top_predictions_to_keep)
    confidences_data_frame = confidences_data_frame.reset_index(drop=True)
    confidences_data_frame.to_csv(infer_results_path, index_label='index')
    return confidences_data_frame
