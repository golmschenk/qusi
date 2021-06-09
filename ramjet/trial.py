"""
Boilerplate code for running trials.
"""
import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from typing import List
from pathlib import Path
try:
    from enum import StrEnum
except ImportError:
    from backports.strenum import StrEnum

import wandb.keras
from tensorflow.python.keras import callbacks

from ramjet.logging.wandb_logger import WandbLogger
from ramjet.photometric_database.standard_and_injected_lightcurve_database import StandardAndInjectedLightcurveDatabase


class LoggingToolName(StrEnum):
    WANDB = 'wandb'
    TENSORBOARD = 'tensorboard'


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
                 number_of_top_predictions_to_keep: int = None) -> pd.DataFrame:
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


def create_logging_metrics() -> List[tf.metrics.Metric]:
    """
    Creates the standard metrics to be used in logging.

    :return: The list of metrics.
    """
    metrics = [tf.keras.metrics.AUC(num_thresholds=20, name='Area_under_ROC_curve', multi_label=True),
               tf.metrics.SpecificityAtSensitivity(0.9, name='Specificity_at_90_percent_sensitivity'),
               tf.metrics.SensitivityAtSpecificity(0.9, name='Sensitivity_at_90_percent_specificity'),
               tf.metrics.BinaryAccuracy(name='Accuracy'),
               tf.metrics.Precision(name='Precision'),
               tf.metrics.Recall(name='Recall')]
    return metrics


def create_logging_callbacks(logs_directory: Path, trial_name: str, database: StandardAndInjectedLightcurveDatabase,
                             logging_tool_name: LoggingToolName = LoggingToolName.WANDB) -> List[callbacks.Callback]:
    """
    Creates the callbacks to perform the logging.

    :param logs_directory: The directory to log to.
    :param trial_name: The name of the trial.
    :return: The callbacks to perform the logging.
    """
    datetime_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    trial_directory = logs_directory.joinpath(f'{trial_name} {datetime_string}')
    trial_directory.mkdir(exist_ok=True, parents=True)
    latest_model_save_path = trial_directory.joinpath('latest_model.ckpt')
    latest_checkpoint_callback = callbacks.ModelCheckpoint(latest_model_save_path, save_weights_only=True)
    best_validation_model_save_path = trial_directory.joinpath('best_validation_model.ckpt')
    best_validation_checkpoint_callback = callbacks.ModelCheckpoint(
        best_validation_model_save_path, monitor='Area_under_ROC_curve', mode='max', save_best_only=True,
        save_weights_only=True)
    logging_callbacks = [latest_checkpoint_callback, best_validation_checkpoint_callback]
    if logging_tool_name == 'tensorboard':
        tensorboard_callback = callbacks.TensorBoard(log_dir=trial_directory)
        logging_callbacks.append(tensorboard_callback)
    else:
        logger = WandbLogger.new(trial_directory)
        database.logger = logger
        logging_callbacks.extend([logger.create_callback(), wandb.keras.WandbCallback()])
    return logging_callbacks
