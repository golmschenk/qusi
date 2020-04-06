"""Code for inference on the contents of a directory."""

import datetime
import io
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

from ramjet.py_mapper import map_py_function_to_dataset
from ramjet.analysis.model_loader import get_latest_log_directory
from ramjet.models import SimpleLightcurveCnn
from ramjet.photometric_database.toi_database import ToiDatabase

log_name = get_latest_log_directory(logs_directory='logs')  # Uses the latest model in the log directory.
# log_name = 'baseline YYYY-MM-DD-hh-mm-ss'  # Specify the path to the model to use.
saved_log_directory = Path(f'logs/{log_name}')
datetime_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

print('Setting up dataset...')
database = ToiDatabase()
example_paths = database.get_all_lightcurve_paths()
example_paths_dataset = database.paths_dataset_from_list_or_generator_factory(example_paths)
mapped_dataset = map_py_function_to_dataset(example_paths_dataset, database.infer_preprocessing,
                                            number_of_parallel_calls=database.number_of_parallel_processes_per_map,
                                            output_types=(tf.string, tf.float32))
batch_dataset = mapped_dataset.batch(database.batch_size).prefetch(5)

print('Loading model...')
model = SimpleLightcurveCnn()
model.load_weights(str(saved_log_directory.joinpath('model.ckpt'))).expect_partial()

print('Inferring and plotting...')
columns = ['Lightcurve path', 'Prediction']
dtypes = [str, int]
predictions_data_frame = pd.read_csv(io.StringIO(''), names=columns, dtype=dict(zip(columns, dtypes)))
old_top_predictions_data_frame = predictions_data_frame
for batch_index, (paths, examples) in enumerate(batch_dataset):
    predictions = model.predict(examples)
    batch_predictions = pd.DataFrame({'Lightcurve path': paths, 'Prediction': np.squeeze(predictions, axis=1)})
    predictions_data_frame = pd.concat([predictions_data_frame, batch_predictions])
    print(f'{batch_index * database.batch_size} examples inferred on.')
predictions_data_frame.sort_values('Prediction', ascending=False).reset_index().to_feather(
    f'{log_name} {datetime_string}.feather'
)
