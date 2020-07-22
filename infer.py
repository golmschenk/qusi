"""Code for inference on the contents of a directory."""

import datetime
import io
import numpy as np
import pandas as pd
from pathlib import Path

from ramjet.photometric_database.derived.tess_two_minute_cadence_lightcurve_collection import \
    TessTwoMinuteCadenceTargetDatasetSplitLightcurveCollection
from ramjet.photometric_database.derived.tess_two_minute_cadence_transit_databases import \
    TessTwoMinuteCadenceStandardTransitDatabase
from ramjet.analysis.model_loader import get_latest_log_directory
from ramjet.basic_models import SimpleLightcurveCnn

log_name = get_latest_log_directory(logs_directory='logs')  # Uses the latest model in the log directory.
# log_name = 'logs/baseline YYYY-MM-DD-hh-mm-ss'  # Specify the path to the model to use.
saved_log_directory = Path(f'{log_name}')
datetime_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

print('Setting up dataset...', flush=True)
database = TessTwoMinuteCadenceStandardTransitDatabase()  # Database of preprocessing to use.
lightcurve_collection = TessTwoMinuteCadenceTargetDatasetSplitLightcurveCollection()  # Lightcurves to infer on.
example_paths_dataset = database.generate_paths_dataset_from_lightcurve_collection(lightcurve_collection)
examples_dataset = database.generate_infer_path_and_lightcurve_dataset(
    example_paths_dataset, lightcurve_collection.load_times_and_fluxes_from_path)
batch_dataset = examples_dataset.batch(database.batch_size).prefetch(5)

print('Loading model...', flush=True)
model = SimpleLightcurveCnn()
model.load_weights(str(saved_log_directory.joinpath('model.ckpt'))).expect_partial()

print('Inferring...', flush=True)
number_of_top_predictions_to_keep = 5000  # Saving all predictions is usually unnecessary.
columns = ['Lightcurve path', 'Prediction']
dtypes = [str, int]
predictions_data_frame = pd.read_csv(io.StringIO(''), names=columns, dtype=dict(zip(columns, dtypes)))
old_top_predictions_data_frame = predictions_data_frame
for batch_index, (paths, examples) in enumerate(batch_dataset):
    predictions = model.predict(examples)
    batch_predictions = pd.DataFrame({'Lightcurve path': paths.numpy().astype(str),
                                      'Prediction': np.squeeze(predictions, axis=1)})
    predictions_data_frame = pd.concat([predictions_data_frame, batch_predictions])
    print(f'{batch_index * database.batch_size} examples inferred on.', flush=True)
    if batch_index % 100 == 0:  # Drop all but top results and updated saved file every 100 batches.
        predictions_data_frame = predictions_data_frame.sort_values(
            'Prediction', ascending=False).head(number_of_top_predictions_to_keep).reset_index(drop=True)
        predictions_data_frame.to_csv(
            saved_log_directory.joinpath(f'infer results {datetime_string}.csv'), index_label='Index')
# Drop all but top results and saved file.
predictions_data_frame = predictions_data_frame.sort_values(
    'Prediction', ascending=False).head(number_of_top_predictions_to_keep).reset_index(drop=True)
predictions_data_frame.to_csv(
    saved_log_directory.joinpath(f'infer results {datetime_string}.csv'), index_label='Index')
