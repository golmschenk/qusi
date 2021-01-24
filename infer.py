"""Code for inference on the contents of a directory."""

import datetime
from pathlib import Path

from ramjet.models.hades import Hades, FfiHades
from ramjet.photometric_database.derived.tess_ffi_transit_databases import \
    TessFfiStandardAndInjectedTransitAntiEclipsingBinaryDatabase
from ramjet.photometric_database.derived.tess_two_minute_cadence_transit_databases import \
    TessTwoMinuteCadenceStandardAndInjectedTransitDatabase
from ramjet.analysis.model_loader import get_latest_log_directory
from ramjet.trial import infer

# log_name = get_latest_log_directory(logs_directory='logs')  # Uses the latest model in the log directory.
log_name = 'logs/FFI transit sai aeb FfiHades mag14 quick pos no neg cont from existing no random start 2020-12-19-16-10-27'  # Specify the path to the model to use.
saved_log_directory = Path(f'{log_name}')
datetime_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

print('Setting up dataset...', flush=True)
database = TessFfiStandardAndInjectedTransitAntiEclipsingBinaryDatabase()
inference_dataset = database.generate_inference_dataset()

print('Loading model...', flush=True)
model = FfiHades()
model.load_weights(str(saved_log_directory.joinpath('model.ckpt'))).expect_partial()

print('Inferring...', flush=True)
infer_results_path = saved_log_directory.joinpath(f'infer results {datetime_string}.csv')
infer(model, inference_dataset, infer_results_path, number_of_top_predictions_to_keep=5000)
