"""Code for inference on the contents of a directory."""
import tensorflow as tf
from pathlib import Path

from ramjet.analysis.lightcurve_visualizer import plot_lightcurve
from ramjet.analysis.model_loader import get_latest_log_directory
from ramjet.models import ConvolutionalLstm
from ramjet.photometric_database.toi_lightcurve_database import ToiLightcurveDatabase

saved_log_directory = get_latest_log_directory('logs')  # Uses the latest log directory's model.
# saved_log_directory = Path('logs/baseline YYYY-MM-DD-hh-mm-ss')  # Specifies a specific log directory's model to use.

print('Setting up dataset...')
database = ToiLightcurveDatabase()
database.obtain_meta_data_frame_for_available_lightcurves()
example_paths = [str(database.lightcurve_directory.joinpath('tess2018319095959-s0005-0000000117979897-0125-s_lc.fits'))]
# Uncomment below to run the inference for all validation files.
# example_paths = pd.read_csv(saved_log_directory.joinpath('validation.csv'), header=None)[0].values

print('Loading model...')
model = ConvolutionalLstm()
model.load_weights(str(saved_log_directory.joinpath('model.ckpt')))

print('Inferring and plotting...')
for example_path in example_paths:
    example, label = database.evaluation_preprocessing(tf.convert_to_tensor(example_path))
    prediction = model.predict(tf.expand_dims(example, axis=0))[0]
    fluxes, times = database.load_fluxes_and_times_from_fits_file(example_path)
    label, prediction = database.inference_postprocessing(label, prediction, times.shape[0])
    tic_id = database.get_tic_id_from_single_sector_obs_id(Path(example_path).stem)
    sector = database.get_sector_from_single_sector_obs_id(Path(example_path).stem)
    plot_title = f'TIC {tic_id} sector {sector}'
    plot_lightcurve(times, fluxes, label, prediction, title=plot_title, save_path=f'{plot_title}.png')
