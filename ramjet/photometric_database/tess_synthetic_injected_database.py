"""
Code to represent the database for injecting synthetic signals into real TESS data.
"""
from pathlib import Path

from ramjet.photometric_database.lightcurve_database import LightcurveDatabase


class TessSyntheticInjectedDatabase(LightcurveDatabase):
    def __init__(self):
        super().__init__()
        self.data_directory: Path = Path('data/microlensing')
        self.lightcurve_directory: Path = self.data_directory.joinpath('lightcurves')
        self.synthetic_signal_directory: Path = self.data_directory.joinpath('synthetic_signals')

    def generate_datasets(self):
        all_lightcurve_paths = list(self.lightcurve_directory.glob('*.fits'))
        all_synthetic_paths = list(self.synthetic_signal_directory.glob('*.feather'))
        lightcurve_paths_datasets = self.get_training_and_validation_datasets_for_file_paths(all_lightcurve_paths)
        training_lightcurve_paths_dataset, validation_lightcurve_paths_dataset = lightcurve_paths_datasets
        shuffled_training_dataset = training_lightcurve_paths_dataset.shuffle(
            buffer_size=len(training_lightcurve_paths_dataset))
        lightcurve_training_dataset = shuffled_training_dataset.map(self.training_preprocessing, num_parallel_calls=16)
        injected_and_not_injected_lightcurve_training_dataset = shuffled_training_dataset.map(self.injection_function,
                                                                                              num_parallel_calls=16)
        batched_training_dataset = lightcurve_training_dataset.batch(100)
        prefetch_training_dataset = batched_training_dataset.prefetch(10)
