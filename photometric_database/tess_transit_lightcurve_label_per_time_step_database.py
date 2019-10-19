"""
Code for a database of TESS transit lightcurves with a label per time step.
"""
from pathlib import Path
from astropy.table import Table
from astroquery.mast import Observations
from astroquery.exceptions import TimeoutError as AstroQueryTimeoutError

from photometric_database.lightcurve_label_per_time_step_database import LightcurveLabelPerTimeStepDatabase


class TessTransitLightcurveLabelPerTimeStepDatabase(LightcurveLabelPerTimeStepDatabase):
    """
    A class for a database of TESS transit lightcurves with a label per time step.
    """
    def __init__(self):
        super().__init__()
        self.data_directory = Path('data/tess')
        self.data_directory.mkdir(parents=True, exist_ok=True)

    def download_database(self):
        """
        Downloads the full lightcurve transit database. This includes the lightcurve files and the data validation files
        (which contain the planet threshold crossing event information).
        """
        print('Downloading TESS observation list...')
        tess_observations = None
        while tess_observations is None:
            try:
                tess_observations = Observations.query_criteria(obs_collection='TESS')
            except AstroQueryTimeoutError:
                print('Timed out connecting to MAST. They have occasional downtime. Trying again...')
        lightcurve_directory = self.data_directory.joinpath('lightcurves')
        lightcurve_directory.mkdir(parents=True, exist_ok=True)
        data_validation_directory = self.data_directory.joinpath('data_validations')
        data_validation_directory.mkdir(parents=True, exist_ok=True)
        for tess_observation in tess_observations:
            download_manifest = None
            while download_manifest is None:
                try:
                    print(f'Downloading data for TIC {tess_observation["target_name"]}...')
                    observation_data_products = Observations.get_product_list(tess_observation)
                    observation_data_products = observation_data_products.to_pandas()
                    lightcurve_and_data_validation_products = observation_data_products[
                        observation_data_products['productFilename'].str.endswith('lc.fits') |
                        observation_data_products['productFilename'].str.endswith('dvr.xml')
                    ]
                    if lightcurve_and_data_validation_products.shape[0] == 0:
                        break  # The observation does not have LC or DVR science products yet.
                    lightcurve_and_data_validation_products = Table.from_pandas(lightcurve_and_data_validation_products)
                    download_manifest = Observations.download_products(lightcurve_and_data_validation_products,
                                                                       download_dir=str(self.data_directory.absolute()))
                    for file_path_string in download_manifest['Local Path']:
                        if file_path_string.endswith('lc.fits'):
                            type_directory = lightcurve_directory
                        else:
                            type_directory = data_validation_directory
                        file_path = Path(file_path_string)
                        file_path.rename(type_directory.joinpath(file_path.name))
                except AstroQueryTimeoutError:
                    print('Timed out connecting to MAST. They have occasional downtime. Trying again...')


if __name__ == '__main__':
    TessTransitLightcurveLabelPerTimeStepDatabase().download_database()
