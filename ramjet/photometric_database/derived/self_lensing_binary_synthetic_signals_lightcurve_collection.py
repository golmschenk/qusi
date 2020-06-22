"""
Code for a lightcurve collection of Agnieszka Cieplak's synthetic signals.
"""
import tarfile
import urllib.request
from pathlib import Path

from ramjet.photometric_database.lightcurve_collection import LightcurveCollection


class SelfLensingBinarySyntheticSignalsLightcurveCollection(LightcurveCollection):
    """
    A lightcurve collection for Agnieszka Cieplak's synthetic signals.
    """
    def __init__(self):
        super().__init__()
        self.data_directory: Path = Path('data/self_lensing_binary_synthetic_signals')

    def download_csv_files(self):
        """
        Downloads Agnieszka Cieplak's synthetic signals in their original CSV form.
        """
        print('Downloading synthetic signal CSV files...')
        tar_file_path = self.data_directory.joinpath('synthetic_signals_csv_files.tar')
        urllib.request.urlretrieve('https://api.onedrive.com/v1.0/shares/s!AjiSFm1N8Bv7ghXushB7JOzABXdv/root/content',
                                   str(tar_file_path))
        with tarfile.open(tar_file_path) as csv_tar_file:
            csv_tar_file.extractall(self.data_directory)
        tar_file_path.unlink()
        csv_uncompressed_directory = self.data_directory.joinpath('LearningSetedgeon_all_sum')
        for path in csv_uncompressed_directory.glob('*'):
            path.rename(self.data_directory.joinpath(path.name))
        csv_uncompressed_directory.rmdir()



if __name__ == '__main__':
    lightcurve_collection = SelfLensingBinarySyntheticSignalsLightcurveCollection()
    lightcurve_collection.data_directory.mkdir(parents=True, exist_ok=True)
    lightcurve_collection.download_csv_files()
