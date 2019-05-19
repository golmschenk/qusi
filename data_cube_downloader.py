"""Code for downloading data cubes."""
import os
from typing import List

import pandas as pd
import numpy as np
import gzip
import urllib.request
import tarfile
import shutil


class DataCubeDownloader:
    """A class for downloading data cubes."""

    def __init__(self):
        self.data_directory = 'data'
        os.makedirs(self.data_directory, exist_ok=True)

    @staticmethod
    def download_tess_input_catalog():
        """Downloads the full TESS input catalog."""
        base_url = 'https://archive.stsci.edu/missions/tess/catalogs/tic_v7/'
        filename = 'tic_v7_full.tar.gz'
        out_file_path = filename[:-3]
        response = urllib.request.urlopen(base_url + filename)
        with open(out_file_path, 'wb') as outfile:
            outfile.write(gzip.decompress(response.read()))
        tar_file = tarfile.open('tic_v7_full.tar')
        tar_file.extractall()

    def get_tess_input_catalog_ids_from_gaia_source_ids(self, gaia_source_id_list: List[int]) -> List[int]:
        """Retrieves the TESS input catalog IDs based on Gaia source IDs."""
        # Column 0 is the TIC ID and column 8 is the Gaia ID.
        tess_input_catalog_path = os.path.join(self.data_directory, 'tess_input_catalog.csv')
        tess_input_catalog_iter = pd.read_csv(tess_input_catalog_path, usecols=[0, 8], names=['ID', 'GAIA'],
                                              iterator=True, chunksize=1000, header=None)
        matched_data_frame = pd.concat([chunk[chunk['GAIA'].isin(gaia_source_id_list)]
                                        for chunk in tess_input_catalog_iter])
        id_list = matched_data_frame['ID'].unique()
        return id_list

    def get_all_cepheid_gaia_source_ids(self):
        """Gets all the Gaia source IDs for all the cepheids in the Gaia DR2."""
        gaia_cepheid_path = os.path.join(self.data_directory, 'gaia_cepheids.csv')
        gaia_cepheid_data_frame = pd.read_csv(gaia_cepheid_path, usecols=['source_id'])
        source_id_list = gaia_cepheid_data_frame['source_id'].unique()
        return source_id_list


if __name__ == '__main__':
    data_cube_downloader = DataCubeDownloader()
    source_ids = data_cube_downloader.get_all_cepheid_gaia_source_ids()
    tic_ids = data_cube_downloader.get_tess_input_catalog_ids_from_gaia_source_ids(source_ids)
    np.save('tic_ceph.npy', tic_ids)