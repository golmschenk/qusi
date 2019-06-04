"""Code for downloading data cubes."""
import json
import math
import os
import random
import sys
from typing import List, Dict
from urllib.parse import urlencode
import numpy as np
from urllib.parse import quote as urlencode
import http.client as httplib
from astroquery.gaia import Gaia
from astroquery.mast import Tesscut
from astropy.coordinates import SkyCoord
from requests.exceptions import HTTPError


class DataCubeDownloader:
    """A class for downloading data cubes."""

    def __init__(self):
        self.data_directory = 'data'
        os.makedirs(self.data_directory, exist_ok=True)
        self.magnitude_filter_value = 16

    @staticmethod
    def mast_query(request: Dict):
        """Make a MAST query."""
        server = 'mast.stsci.edu'
        python_version = '.'.join(map(str, sys.version_info[:3]))
        http_headers = {'Content-type': 'application/x-www-form-urlencoded',
                        'Accept': 'text/plain',
                        'User-agent': 'python-requests/' + python_version}
        request_string = json.dumps(request)
        request_string = urlencode(request_string)
        https_connection = httplib.HTTPSConnection(server)
        https_connection.request('POST', '/api/v0/invoke', 'request=' + request_string, http_headers)
        response = https_connection.getresponse()
        head = response.getheaders()
        content = response.read().decode('utf-8')
        https_connection.close()
        return head, content

    @staticmethod
    def launch_gaia_job(query_string):
        """Query on Gaia repeating if there is a time out error."""
        job = None
        while True:
            try:
                job = Gaia.launch_job_async(query_string)
                break
            except TimeoutError:
                print('Timed out, trying again...')
                continue
        return job

    @staticmethod
    def get_tess_cuts(coordinates, cube_side_size):
        """Gets the TESS cuts for a given set of coordinates. Retries on HTTP error."""
        cutouts = None
        while True:
            try:
                cutouts = Tesscut.get_cutouts(coordinates, cube_side_size)
                break
            except HTTPError:
                continue
        return cutouts

    def get_tess_input_catalog_ids_from_gaia_source_ids(self, gaia_source_id_list: List[int]) -> List[int]:
        """
        Retrieves the TESS input catalog IDs based on Gaia source IDs.
        Note, only the brightest Gaia source corresponding to a TIC star will return a TIC ID.
        """
        gaia_source_ids = list(map(str, gaia_source_id_list))
        request = {'service': 'Mast.Catalogs.Filtered.Tic',
                   'format': 'json',
                   'params': {
                       'columns': 'ID, GAIA',
                       'filters': [
                           {'paramName': 'GAIA',
                            'values': gaia_source_ids}
                       ]
                   }}
        headers, response_string = self.mast_query(request)
        response_json = json.loads(response_string)
        tess_input_catalog_id_list = [entry['ID'] for entry in response_json['data']]
        return tess_input_catalog_id_list

    def get_ra_and_dec_for_tess_input_catalog_id(self, tess_input_catalog_id: int) -> (float, float):
        """Retrieves the RA and DEC for a TESS input catalog ID."""
        tess_input_catalog_id = str(tess_input_catalog_id)
        request = {'service': 'Mast.Catalogs.Filtered.Tic',
                   'format': 'json',
                   'params': {
                       'columns': 'ra, dec',
                       'filters': [
                           {'paramName': 'ID',
                            'values': [tess_input_catalog_id]}
                       ]
                   }}
        headers, response_string = self.mast_query(request)
        response_json = json.loads(response_string)
        ra = response_json['data'][0]['ra']
        dec = response_json['data'][0]['dec']
        return ra, dec

    def get_ra_and_dec_for_gaia_source_id(self, gaia_source_id: int) -> (float, float):
        """Retrieves the RA and DEC for a Gaia source ID."""
        # noinspection SqlResolve,SqlNoDataSourceInspection
        job = self.launch_gaia_job(f'select ra, dec from gaiadr2.gaia_source where source_id={gaia_source_id}')
        job_results = job.get_results()
        ra = job_results['ra'].data[0]
        dec = job_results['dec'].data[0]
        return ra, dec

    def get_all_cepheid_gaia_source_ids(self):
        """Gets all the Gaia source IDs for all the cepheids in the Gaia DR2."""
        # noinspection SqlResolve,SqlNoDataSourceInspection
        cepheid_query = f'''
        SELECT t1.source_id
        FROM gaiadr2.gaia_source t1 LEFT JOIN gaiadr2.vari_cepheid t2 ON t1.source_id = t2.source_id
        WHERE t2.source_id IS NOT NULL AND (t1.phot_bp_mean_mag < {self.magnitude_filter_value}
                                            OR t1.phot_g_mean_flux < {self.magnitude_filter_value})
        '''
        job = self.launch_gaia_job(cepheid_query)
        job_results = job.get_results()
        source_id_list = job_results['source_id'].data.tolist()
        return source_id_list

    def get_non_cepheid_gaia_source_ids(self):
        """Gets Gaia source IDs for any non-cepheids source in the Gaia DR2."""
        # noinspection SqlResolve,SqlNoDataSourceInspection
        non_cepheid_query = f'''
        SELECT t1.source_id
        FROM gaiadr2.gaia_source t1 LEFT JOIN gaiadr2.vari_cepheid t2 ON t1.source_id = t2.source_id
        WHERE t2.source_id IS NULL AND (t1.phot_bp_mean_mag < {self.magnitude_filter_value}
                                        OR t1.phot_g_mean_flux < {self.magnitude_filter_value})
        '''
        job = self.launch_gaia_job(non_cepheid_query)
        job_results = job.get_results()
        source_id_list = job_results['source_id'].data.tolist()
        return source_id_list

    def get_all_classic_cepheid_gaia_source_ids(self):
        """Gets all the Gaia source IDs for all the classic cepheids in the Gaia DR2."""
        # noinspection SqlResolve,SqlNoDataSourceInspection
        job = self.launch_gaia_job("select source_id from gaiadr2.vari_cepheid where type_best_classification='DCEP'")
        job_results = job.get_results()
        source_id_list = job_results['source_id'].data.tolist()
        return source_id_list

    def get_all_type_2_cepheid_gaia_source_ids(self):
        """Gets all the Gaia source IDs for all the type 2 cepheids in the Gaia DR2."""
        # noinspection SqlResolve,SqlNoDataSourceInspection
        job = self.launch_gaia_job("select source_id from gaiadr2.vari_cepheid where type_best_classification='T2CEP'")
        job_results = job.get_results()
        source_id_list = job_results['source_id'].data.tolist()
        return source_id_list

    def get_data_cubes_for_gaia_source_id(self, gaia_source_id: int, cube_side_size: int = 10) -> List[np.ndarray]:
        """Get the available TESS data cubes from FFIs for a Gaia source ID."""
        ra, dec = self.get_ra_and_dec_for_gaia_source_id(gaia_source_id)
        coordinates = SkyCoord(ra, dec, unit="deg")
        cutouts = self.get_tess_cuts(coordinates, cube_side_size)
        cubes = []
        for cutout in cutouts:
            # The HDU at index 1 is the flux table.
            cube = np.stack([frame['FLUX'] for frame in cutout[1].data], axis=-1)
            cubes.append(cube)
        return cubes

    def download_classic_cepheid_and_type_2_cepheid_database(self):
        """Downloads a positive/negative cepheid database."""
        type_2_cepheid_source_ids = self.get_all_type_2_cepheid_gaia_source_ids()
        self.download_cubes_for_gaia_source_id_list('type-2-cepheids', type_2_cepheid_source_ids)
        classic_cepheid_source_ids = self.get_all_classic_cepheid_gaia_source_ids()
        self.download_cubes_for_gaia_source_id_list('classic-cepheids', classic_cepheid_source_ids)

    def download_positive_negative_cepheid_database(self, maximum_positive_examples: int = 10000,
                                                    maximum_negative_examples: int = 40000):
        """Downloads a positive/negative cepheid database."""
        cepheid_source_ids = self.get_all_cepheid_gaia_source_ids()
        self.download_cubes_for_gaia_source_id_list('positive', cepheid_source_ids,
                                                    maximum_cubes=maximum_positive_examples)
        non_cepheid_source_ids = self.get_non_cepheid_gaia_source_ids()
        self.download_cubes_for_gaia_source_id_list('negative', non_cepheid_source_ids,
                                                    maximum_cubes=maximum_negative_examples)

    def download_cubes_for_gaia_source_id_list(self, dataset_name: str, source_ids: List[int],
                                               maximum_cubes: int = math.inf):
        """Downloads a set of cubes from a set of Gaia source IDs."""
        if maximum_cubes != math.inf:
            random.seed(0)
            random.shuffle(source_ids)  # Randomize which sources to download if not downloading them all.
        dataset_directory = os.path.join(self.data_directory, dataset_name)
        os.makedirs(dataset_directory, exist_ok=True)
        cube_count = 0
        for source_id in source_ids:
            cubes = self.get_data_cubes_for_gaia_source_id(source_id)
            for index, cube in enumerate(cubes):
                np.save(os.path.join(dataset_directory, f'{source_id}_{index}.npy'), cube)
                cube_count += 1
                if cube_count >= maximum_cubes:
                    break
            if cube_count >= maximum_cubes:
                break


if __name__ == '__main__':
    data_cube_downloader = DataCubeDownloader()
    data_cube_downloader.download_positive_negative_cepheid_database()
