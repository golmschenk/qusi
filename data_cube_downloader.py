"""Code for downloading data cubes."""
import json
import os
import sys
from typing import List, Dict
from urllib.parse import urlencode
import numpy as np
from urllib.parse import quote as urlencode
import http.client as httplib
from astroquery.gaia import Gaia
from astroquery.mast import Tesscut
from astropy.coordinates import SkyCoord


class DataCubeDownloader:
    """A class for downloading data cubes."""

    def __init__(self):
        self.data_directory = 'data'
        os.makedirs(self.data_directory, exist_ok=True)

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

    @staticmethod
    def get_ra_and_dec_for_gaia_source_id(gaia_source_id: int) -> (float, float):
        """Retrieves the RA and DEC for a Gaia source ID."""
        # noinspection SqlResolve,SqlNoDataSourceInspection
        job = Gaia.launch_job_async(f'select ra, dec from gaiadr2.gaia_source where source_id={gaia_source_id}')
        job_results = job.get_results()
        ra = job_results['ra'].data[0]
        dec = job_results['dec'].data[0]
        return ra, dec

    @staticmethod
    def get_all_cepheid_gaia_source_ids():
        """Gets all the Gaia source IDs for all the cepheids in the Gaia DR2."""
        # noinspection SqlResolve,SqlNoDataSourceInspection
        job = Gaia.launch_job_async('select source_id from gaiadr2.vari_cepheid')
        job_results = job.get_results()
        source_id_list = job_results['source_id'].data.tolist()
        return source_id_list

    @staticmethod
    def get_non_cepheid_gaia_source_ids(number_to_get=10000):
        """Gets Gaia source IDs for any non-cepheids source in the Gaia DR2."""
        # noinspection SqlResolve,SqlNoDataSourceInspection
        non_cepheid_query = f'''
        SELECT TOP {number_to_get} gaiadr2.gaia_source.source_id
        FROM gaiadr2.gaia_source
        WHERE NOT EXISTS (SELECT gaiadr2.vari_cepheid.source_id FROM gaiadr2.vari_cepheid
                          WHERE gaiadr2.gaia_source.source_id = gaiadr2.gaia_source.source_id)
        '''
        job = Gaia.launch_job_async(non_cepheid_query)
        job_results = job.get_results()
        source_id_list = job_results['source_id'].data.tolist()
        return source_id_list

    def get_data_cubes_for_gaia_source_id(self, gaia_source_id: int, cube_side_size: int = 5) -> List[np.ndarray]:
        """Get the available TESS data cubes from FFIs for a Gaia source ID."""
        ra, dec = self.get_ra_and_dec_for_gaia_source_id(gaia_source_id)
        coordinates = SkyCoord(ra, dec, unit="deg")
        cutouts = Tesscut.get_cutouts(coordinates, cube_side_size)
        cubes = []
        for cutout in cutouts:
            # The HDU at index 1 is the flux table.
            cube = np.stack([frame['FLUX'] for frame in cutout[1].data], axis=-1)
            cubes.append(cube)
        return cubes


if __name__ == '__main__':
    data_cube_downloader = DataCubeDownloader()
    source_ids_ = data_cube_downloader.get_all_cepheid_gaia_source_ids()
    cube_count = 0
    for source_id in source_ids_:
        cubes = data_cube_downloader.get_data_cubes_for_gaia_source_id(source_id)
        cube_count += len(cubes)
        print(cube_count)
