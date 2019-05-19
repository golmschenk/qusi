"""Code for downloading data cubes."""
import json
import os
import sys
from typing import List
from urllib.parse import urlencode
import numpy as np
from urllib.parse import quote as urlencode
import http.client as httplib
from astroquery.gaia import Gaia


class DataCubeDownloader:
    """A class for downloading data cubes."""

    def __init__(self):
        self.data_directory = 'data'
        os.makedirs(self.data_directory, exist_ok=True)

    @staticmethod
    def mast_query(request):
        """Make a MAST query """
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
        """Retrieves the TESS input catalog IDs based on Gaia source IDs."""
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

    @staticmethod
    def get_all_cepheid_gaia_source_ids():
        """Gets all the Gaia source IDs for all the cepheids in the Gaia DR2."""
        # noinspection SqlResolve,SqlNoDataSourceInspection
        job = Gaia.launch_job_async('select source_id from gaiadr2.vari_cepheid')
        job_results = job.get_results()
        source_id_list = job_results['source_id'].data.tolist()
        return source_id_list


if __name__ == '__main__':
    data_cube_downloader = DataCubeDownloader()
    source_ids_ = data_cube_downloader.get_all_cepheid_gaia_source_ids()
    tic_ids = data_cube_downloader.get_tess_input_catalog_ids_from_gaia_source_ids(source_ids_)
    np.save('tic_cepheid.npy', tic_ids)
