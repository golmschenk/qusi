import random
from pathlib import Path
from typing import Union

import pandas as pd
import uvloop
from bokeh import plotting
from bokeh.application import Application
from bokeh.application.handlers import DirectoryHandler
from bokeh.server.server import Server

plotting.output_notebook.__doc__ = ''

from ramjet.photometric_database.tess_ffi_light_curve import TessFfiLightCurve


def run_viewer(light_curve_path: Path, port: int = 5007):
    uvloop.install()
    print(f'Opening viewer on http://localhost:{port}/')
    handler = DirectoryHandler(filename=str(Path(__file__).parent), argv=[light_curve_path])
    application = Application(handler)
    server = Server(application, port=port)
    server.start()
    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()


if __name__ == '__main__':
    data_frame = pd.read_csv('/Users/golmschenk/Documents/short_period_variable_neural_network_paper/data/'
                             'filtered_infer_results_2022-09-07-15-21-50.csv')
    # data_frame = data_frame[data_frame['period'] > 0.0218333]
    data_frame = data_frame[data_frame['period'] < 0.208333]
    data_frame = data_frame[data_frame['period'] > 0.041667]
    def local_path_from_row(row: pd.Series) -> Path:
        tic_id, sector = TessFfiLightCurve.get_tic_id_and_sector_from_file_path(row['light_curve_path'])
        local_path = Path('/Users/golmschenk/Documents/short_period_variable_neural_network_paper/data/gathered').joinpath(
            f'tic_id_{tic_id}_sector_{sector}_ffi_light_curve.pkl')
        return local_path

    data_frame['local_light_curve_path'] = data_frame.apply(local_path_from_row, axis=1)
    index = random.randrange(data_frame.shape[0])
    light_curve_path = data_frame['local_light_curve_path'].iloc[index]
    print(data_frame.shape[0])
    print(light_curve_path)
    run_viewer(light_curve_path)
