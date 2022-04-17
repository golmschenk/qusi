from pathlib import Path
from typing import Union
import uvloop
from bokeh import plotting
from bokeh.application import Application
from bokeh.application.handlers import DirectoryHandler
from bokeh.server.server import Server

plotting.output_notebook.__doc__ = ''


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
    run_viewer(Path('logs/go/gathered/data/tess_ffi_light_curves/tesslcs_sector_6_104/tesslcs_tmag_12_13/tesslc_71970184.pkl'))
