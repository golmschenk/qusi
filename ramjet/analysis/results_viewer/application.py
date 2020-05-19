from pathlib import Path
from typing import Union
from bokeh import plotting
from bokeh.application import Application
from bokeh.application.handlers import DirectoryHandler
from bokeh.server.server import Server

plotting.output_notebook.__doc__ = ''


def run_viewer(results_path: Union[Path, str], port: int = 5007, starting_index: int = 0):
    print(f'Opening viewer on http://localhost:{port}/')
    handler = DirectoryHandler(filename=str(Path(__file__).parent), argv=[results_path, starting_index])
    application = Application(handler)
    server = Server(application, port=port)
    server.start()
    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()


if __name__ == '__main__':
    run_viewer('path/to/results/file.feather')
    # run_viewer('path/to/results/file.feather', starting_index=100)  # Will start at the 100th result.
