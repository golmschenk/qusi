import argparse
import logging
import platform
from pathlib import Path

from bokeh import plotting
from bokeh.application import Application
from bokeh.application.handlers import DirectoryHandler
from bokeh.server.server import Server

plotting.output_notebook.__doc__ = ""
logger = logging.getLogger(__name__)


def run_viewer(light_curve_path: Path, port: int = 5007):
    if platform.system() != "Windows":
        import uvloop

        uvloop.install()
    logger.info(f"Opening viewer on http://localhost:{port}/")
    handler = DirectoryHandler(filename=str(Path(__file__).parent), argv=[light_curve_path])
    application = Application(handler)
    server = Server(application, port=port)
    server.start()
    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("light_curve_path")
    parser.add_argument("--port", nargs="?", const=5007, type=int)
    args = parser.parse_args()
    light_curve_path = Path(args.light_curve_path)
    if not light_curve_path.exists():
        logger.info(f"File {light_curve_path} not found.")
        raise SystemExit(1)
    run_viewer(light_curve_path, port=args.port)


if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    main()
