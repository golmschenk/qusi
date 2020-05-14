from pathlib import Path

from bokeh.application import Application
from bokeh.application.handlers import DirectoryHandler
from bokeh.server.server import Server

print('Opening Bokeh application on http://localhost:5006/')
handler = DirectoryHandler(filename=str(Path(__file__).parent))
application = Application(handler)
server = Server(application, port=5007)
server.start()
# Start the specific application on the server.
server.io_loop.add_callback(server.show, "/")
server.io_loop.start()
