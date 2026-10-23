"""Run the FastMODA web service in-process, without Docker.

From a notebook or a script::

    from fastmoda import LocalServer

    server = LocalServer().start()   # serves in a background thread
    server.url                       # 'http://127.0.0.1:5000'
    server.stop()

    with LocalServer(port=0) as server:   # port=0 picks any free port
        requests.post(server.url + '/api/...', ...)

From the command line: ``fastmoda-server --port 5000 --open``.

Needs the ``server`` extra: ``pip install "fastmoda[server]"``.

This uses Werkzeug's threaded development server, which is fine for one
person on their own machine. For a shared deployment use gunicorn
(``gunicorn fastmoda.web.app:app``) or the Docker images.
"""

import argparse
import os
import tempfile
import threading
import time
import urllib.error
import urllib.request
import webbrowser

# Where recordings, labels and trained models are kept when neither
# data_dir nor MODA_DATA_DIR says otherwise. The library's own default is
# next to the package, which for a pip install is inside site-packages.
DEFAULT_DATA_DIR = os.path.join(os.path.expanduser('~'), '.fastmoda')


def _load_app(data_dir, upload_dir):
    """Import the Flask app with its storage pointed at data_dir/upload_dir.

    fastmoda.storage reads MODA_DATA_DIR and the app reads
    FASTMODA_UPLOAD_DIR when they are first imported, so the environment is
    set before importing them; if either was already imported, its settings
    are updated in place as well.
    """
    os.environ['MODA_DATA_DIR'] = data_dir
    os.environ['FASTMODA_UPLOAD_DIR'] = upload_dir

    from fastmoda import storage
    storage.DATA_DIR = data_dir
    storage.DB_PATH = os.path.join(data_dir, 'moda.db')
    storage.RECORDINGS_DIR = os.path.join(data_dir, 'recordings')
    storage.MODELS_DIR = os.path.join(data_dir, 'models')

    try:
        from fastmoda.web import app as module
    except ImportError as e:
        raise ImportError(
            f'The FastMODA web service needs the server extra ({e}). '
            'Install it with: pip install "fastmoda[server]"') from e

    storage.init_db()
    os.makedirs(upload_dir, exist_ok=True)
    module.app.config['UPLOAD_FOLDER'] = upload_dir
    return module.app


class LocalServer:
    """The FastMODA web service (the same app the Docker images run).

    Args:
        host: interface to listen on. The default only accepts connections
            from this machine; use '0.0.0.0' to serve the local network.
        port: port to listen on; 0 picks a free one (see ``url``).
        data_dir: where the SQLite database, recordings and models live.
            Defaults to MODA_DATA_DIR if set, else ~/.fastmoda.
        upload_dir: scratch space for uploaded files. Defaults to
            FASTMODA_UPLOAD_DIR if set, else a new temporary directory.
        log_requests: print a line per HTTP request, as the Docker images do.
            Off by default, so requests don't fill notebook cell output.

    The app is a module-level singleton, so every LocalServer in one Python
    process serves the same app and the most recent data_dir/upload_dir.
    """

    def __init__(self, host='127.0.0.1', port=5000, data_dir=None, upload_dir=None,
                 log_requests=False):
        self.host = host
        self.port = port
        self.data_dir = data_dir or os.environ.get('MODA_DATA_DIR') or DEFAULT_DATA_DIR
        self.upload_dir = (upload_dir or os.environ.get('FASTMODA_UPLOAD_DIR')
                           or tempfile.mkdtemp(prefix='fastmoda-uploads-'))
        self.log_requests = log_requests
        self._server = None
        self._thread = None

    @property
    def url(self):
        """Base URL of the running server, e.g. 'http://127.0.0.1:5000'."""
        host = '127.0.0.1' if self.host in ('0.0.0.0', '') else self.host
        if ':' in host:
            host = f'[{host}]'   # IPv6
        return f'http://{host}:{self.port}'

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    def _bind(self):
        from werkzeug.serving import WSGIRequestHandler, make_server
        app = _load_app(self.data_dir, self.upload_dir)
        handler = WSGIRequestHandler
        if not self.log_requests:
            class handler(WSGIRequestHandler):
                def log_request(self, *args, **kwargs):
                    pass   # errors are still logged
        try:
            self._server = make_server(self.host, self.port, app, threaded=True,
                                       request_handler=handler)
        except (OSError, SystemExit) as e:
            # Werkzeug prints a message and calls sys.exit(1) when the port is
            # taken, which would end a notebook kernel; raise an error instead.
            raise OSError(
                f'Could not listen on {self.host}:{self.port}. '
                'Is another server using that port? Pass port=0 to pick a free one.') from e
        self.port = self._server.server_port

    def start(self, timeout=30):
        """Serve in a background thread and wait until /health answers.

        Returns self, so ``server = LocalServer().start()`` works.
        """
        if self.running:
            return self
        if self._server is None:
            self._bind()
        self._thread = threading.Thread(
            target=self._server.serve_forever, name='fastmoda-server', daemon=True)
        self._thread.start()
        self.wait_until_ready(timeout)
        return self

    def wait_until_ready(self, timeout=30):
        """Block until GET /health succeeds; raise TimeoutError otherwise."""
        deadline = time.monotonic() + timeout
        while True:
            try:
                with urllib.request.urlopen(self.url + '/health', timeout=5):
                    return
            except (urllib.error.URLError, ConnectionError, OSError):
                if time.monotonic() > deadline:
                    raise TimeoutError(f'FastMODA did not answer at {self.url}/health '
                                       f'within {timeout} s')
                time.sleep(0.1)

    def stop(self):
        """Stop serving and release the port."""
        if self._server is None:
            return
        self._server.shutdown()
        if self._thread is not None:
            self._thread.join()
        self._server.server_close()
        self._server = None
        self._thread = None

    def serve_forever(self):
        """Serve in this thread until Ctrl+C (for scripts and the CLI)."""
        if self._server is None:
            self._bind()
        try:
            self._server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            self._server.server_close()
            self._server = None

    def open_browser(self, path='/'):
        """Open the web interface in the default browser."""
        webbrowser.open(self.url + path)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    def __repr__(self):
        state = 'running' if self.running else 'stopped'
        return f'<LocalServer {self.url} ({state})>'

    def _repr_html_(self):
        # A clickable link when the server is the last value in a notebook cell.
        state = 'running' if self.running else 'stopped'
        return f'FastMODA <a href="{self.url}" target="_blank">{self.url}</a> ({state})'


def main(argv=None):
    """Entry point for the ``fastmoda-server`` command."""
    parser = argparse.ArgumentParser(
        prog='fastmoda-server', description='Run the FastMODA web service locally.')
    parser.add_argument('--host', default='127.0.0.1',
                        help="interface to listen on (default 127.0.0.1; '0.0.0.0' for the network)")
    parser.add_argument('--port', type=int, default=5000, help='port (default 5000; 0 = any free port)')
    parser.add_argument('--data-dir', help=f'database, recordings and models (default {DEFAULT_DATA_DIR})')
    parser.add_argument('--upload-dir', help='scratch space for uploads (default: a temporary directory)')
    parser.add_argument('--open', action='store_true', help='open the web interface in a browser')
    parser.add_argument('--quiet', action='store_true', help='don\'t print a line per request')
    args = parser.parse_args(argv)

    server = LocalServer(args.host, args.port, args.data_dir, args.upload_dir,
                         log_requests=not args.quiet)
    try:
        server._bind()   # first, so a port of 0 is resolved before printing the URL
    except OSError as e:
        parser.exit(1, f'fastmoda-server: {e}\n')
    print(f'FastMODA data in {server.data_dir}')
    print(f'Serving on {server.url} (Ctrl+C to stop)')
    if args.open:
        server.open_browser()
    server.serve_forever()


if __name__ == '__main__':
    main()
