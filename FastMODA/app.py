"""The FastMODA web service now lives in the package as fastmoda.web.app, so
that it installs with pip and can be run by fastmoda.LocalServer.

This module stays so that `gunicorn app:app`, `python app.py` and
`import app` (the Docker images and the parity tests) keep working: importing
it gives the fastmoda.web.app module itself.
"""

import sys

if __name__ == '__main__':
    import runpy
    runpy.run_module('fastmoda.web.app', run_name='__main__')
else:
    from fastmoda.web import app as _app
    sys.modules[__name__] = _app
