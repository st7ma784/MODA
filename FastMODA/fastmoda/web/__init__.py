"""The FastMODA web service: the Flask app (fastmoda.web.app), its templates
and static files, and LocalServer for running it without Docker.

Importing this package does not import Flask; LocalServer does that when it
starts. Needs the ``server`` extra: ``pip install "fastmoda[server]"``.
"""

from fastmoda.web.server import LocalServer

__all__ = ['LocalServer']
