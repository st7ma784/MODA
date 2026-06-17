"""Bounds two things that were previously unbounded in app.py:

1. How many analysis jobs can run at once. Every `/analyze*` route spawned a
   raw `threading.Thread` with no limit, so N concurrent uploads meant N
   concurrent full in-memory working sets (signal + transform matrices +
   surrogates). BoundedJobRunner caps that at MAX_CONCURRENT_JOBS, queuing
   the rest as plain Python threads waiting on a semaphore (no new
   infrastructure, no change to any worker function's code).

2. How long uploaded scratch files live on disk. `_save_upload()` writes a
   UUID-prefixed file but nothing reliably deleted it (only 2 of ~20 upload
   sites in app.py do their own cleanup), so the uploads folder grew
   forever. start_upload_janitor() sweeps it on an interval and removes
   anything older than its TTL. This is deliberately time-based rather than
   "delete right after this request's job finishes" because at least one
   flow (/analyze followed by /find_changepoints) re-opens the same
   uploaded file from a *later*, separate request.
"""

import os
import threading
import time
from threading import Thread


class BoundedJobRunner:
    def __init__(self, max_concurrent=None):
        self.max_concurrent = max_concurrent or int(os.environ.get('MAX_CONCURRENT_JOBS', 4))
        self._semaphore = threading.BoundedSemaphore(self.max_concurrent)

    def run(self, target, *args, **kwargs):
        """Start target(*args, **kwargs) in a background thread, bounded to
        max_concurrent simultaneous jobs. Returns immediately like the
        Thread(...).start() pattern it replaces."""

        def _wrapped():
            self._semaphore.acquire()
            try:
                target(*args, **kwargs)
            finally:
                self._semaphore.release()

        t = Thread(target=_wrapped)
        t.daemon = True
        t.start()
        return t


def start_upload_janitor(folder, ttl_seconds=None, interval_seconds=None):
    """Start a daemon thread that periodically deletes files in `folder`
    older than ttl_seconds. Safe to call once at app startup."""
    ttl_seconds = ttl_seconds or int(os.environ.get('UPLOAD_TTL_SECONDS', 3600))
    interval_seconds = interval_seconds or int(os.environ.get('UPLOAD_JANITOR_INTERVAL', 300))

    def _sweep_loop():
        while True:
            try:
                cutoff = time.time() - ttl_seconds
                for name in os.listdir(folder):
                    path = os.path.join(folder, name)
                    try:
                        if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
                            os.remove(path)
                    except OSError:
                        pass  # file removed/replaced concurrently — fine, skip it
            except OSError:
                pass  # folder briefly missing/unreadable — try again next interval
            time.sleep(interval_seconds)

    t = Thread(target=_sweep_loop)
    t.daemon = True
    t.start()
    return t
