"""Dict-like job-status store for tracking async analysis progress.

Replaces a plain in-process `dict` (the original `processing_status = {}`
in app.py). That dict is invisible to any other gunicorn worker process or
pod replica, so a status poll routed to a different worker than the one
running the job would 404. It also never evicted finished jobs, so it grew
for the lifetime of the process.

This store keeps the exact same call-site API (`store[key] = {...}`,
`store[key].update({...})`, `store[key]['x'] = y`, `key in store`,
`store.get(key)`) so no caller needs to change, but:
  - backs onto Redis (shared across every worker/pod) when REDIS_URL is set
  - falls back to an in-process dict for local/dev use without Redis
  - every write carries a TTL, so old job entries self-evict instead of
    accumulating forever
  - optionally calls an `on_terminal(key, data)` hook the first time a job
    reaches a terminal status ('complete' or 'error'), so callers can persist
    durable history elsewhere (see storage.record_job_event)
"""

import json
import os
import threading
import time

TERMINAL_STATUSES = {'complete', 'error'}


class _StatusProxy(dict):
    """A snapshot of one job's status that writes through to the store on mutation."""

    def __init__(self, store, key, data):
        super().__init__(data or {})
        self._store = store
        self._key = key

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        self._store._write(self._key, dict(self))

    def __setitem__(self, k, v):
        super().__setitem__(k, v)
        self._store._write(self._key, dict(self))


class JobStatusStore:
    def __init__(self, ttl_seconds=None, on_terminal=None):
        self.ttl_seconds = ttl_seconds or int(os.environ.get('JOB_STATUS_TTL', 3600))
        self.on_terminal = on_terminal
        # key -> expiry timestamp; bounds the dedup set itself instead of
        # growing it for the lifetime of the process (see _maybe_fire_terminal).
        self._fired_terminal = {}
        self._redis = None
        url = os.environ.get('REDIS_URL')
        if url:
            try:
                import redis
                self._redis = redis.from_url(url)
                self._redis.ping()
            except Exception as e:
                print(f"JobStatusStore: REDIS_URL set but Redis unreachable ({e}); "
                      f"falling back to in-process status (NOT shared across workers).")
                self._redis = None
        if self._redis is None:
            self._mem = {}
            self._mem_expiry = {}
            self._lock = threading.Lock()

    def _redis_key(self, key):
        return f'fastmoda:job:{key}'

    def _read(self, key):
        if self._redis is not None:
            raw = self._redis.get(self._redis_key(key))
            return json.loads(raw) if raw else None
        with self._lock:
            if self._mem_expiry.get(key, 0) < time.time():
                self._mem.pop(key, None)
                self._mem_expiry.pop(key, None)
                return None
            return self._mem.get(key)

    def _write(self, key, value):
        if self._redis is not None:
            self._redis.set(self._redis_key(key), json.dumps(value), ex=self.ttl_seconds)
        else:
            with self._lock:
                self._mem[key] = value
                self._mem_expiry[key] = time.time() + self.ttl_seconds
        self._maybe_fire_terminal(key, value)

    def _maybe_fire_terminal(self, key, value):
        if not self.on_terminal or key in self._fired_terminal:
            return
        if isinstance(value, dict) and value.get('status') in TERMINAL_STATUSES:
            now = time.time()
            if len(self._fired_terminal) > 10000:
                self._fired_terminal = {k: exp for k, exp in self._fired_terminal.items() if exp > now}
            self._fired_terminal[key] = now + self.ttl_seconds
            try:
                self.on_terminal(key, value)
            except Exception as e:
                print(f"JobStatusStore: on_terminal hook failed for {key}: {e}")

    def __setitem__(self, key, value):
        self._write(key, dict(value))

    def __getitem__(self, key):
        data = self._read(key)
        if data is None:
            raise KeyError(key)
        return _StatusProxy(self, key, data)

    def __contains__(self, key):
        return self._read(key) is not None

    def get(self, key, default=None):
        data = self._read(key)
        return _StatusProxy(self, key, data) if data is not None else default
