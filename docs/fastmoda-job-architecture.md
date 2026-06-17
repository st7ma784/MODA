# FastMODA job status, concurrency, and upload lifecycle

This implements the three phases from the FastMODA memory/overload
investigation (see [`matlab-memory-optimizations.md`](matlab-memory-optimizations.md)
for the MATLAB-side half of that investigation).

## Problem recap

- `processing_status = {}` (a plain Python dict) was per-process. With
  gunicorn `--workers 2` (`Dockerfile`) and Kubernetes HPA
  (`helm/moda/templates/fastmoda-deployment.yaml`, `autoscaling.enabled`),
  a `/status/<task_id>` poll routed to a different worker/pod than the one
  that created the job would 404. It also never evicted finished jobs.
- Every `/analyze*` route spawned a raw, uncapped `threading.Thread` — no
  limit on how many analyses (each holding a full signal + transform
  matrices + surrogates in memory) could run at once.
- Uploaded files (`_save_upload`, `app.py`) were written with a UUID prefix
  but almost never deleted — disk usage in `uploads/` grew indefinitely.

## What changed

### `FastMODA/fastmoda/job_status.py` — `JobStatusStore`

Drop-in replacement for the old `processing_status = {}` dict. Same
call-site API (`store[key] = {...}`, `store[key].update({...})`,
`store[key]['x'] = y`, `key in store`, `store.get(key)`) — **no route
handler needed to change** to adopt it; only the one declaration in
`app.py` did.

- Backs onto Redis (`redis.from_url(REDIS_URL)`) when `REDIS_URL` is set —
  shared across every gunicorn worker and every pod replica.
- Falls back to an in-process dict with the same TTL behavior when
  `REDIS_URL` is unset (local/dev) or Redis is unreachable (logs a warning
  and degrades rather than crashing).
- Every write carries a TTL (`JOB_STATUS_TTL`, default 3600s) so entries
  self-evict instead of accumulating for the life of the process.
- Calls an `on_terminal(task_id, status_dict)` hook the first time a job's
  status becomes `'complete'` or `'error'` — wired to
  `storage.record_job_event` (Phase 3, below).

### `FastMODA/fastmoda/concurrency.py` — `BoundedJobRunner` and `start_upload_janitor`

- `BoundedJobRunner.run(target, *args, **kwargs)` replaces every
  `Thread(target=..., args=...); thread.start()` call site in `app.py`
  (14 of them, including the shared `_async_route` helper). It runs the
  job in a background thread gated by a `threading.BoundedSemaphore`, so
  at most `MAX_CONCURRENT_JOBS` (default 4) analyses run at once per
  process; the rest queue rather than all running simultaneously.
- `start_upload_janitor(folder)` runs a daemon thread that, every
  `UPLOAD_JANITOR_INTERVAL` seconds (default 300), deletes files in
  `uploads/` older than `UPLOAD_TTL_SECONDS` (default 3600).
  Time-based deletion (rather than "delete right after this request's job
  finishes") was deliberate: `/analyze` stores `filepath` in the job status
  for `/find_changepoints` to reopen later in a *separate* request, so an
  immediate per-request delete would break that flow. The `/recordings`
  upload route already does its own immediate `try/finally: os.remove(...)`
  cleanup for its own temp file and is untouched by the janitor.

### `FastMODA/fastmoda/storage.py` — `jobs` table + `record_job_event`

Phase 3: durable history, separate from the ephemeral, TTL'd job-status
cache above. Add-only — no existing table or function changed. Stores only
`status` / `stage` / `error` / `finished_at`, not the (potentially large)
plot/result payload, since this is for "what ran and how did it end", not
for re-serving results.

### Helm chart / docker-compose

- New `helm/moda/templates/redis-deployment.yaml` + a `redis` Service entry
  in `services.yaml`, gated by `values.yaml`'s `redis.enabled` (default
  `true`). Single replica, no persistent volume — it's a cache, not durable
  storage; losing it just means in-flight jobs' status polls 404 until
  those jobs finish and get re-run.
- `fastmoda-deployment.yaml` now injects `REDIS_URL` pointing at that
  service when `redis.enabled`.
- `values.yaml`'s `fastmoda.env` gained `MAX_CONCURRENT_JOBS`,
  `JOB_STATUS_TTL`, `UPLOAD_TTL_SECONDS` (all overridable per-deployment).
- `docker-compose.emulator.yml` gained a `moda-redis` service for local dev
  parity; FastMODA still runs fine without it (falls back to in-process
  status), so removing the service is safe for quick single-container runs.

## What was deliberately not done

A full distributed task queue (e.g. RQ/Celery backed by the same Redis,
with a separately-scaled worker pool) was considered for Phase 2 and
rejected for this pass: it requires worker functions to run in a separate
process/pod from the web request that uploaded the file, which means
either shared `ReadWriteMany` storage for uploads or passing payloads
through Redis directly — a bigger, riskier change that also couldn't be
exercised here (no Redis/GPU available in this environment to test
against). `BoundedJobRunner` gets the practical benefit (bounded concurrent
memory use) without that operational cost. If concurrent load outgrows
what in-process thread-pools across HPA-scaled replicas can handle, that's
the natural next step — revisit then.

## Configuration reference

| Env var | Default | Effect |
|---|---|---|
| `REDIS_URL` | unset | When set, job status is shared via Redis. Unset = in-process only. |
| `MAX_CONCURRENT_JOBS` | `4` | Max simultaneous background analyses per process. |
| `JOB_STATUS_TTL` | `3600` | Seconds before a job-status entry is evicted. |
| `UPLOAD_TTL_SECONDS` | `3600` | Seconds before an uploaded scratch file is deleted. |
| `UPLOAD_JANITOR_INTERVAL` | `300` | How often the upload-cleanup sweep runs. |
