# Hosting FastMODA for Devices

A microcontroller can capture a signal but rarely has the memory or the time to run a
wavelet transform on it. FastMODA can do that work elsewhere: run the server on a
laptop, a Raspberry Pi, a workstation or a cloud VM, and have the device send it
samples over HTTP and read back the results.

```
 sensor ──► device ──── HTTP POST samples ────► FastMODA server
            (Pico,  ◄─── task_id ──────────────  (pip install, or Docker)
            ESP32,  ──── GET /status/<task_id> ►
             …)     ◄─── results (JSON) ───────
```

The [Pico sensor harness](../hardware/pico-harness.md) is a complete example of this:
MicroPython firmware that relays captured buffers to a FastMODA server.

## Start a server the device can reach

Install the library with the web service, on the machine that will do the computing:

```bash
pip install "fastmoda[server]"
```

The server listens on `127.0.0.1` by default, which only accepts connections from the
same machine. To accept connections from a device on the network, listen on
`0.0.0.0`:

=== "Command line"

    ```bash
    fastmoda-server --host 0.0.0.0 --port 5000
    ```

=== "Python"

    ```python
    from fastmoda import LocalServer

    server = LocalServer(host="0.0.0.0", port=5000).start()
    ```

    `start()` returns once the server answers, and the server runs until
    `server.stop()` or the end of the Python process. `serve_forever()` instead
    blocks until Ctrl+C, which suits a script.

Find the machine's address (`ipconfig` on Windows, `ip addr` on Linux, `ipconfig
getifaddr en0` on macOS). Then check from another machine on the same network:

```bash
curl http://192.168.1.50:5000/health      # {"status":"ok"}
```

If that doesn't answer, the host's firewall is usually blocking port 5000. Allow
incoming connections on it, or for Python, when the operating system asks.

## Call it from the device

Every analysis is two HTTP requests:

1. `POST /<route>`, a `multipart/form-data` upload with:
    - a `file` part holding the samples, and
    - the route's parameters as form fields, at least `fs`, the sampling rate in Hz.

    The reply is `{"task_id": "..."}`.
2. `GET /status/<task_id>`, repeated until `status` is `complete` (results are in
   `results`) or `error` (the reason is in `error`).

The simplest format for the samples is a CSV with one value per line and a filename
ending in `.csv`; the server reads the format from the extension. `.npy`, `.txt` and
`.mat` also work.

```python
import io, time
import numpy as np
import requests

SERVER = "http://192.168.1.50:5000"

def analyse(route, samples, fs, **params):
    """Upload samples to an /analyze_* route and wait for its results."""
    csv = io.StringIO()
    np.savetxt(csv, samples, fmt="%.6f")              # one value per line
    r = requests.post(f"{SERVER}/{route}",
                      files={"file": ("signal.csv", csv.getvalue(), "text/csv")},
                      data={"fs": fs, **params}, timeout=30)
    if not r.ok:                                      # 400s say what was wrong
        raise RuntimeError(f"{route}: {r.status_code} {r.json().get('error')}")
    task_id = r.json()["task_id"]

    while True:                                       # poll until done
        job = requests.get(f"{SERVER}/status/{task_id}", timeout=30).json()
        if job["status"] == "complete":
            return job["results"]
        if job["status"] == "error":
            raise RuntimeError(job["error"])
        time.sleep(0.5)

fs = 50.0
samples = np.sin(2 * np.pi * 1.2 * np.arange(0, 60, 1 / fs))
features = analyse("analyze_features", samples, fs)
print(dict(zip(features["feature_names"], features["feature_vector"])))
```

### Without an HTTP library

On a microcontroller, write the request over a TCP socket yourself. This is the
whole request for 1,500 samples at 50 Hz (a 1.2 Hz sine between 0.5 and 1.5):

```http
POST /analyze_features HTTP/1.1
Host: 192.168.1.50:5000
Connection: close
Content-Type: multipart/form-data; boundary=XyZ
Content-Length: 13668

--XyZ
Content-Disposition: form-data; name="fs"

50
--XyZ
Content-Disposition: form-data; name="file"; filename="signal.csv"
Content-Type: text/csv

1.000000
1.075113
…
--XyZ--
```

Every line ends in `\r\n` except the CSV values, which can end in a plain `\n`.
HTTP needs `Content-Length` before the body. If every value is printed at a fixed
width (as `%.6f` does for values from 0 to 9.999999, which is 9 bytes with the
newline), the length is known before any of the body is generated, so it can be
streamed straight from the sample buffer. The Pico firmware's
[`proxy.py`](https://github.com/st7ma784/MODA/blob/main/pico/firmware/proxy.py)
does exactly this.

### Choose results that fit the device

Most `/analyze_*` routes return Plotly figures for the browser interface, which are
large. For one minute of data at 50 Hz:

| Route | Returns | Size of the final `/status` reply |
|---|---|---|
| `/analyze_features` | 33 numbers: spectral and phase features | about 1.6 KB |
| `/analyze_cwt` | the wavelet transform as a figure | about 1.1 MB |

A device that only needs to act on the result should ask for `/analyze_features`, or
call [`/classify`](rest-api-reference.md#post-classify-and-stored-features), which
answers in the same request with no polling. If the device does need a figure route,
read the reply in chunks rather than all at once, as `proxy.py` does. The
[REST API Reference](rest-api-reference.md) lists every route and its parameters.

Results are kept for an hour (`JOB_STATUS_TTL`), so a device that sleeps between
upload and poll can collect them later.

## Secure it

FastMODA serves plain HTTP, and only some routes need a key:

| Routes | Protected by `FASTMODA_API_KEY` |
|---|---|
| `/recordings`, `/baseline`, `/classify`, `/labels`, `/labelling` | yes, when the variable is set |
| `/analyze_*`, `/status`, `/filter_butter`, the web pages | no |

Set the key on the server, and send it from the device in an `X-API-Key` header:

```bash
FASTMODA_API_KEY=change-me fastmoda-server --host 0.0.0.0
```

Anyone who can reach the server can run analyses on it. Keep it on a network you
control: the local network, a VPN such as Tailscale or WireGuard, or behind a
reverse proxy (nginx, Caddy) that adds HTTPS and authentication. Don't open it
directly to the internet.

## Run it as a service

`fastmoda-server` and `LocalServer` use Werkzeug's threaded server, which is fine for a
handful of devices. For more, or for a server that should start on boot, use a
production server. On Linux or macOS:

```bash
pip install "fastmoda[server]"
gunicorn --bind 0.0.0.0:5000 --workers 1 --threads 8 --timeout 300 fastmoda.web.app:app
```

Keep `--workers 1` unless `REDIS_URL` points at a Redis server. Without Redis, each
worker process keeps its own job list, so a `/status` poll that reaches a different
worker from the upload gets a 404. Threads share one list. gunicorn does not run on
Windows; there, use `fastmoda-server` or the Docker images.

The repository's [`docker-compose.yml`](https://github.com/st7ma784/MODA/blob/main/docker-compose.yml)
runs the same app with Redis, and [`helm/`](https://github.com/st7ma784/MODA/tree/main/helm)
deploys it on Kubernetes.

### Settings

All are environment variables, set before the server starts.

| Variable | Default | |
|---|---|---|
| `FASTMODA_API_KEY` | unset (no key needed) | key for the routes listed under [Secure it](#secure-it) |
| `MODA_DATA_DIR` | `~/.fastmoda` (`LocalServer`, `fastmoda-server`); next to the installed package (gunicorn), so set it there | SQLite database, recordings, trained models |
| `FASTMODA_UPLOAD_DIR` | a temporary directory (`LocalServer`), `uploads/` (gunicorn) | uploaded files while they are analysed |
| `UPLOAD_TTL_SECONDS` | `3600` | how long uploads are kept |
| `JOB_STATUS_TTL` | `3600` | how long results stay available at `/status` |
| `MAX_CONCURRENT_JOBS` | `4` | analyses run at once per process; more are queued |
| `REDIS_URL` | unset | share job status between worker processes |
| `USE_GPU` | `auto` | `true`, `false` or `auto` (PyTorch with CUDA if installed) |

Uploads are limited to 50 MB.
