# MODA Pico Harness

MicroPython firmware for a Raspberry Pi Pico 2 W that samples an analogue sensor,
serves a web UI over Wi-Fi, and relays captured buffers to a FastMODA server.

Full documentation — wiring diagram, RC filter sizing, flashing, first boot — is in
[`docs/hardware/pico-harness.md`](../docs/hardware/pico-harness.md).

## Layout

```
firmware/
  main.py      boot: config -> Wi-Fi -> sampler -> server
  config.py    persisted JSON settings, validation, secret masking
  sampler.py   Ring buffer + timer-driven ADC capture
  netcfg.py    Wi-Fi join with access-point fallback
  proxy.py     streaming HTTP client/relay to FastMODA
  server.py    the board's own HTTP server and JSON API
  www/         the web UI (no external dependencies)
tools/
  host_sim.py  run the firmware on a laptop with a synthetic sensor
tests/         pytest suite, runs on CPython
```

`config.py`, `sampler.Ring` and `proxy.py`'s encoders are plain Python with no
`machine`/`network` imports, which is what lets the test suite and the simulator run
them on CPython unchanged. Only `sampler.AdcSource` and `netcfg` touch hardware.

## HTTP API

The web UI is a client of this; so can anything else on the network be.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/status` | Device, network and acquisition state |
| `GET` | `/api/config` | Current settings, secrets masked |
| `POST` | `/api/config` | Update settings (JSON body); reports if a reboot is needed |
| `GET` | `/api/wifi/scan` | Visible networks, strongest first |
| `GET` | `/api/signal?n=` | The most recent N samples as raw ADC counts |
| `GET` | `/api/stream?mark=` | SSE stream of new samples |
| `GET` | `/api/backend/health` | Proxied FastMODA `/health` |
| `POST` | `/api/analyze` | Upload the buffer to a FastMODA route; returns its `task_id` |
| `GET` | `/api/task/<id>` | Streams FastMODA's `/status/<id>` straight through |

```bash
curl -X POST http://moda-pico.local/api/analyze \
  -H 'Content-Type: application/json' \
  -d '{"route": "analyze_cwt", "samples": 4000,
       "params": {"freq_min": "0.1", "freq_max": "20", "n_freqs": "50"}}'
```

`route` is checked against an allowlist (`server.ANALYSIS_ROUTES`) — the browser must
not be able to aim the device's credentialed HTTP client at arbitrary paths. `fs` and
`file` are ignored if passed in `params`; the sampling frequency always comes from the
device's own configuration.

## Memory discipline

Two constraints shape most of this code:

- **`Ring.push()` runs in a soft-IRQ timer callback**, where allocation raises
  `MemoryError`. The buffer and every index are preallocated; nothing in the sampling
  path allocates after `__init__`.
- **FastMODA's `/status` responses are Plotly figures running to megabytes**, which the
  board cannot hold. `proxy.relay()` pumps them through in 512-byte chunks, and
  `proxy.upload_signal()` generates the CSV body as it writes it. Each CSV line is a
  fixed 9 bytes (`%.6f\n` over a clamped value), which is what makes an exact
  `Content-Length` possible without building the body first.

## Running the tests

```bash
pip install pytest pytest-asyncio
pytest
```

Or without a local Python:

```bash
docker run --rm -v "$PWD:/w" -w /w python:3.13-slim \
  sh -c "pip install -q pytest pytest-asyncio && pytest -q"
```
