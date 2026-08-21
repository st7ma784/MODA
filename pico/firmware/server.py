"""The Pico's own HTTP server: static UI, live signal stream, backend relay.

Deliberately minimal - one asyncio task per connection, `Connection: close`
throughout, no keep-alive bookkeeping. The board only ever serves one or two
browsers, and the RAM saved by not tracking connection state is RAM the ring
buffer and the relay can use.
"""

try:
    import asyncio
except ImportError:
    import uasyncio as asyncio

try:
    import ujson as json
except ImportError:
    import json

import config
import proxy

WWW_ROOT = 'www'
STREAM_INTERVAL = 0.1      # seconds between SSE frames
STREAM_MAX_SAMPLES = 400   # per frame; older samples are dropped, not queued

_CONTENT_TYPES = {
    '.html': 'text/html; charset=utf-8',
    '.js': 'application/javascript; charset=utf-8',
    '.css': 'text/css; charset=utf-8',
    '.svg': 'image/svg+xml',
}

# FastMODA routes the UI may ask the board to POST the captured buffer to.
# An allowlist, not a pass-through: the browser must not be able to aim the
# device's credentialed HTTP client at arbitrary paths on the network.
ANALYSIS_ROUTES = (
    'analyze', 'analyze_cwt', 'analyze_wft', 'analyze_stft', 'analyze_modwt',
    'analyze_hilbert', 'analyze_ridge', 'analyze_changepoints',
    'analyze_bispectrum', 'analyze_biphase', 'analyze_bayesian',
    'analyze_surrogates', 'analyze_features', 'filter_butter',
)


class Harness:
    """Holds everything a request handler needs: config, sampler, backend."""

    def __init__(self, cfg, source, net_mode, ip):
        self.cfg = cfg
        self.source = source
        self.net_mode = net_mode
        self.ip = ip
        self.last_error = ''

    @property
    def ring(self):
        return self.source.ring

    def backend(self):
        return proxy.backend_from_config(self.cfg)

    def status(self):
        ring = self.ring
        return {
            'device_id': self.cfg['device_id'],
            'net_mode': self.net_mode,
            'ip': self.ip,
            'sample_rate': self.cfg['sample_rate'],
            'adc_channel': self.cfg['adc_channel'],
            'adc_gpio': 26 + int(self.cfg['adc_channel']),
            'volts_full_scale': self.cfg['volts_full_scale'],
            'capacity': ring.capacity,
            'buffered': ring.available(),
            'total_samples': ring.total,
            'backend_url': self.cfg['backend_url'],
            'last_error': self.last_error,
        }


# ---------------------------------------------------------------- HTTP plumbing

async def _read_request(reader):
    """Parse one request. Returns `(method, path, query, headers, body)`."""
    line = await reader.readline()
    if not line:
        return None
    try:
        method, target, _ = line.decode().split()
    except ValueError:
        return None
    path, _, query = target.partition('?')

    headers = {}
    while True:
        line = await reader.readline()
        if not line or line in (b'\r\n', b'\n'):
            break
        name, _, value = line.decode().partition(':')
        headers[name.strip().lower()] = value.strip()

    body = b''
    length = int(headers.get('content-length') or 0)
    while len(body) < length:
        chunk = await reader.read(length - len(body))
        if not chunk:
            break
        body += chunk
    return method, path, _parse_query(query), headers, body


def _parse_query(query):
    out = {}
    for pair in query.split('&'):
        if not pair:
            continue
        name, _, value = pair.partition('=')
        out[_unquote(name)] = _unquote(value)
    return out


def _unquote(text):
    text = text.replace('+', ' ')
    if '%' not in text:
        return text
    parts = text.split('%')
    out = [parts[0]]
    for part in parts[1:]:
        try:
            out.append(chr(int(part[:2], 16)) + part[2:])
        except ValueError:
            out.append('%' + part)
    return ''.join(out)


async def _send(writer, status, content_type, body, extra=''):
    payload = body if isinstance(body, bytes) else body.encode()
    writer.write(('HTTP/1.1 %s\r\nContent-Type: %s\r\nContent-Length: %d\r\n'
                  'Cache-Control: no-store\r\nConnection: close\r\n%s\r\n'
                  % (status, content_type, len(payload), extra)).encode())
    writer.write(payload)
    await writer.drain()


async def _send_json(writer, payload, status='200 OK'):
    await _send(writer, status, 'application/json', json.dumps(payload))


async def _send_file(writer, name):
    """Stream a file out of flash without reading it whole."""
    path = WWW_ROOT + '/' + name
    try:
        size = _file_size(path)
    except OSError:
        await _send_json(writer, {'error': 'not found'}, '404 Not Found')
        return
    ext = name[name.rfind('.'):] if '.' in name else ''
    writer.write(('HTTP/1.1 200 OK\r\nContent-Type: %s\r\nContent-Length: %d\r\n'
                  'Connection: close\r\n\r\n'
                  % (_CONTENT_TYPES.get(ext, 'application/octet-stream'),
                     size)).encode())
    with open(path, 'rb') as handle:
        while True:
            chunk = handle.read(proxy.CHUNK)
            if not chunk:
                break
            writer.write(chunk)
            await writer.drain()


def _file_size(path):
    with open(path, 'rb') as handle:
        handle.seek(0, 2)
        return handle.tell()


# -------------------------------------------------------------------- handlers

def make_handler(harness):

    async def handle(reader, writer):
        try:
            request = await _read_request(reader)
            if request is None:
                return
            method, path, query, _headers, body = request
            await route(method, path, query, body, writer)
        except proxy.BackendError as exc:
            harness.last_error = str(exc)
            try:
                await _send_json(writer, {'error': str(exc)},
                                 '502 Bad Gateway')
            except OSError:
                pass
        except OSError:
            pass                      # browser hung up mid-response
        except Exception as exc:      # noqa: BLE001 - never kill the server
            harness.last_error = repr(exc)
            try:
                await _send_json(writer, {'error': repr(exc)},
                                 '500 Internal Server Error')
            except OSError:
                pass
        finally:
            await proxy._close(writer)

    async def route(method, path, query, body, writer):
        if path == '/' or path == '/index.html':
            return await _send_file(writer, 'index.html')
        if path in ('/app.js', '/style.css'):
            return await _send_file(writer, path[1:])

        if path == '/api/status' and method == 'GET':
            return await _send_json(writer, harness.status())

        if path == '/api/config':
            if method == 'GET':
                return await _send_json(writer, config.public(harness.cfg))
            if method == 'POST':
                return await _save_config(body, writer)

        if path == '/api/wifi/scan' and method == 'GET':
            import netcfg
            return await _send_json(writer, {'networks': netcfg.scan()})

        if path == '/api/signal' and method == 'GET':
            count = _as_int(query.get('n'), 1000)
            ring = harness.ring
            return await _send_json(writer, {
                'sample_rate': harness.cfg['sample_rate'],
                'volts_full_scale': harness.cfg['volts_full_scale'],
                'counts': ring.latest(count),
                'total_samples': ring.total,
            })

        if path == '/api/stream' and method == 'GET':
            return await _stream(writer, _as_int(query.get('mark'), -1))

        if path == '/api/backend/health' and method == 'GET':
            status, text = await proxy.collect(harness.backend(), 'GET',
                                               '/health')
            return await _send_json(writer, {'status': status, 'body': text})

        if path == '/api/analyze' and method == 'POST':
            return await _analyze(body, writer)

        if path.startswith('/api/task/') and method == 'GET':
            task_id = path[len('/api/task/'):]
            if not _is_token(task_id):
                return await _send_json(writer, {'error': 'bad task id'},
                                        '400 Bad Request')
            return await proxy.relay(harness.backend(), 'GET',
                                     '/status/' + task_id, writer)

        await _send_json(writer, {'error': 'not found'}, '404 Not Found')

    async def _save_config(body, writer):
        try:
            changes = json.loads(body or b'{}')
        except ValueError:
            return await _send_json(writer, {'error': 'invalid JSON'},
                                    '400 Bad Request')
        # A blank secret means "leave it alone": the UI never receives the
        # current value, so it cannot echo it back on an unrelated save.
        for key in ('wifi_password', 'ap_password', 'backend_api_key'):
            if changes.get(key) == '':
                changes.pop(key)
        try:
            updated = config.update(harness.cfg, changes)
        except config.ConfigError as exc:
            return await _send_json(writer, {'error': str(exc)},
                                    '400 Bad Request')

        needs_reboot = any(updated[key] != harness.cfg[key]
                           for key in ('wifi_ssid', 'wifi_password', 'hostname',
                                       'ap_ssid', 'ap_password'))
        resampled = any(updated[key] != harness.cfg[key]
                        for key in ('sample_rate', 'adc_channel',
                                    'buffer_seconds'))
        harness.cfg = updated
        config.save(updated)
        if resampled:
            harness.source.reconfigure(int(updated['adc_channel']),
                                       int(updated['sample_rate']),
                                       config.buffer_samples(updated))
        await _send_json(writer, {'saved': True, 'needs_reboot': needs_reboot,
                                  'config': config.public(updated)})

    async def _analyze(body, writer):
        try:
            request = json.loads(body or b'{}')
        except ValueError:
            return await _send_json(writer, {'error': 'invalid JSON'},
                                    '400 Bad Request')
        route_name = str(request.get('route') or 'analyze_cwt').strip('/')
        if route_name not in ANALYSIS_ROUTES:
            return await _send_json(writer, {'error': 'route not allowed: %s'
                                             % route_name}, '400 Bad Request')

        count = _as_int(request.get('samples'), harness.ring.available())
        samples = harness.ring.latest(count)
        if not samples:
            return await _send_json(writer, {'error': 'no samples captured yet'},
                                    '409 Conflict')

        # fs always comes from the device, never the browser: FastMODA reads
        # 1.0 out of a CSV, so every frequency axis downstream depends on this.
        fields = {'fs': str(harness.cfg['sample_rate']),
                  'device_id': harness.cfg['device_id']}
        for name, value in (request.get('params') or {}).items():
            if name not in ('fs', 'file', 'device_id'):
                fields[str(name)] = str(value)

        status, text = await proxy.upload_signal(
            harness.backend(), '/' + route_name, samples,
            float(harness.cfg['volts_full_scale']), fields)
        await _send(writer, '200 OK' if status == 200 else '502 Bad Gateway',
                    'application/json',
                    text or json.dumps({'error': 'empty reply from backend'}))

    async def _stream(writer, mark):
        ring = harness.ring
        if mark < 0:
            mark = ring.total
        writer.write(b'HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n'
                     b'Cache-Control: no-cache\r\nConnection: close\r\n\r\n')
        await writer.drain()
        while True:
            mark, dropped, counts = ring.since(mark, STREAM_MAX_SAMPLES)
            frame = json.dumps({'mark': mark, 'dropped': dropped,
                                'counts': counts})
            writer.write(('data: %s\n\n' % frame).encode())
            await writer.drain()
            await asyncio.sleep(STREAM_INTERVAL)

    return handle


def _as_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _is_token(text):
    """Task ids are UUIDs; reject anything that could escape the URL path."""
    return bool(text) and len(text) <= 64 and all(
        char.isalpha() or char.isdigit() or char in '-_' for char in text)


async def serve(harness, port=80):
    return await asyncio.start_server(make_handler(harness), '0.0.0.0', port)
