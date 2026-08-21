"""Streaming HTTP client/relay between the browser and a FastMODA server.

The Pico has a few hundred KB of RAM and FastMODA's `/status/<task_id>`
responses are Plotly figures that routinely run to megabytes, so nothing here
ever holds a whole response. `relay()` pumps upstream bytes into the browser's
socket a chunk at a time, and `upload_signal()` generates the CSV body from the
ring buffer on the fly instead of materialising it.

Routing everything through the board (rather than letting the browser talk to
FastMODA directly) means the API key stays on the device and FastMODA needs no
CORS headers - the browser only ever sees one origin, the Pico.
"""

try:
    import asyncio
except ImportError:                      # MicroPython < 1.21
    import uasyncio as asyncio

CHUNK = 512

# Every CSV line is written as `%.6f\n` over a value clamped to [0, 9.999999],
# which is always exactly 8 characters plus the newline. That fixed width is
# what lets `upload_signal` state an exact Content-Length without building the
# body first - HTTP requires the length up front, and chunked request bodies
# are not worth the compatibility risk against a WSGI server.
LINE_BYTES = 9
_MAX_LINE_VALUE = 9.999999


class BackendError(OSError):
    """Upstream was unreachable or answered with something unusable."""


def split_url(url):
    """`http://host:port/prefix` -> `('host', port, '/prefix')`.

    Only plain HTTP: the Pico's TLS stack cannot keep up with a megabyte-scale
    streamed response, and this is a lab-network development harness.
    """
    url = (url or '').strip()
    if url.startswith('https://'):
        raise BackendError('https backends are not supported; use http://')
    if url.startswith('http://'):
        url = url[len('http://'):]
    url = url.rstrip('/')
    if not url:
        raise BackendError('backend_url is not set')
    authority, _, path = url.partition('/')
    host, _, port = authority.partition(':')
    if not host:
        raise BackendError('backend_url has no host')
    try:
        port = int(port) if port else 80
    except ValueError:
        raise BackendError('backend_url has a non-numeric port')
    return host, port, ('/' + path if path else '')


def format_sample(volts):
    """One fixed-width CSV line. Clamped so the width is guaranteed."""
    if volts < 0:
        volts = 0.0
    elif volts > _MAX_LINE_VALUE:
        volts = _MAX_LINE_VALUE
    return '%.6f\n' % volts


def multipart(boundary, filename, fields, body_length):
    """Return `(prologue, epilogue, content_length)` around a file part.

    The file's bytes are never passed in - the caller streams them between the
    two returned strings.
    """
    parts = []
    for name, value in fields.items():
        parts.append(
            '--%s\r\nContent-Disposition: form-data; name="%s"\r\n\r\n%s\r\n'
            % (boundary, name, value))
    parts.append(
        '--%s\r\nContent-Disposition: form-data; name="file"; filename="%s"\r\n'
        'Content-Type: text/csv\r\n\r\n' % (boundary, filename))
    prologue = ''.join(parts)
    epilogue = '\r\n--%s--\r\n' % boundary
    return prologue, epilogue, len(prologue) + body_length + len(epilogue)


def request_head(method, path, host, headers):
    lines = ['%s %s HTTP/1.1' % (method, path), 'Host: %s' % host,
             'Connection: close']
    for name, value in headers.items():
        if value:
            lines.append('%s: %s' % (name, value))
    return '\r\n'.join(lines) + '\r\n\r\n'


def backend_from_config(cfg):
    host, port, prefix = split_url(cfg.get('backend_url'))
    return {'host': host, 'port': port, 'prefix': prefix,
            'api_key': cfg.get('backend_api_key') or ''}


async def _open(backend):
    try:
        reader, writer = await asyncio.open_connection(backend['host'],
                                                       backend['port'])
    except OSError as exc:
        raise BackendError('cannot reach %s:%s (%s)'
                           % (backend['host'], backend['port'], exc))
    return reader, writer


async def relay(backend, method, path, client_writer, headers=None, body=None):
    """Forward one request upstream and stream the response to `client_writer`.

    Status line and headers are copied through verbatim apart from
    `Connection`, so the browser sees FastMODA's own status code and content
    type. Returns the upstream status code.
    """
    reader, writer = await _open(backend)
    try:
        await _send(writer, backend, method, path, headers, body)

        status_line = await reader.readline()
        if not status_line:
            raise BackendError('backend closed the connection without replying')
        status = _status_code(status_line)

        client_writer.write(status_line)
        while True:
            line = await reader.readline()
            if not line or line in (b'\r\n', b'\n'):
                break
            if line.lower().startswith(b'connection:'):
                continue
            client_writer.write(line)
        client_writer.write(b'Connection: close\r\n\r\n')

        while True:
            chunk = await reader.read(CHUNK)
            if not chunk:
                break
            client_writer.write(chunk)
            await client_writer.drain()
        return status
    finally:
        await _close(writer)


async def collect(backend, method, path, headers=None, body=None, limit=8192):
    """Same request, but buffer a small response and return `(status, text)`.

    Only for endpoints known to answer briefly - `/health`, and the `task_id`
    handshake from an `/analyze_*` route. `limit` truncates rather than
    trusting the upstream to stay small.
    """
    reader, writer = await _open(backend)
    try:
        await _send(writer, backend, method, path, headers, body)
        status = _status_code(await reader.readline())
        await _skip_headers(reader)
        payload = await reader.read(limit)
        return status, (payload or b'').decode('utf-8', 'replace')
    finally:
        await _close(writer)


async def upload_signal(backend, path, samples, full_scale, fields,
                        filename='pico.csv'):
    """POST the captured buffer as a one-column CSV; return `(status, text)`.

    `samples` is a list of raw ADC counts. They are converted to volts and
    written straight to the socket in `CHUNK`-sized pieces, so a 20-second
    buffer costs a few hundred bytes of heap rather than a few hundred KB.
    """
    from sampler import counts_to_volts

    boundary = '----picomoda%d' % (len(samples) * 7 + 13)
    prologue, epilogue, length = multipart(
        boundary, filename, fields, LINE_BYTES * len(samples))

    reader, writer = await _open(backend)
    try:
        head = request_head('POST', backend['prefix'] + path, backend['host'], {
            'X-API-Key': backend['api_key'],
            'Content-Type': 'multipart/form-data; boundary=%s' % boundary,
            'Content-Length': str(length),
        })
        writer.write(head.encode())
        writer.write(prologue.encode())
        await writer.drain()

        lines_per_chunk = CHUNK // LINE_BYTES
        buf = []
        for count in samples:
            buf.append(format_sample(counts_to_volts(count, full_scale)))
            if len(buf) >= lines_per_chunk:
                writer.write(''.join(buf).encode())
                await writer.drain()
                buf = []
        if buf:
            writer.write(''.join(buf).encode())
        writer.write(epilogue.encode())
        await writer.drain()

        status = _status_code(await reader.readline())
        await _skip_headers(reader)
        payload = await reader.read(4096)
        return status, (payload or b'').decode('utf-8', 'replace')
    finally:
        await _close(writer)


async def _send(writer, backend, method, path, headers, body):
    sent = dict(headers or {})
    sent['X-API-Key'] = backend['api_key']
    if body is not None:
        body = body if isinstance(body, bytes) else body.encode()
        sent['Content-Length'] = str(len(body))
    writer.write(request_head(method, backend['prefix'] + path,
                              backend['host'], sent).encode())
    if body is not None:
        writer.write(body)
    await writer.drain()


async def _skip_headers(reader):
    while True:
        line = await reader.readline()
        if not line or line in (b'\r\n', b'\n'):
            return


def _status_code(status_line):
    try:
        return int(status_line.split()[1])
    except (IndexError, ValueError):
        raise BackendError('malformed status line from backend')


async def _close(writer):
    try:
        writer.close()
        await writer.wait_closed()
    except (OSError, AttributeError):
        pass
