"""End-to-end tests: browser -> Pico server -> (fake) FastMODA.

The fake backend is a real socket server speaking real HTTP, so the streaming
relay and the hand-rolled multipart upload are exercised as they would be on
the board rather than mocked out.
"""

import asyncio
import json
import os

import pytest

import config
import sampler
import server

FIRMWARE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'firmware'))

# Far bigger than proxy.CHUNK, so `relay` has to pump many chunks - that is the
# case the board can only survive by streaming rather than buffering.
BIG_PLOT = json.dumps({'data': [{'type': 'heatmap',
                                 'z': [[0.125] * 200 for _ in range(40)]}]})


class FakeBackend:
    """Minimal stand-in for FastMODA's `/health`, `/analyze_*` and `/status`."""

    def __init__(self):
        self.requests = []          # (method, path, headers, body)
        self.port = None
        self._server = None

    async def start(self):
        self._server = await asyncio.start_server(self._handle, '127.0.0.1', 0)
        self.port = self._server.sockets[0].getsockname()[1]

    async def stop(self):
        self._server.close()
        await self._server.wait_closed()

    async def _handle(self, reader, writer):
        request = await server._read_request(reader)
        if request is None:
            writer.close()
            return
        method, path, _query, headers, body = request
        self.requests.append((method, path, headers, body))

        if path.endswith('/health'):
            payload = json.dumps({'status': 'ok'})
        elif path.endswith('/analyze_cwt'):
            payload = json.dumps({'task_id': 'task-1',
                                  'signal_length': body.count(b'\n')})
        elif '/status/' in path:
            payload = json.dumps({'status': 'complete', 'progress': 100,
                                  'results': {'cwt_plot': BIG_PLOT}})
        else:
            payload = json.dumps({'error': 'unexpected path ' + path})

        writer.write(('HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n'
                      'Content-Length: %d\r\nConnection: close\r\n\r\n'
                      % len(payload)).encode())
        writer.write(payload.encode())
        await writer.drain()
        writer.close()


class FakeSource:
    def __init__(self, capacity=512):
        self.ring = sampler.Ring(capacity)
        self.reconfigured = None

    def fill(self, count):
        for i in range(count):
            self.ring.push(i % 4096)

    def reconfigure(self, channel, sample_rate, capacity):
        self.reconfigured = (channel, sample_rate, capacity)
        if capacity != self.ring.capacity:
            self.ring = sampler.Ring(capacity)


async def http(port, method, path, body=None, headers=''):
    reader, writer = await asyncio.open_connection('127.0.0.1', port)
    request = '%s %s HTTP/1.1\r\nHost: pico\r\nConnection: close\r\n%s' % (
        method, path, headers)
    if body is not None:
        request += 'Content-Length: %d\r\n' % len(body)
    writer.write((request + '\r\n').encode())
    if body is not None:
        writer.write(body.encode())
    await writer.drain()
    raw = await reader.read(-1)
    writer.close()
    head, _, payload = raw.partition(b'\r\n\r\n')
    status = int(head.split()[1])
    return status, payload.decode('utf-8', 'replace')


@pytest.fixture
def harness_env(tmp_path, monkeypatch):
    monkeypatch.chdir(FIRMWARE)          # server resolves www/ relatively
    monkeypatch.setattr(config, 'CONFIG_PATH', str(tmp_path / 'config.json'))
    monkeypatch.setattr(config, 'save',
                        lambda cfg, path=str(tmp_path / 'config.json'):
                        cfg)
    return tmp_path


@pytest.fixture
async def running(harness_env):
    backend = FakeBackend()
    await backend.start()

    cfg = config.update(dict(config.DEFAULTS), {
        'backend_url': 'http://127.0.0.1:%d' % backend.port,
        'backend_api_key': 'secret-key',
        'sample_rate': 100,
    })
    source = FakeSource()
    harness = server.Harness(cfg, source, 'sta', '127.0.0.1')
    pico = await server.serve(harness, port=0)
    port = pico.sockets[0].getsockname()[1]
    try:
        yield harness, source, backend, port
    finally:
        pico.close()
        await pico.wait_closed()
        await backend.stop()


@pytest.mark.asyncio
async def test_serves_the_ui_from_flash(running):
    _harness, _source, _backend, port = running
    status, body = await http(port, 'GET', '/')
    assert status == 200
    assert '<title>MODA Pico Harness</title>' in body

    status, body = await http(port, 'GET', '/app.js')
    assert status == 200 and 'EventSource' in body


@pytest.mark.asyncio
async def test_unknown_path_is_404_not_a_crash(running):
    _harness, _source, _backend, port = running
    status, _ = await http(port, 'GET', '/../config.json')
    assert status == 404


@pytest.mark.asyncio
async def test_status_reports_the_acquisition_setup(running):
    harness, source, _backend, port = running
    source.fill(150)
    status, body = await http(port, 'GET', '/api/status')
    payload = json.loads(body)
    assert status == 200
    assert payload['sample_rate'] == 100
    assert payload['buffered'] == 150
    assert payload['adc_gpio'] == 26
    assert 'secret' not in body


@pytest.mark.asyncio
async def test_config_endpoint_never_returns_secrets(running):
    _harness, _source, _backend, port = running
    status, body = await http(port, 'GET', '/api/config')
    assert status == 200
    assert 'secret-key' not in body
    assert json.loads(body)['backend_api_key_set'] is True


@pytest.mark.asyncio
async def test_saving_a_blank_password_keeps_the_stored_one(running):
    harness, _source, _backend, port = running
    status, _ = await http(port, 'POST', '/api/config',
                           json.dumps({'wifi_ssid': 'lab',
                                       'backend_api_key': ''}))
    assert status == 200
    assert harness.cfg['wifi_ssid'] == 'lab'
    assert harness.cfg['backend_api_key'] == 'secret-key'


@pytest.mark.asyncio
async def test_saving_acquisition_settings_reconfigures_the_sampler(running):
    harness, source, _backend, port = running
    status, body = await http(port, 'POST', '/api/config',
                              json.dumps({'sample_rate': 400,
                                          'buffer_seconds': 5}))
    assert status == 200
    assert json.loads(body)['needs_reboot'] is False
    assert source.reconfigured == (0, 400, 2000)


@pytest.mark.asyncio
async def test_wifi_changes_are_flagged_as_needing_a_reboot(running):
    _harness, _source, _backend, port = running
    _status, body = await http(port, 'POST', '/api/config',
                               json.dumps({'wifi_ssid': 'other-net'}))
    assert json.loads(body)['needs_reboot'] is True


@pytest.mark.asyncio
async def test_invalid_settings_are_rejected_with_a_reason(running):
    harness, _source, _backend, port = running
    status, body = await http(port, 'POST', '/api/config',
                              json.dumps({'sample_rate': 99999}))
    assert status == 400
    assert 'sample_rate' in json.loads(body)['error']
    assert harness.cfg['sample_rate'] == 100      # unchanged


@pytest.mark.asyncio
async def test_signal_snapshot_returns_the_newest_samples(running):
    _harness, source, _backend, port = running
    source.fill(300)
    _status, body = await http(port, 'GET', '/api/signal?n=50')
    payload = json.loads(body)
    assert len(payload['counts']) == 50
    assert payload['counts'][-1] == 299 % 4096
    assert payload['total_samples'] == 300


@pytest.mark.asyncio
async def test_backend_health_is_proxied_with_the_api_key(running):
    _harness, _source, backend, port = running
    status, body = await http(port, 'GET', '/api/backend/health')
    assert status == 200
    assert json.loads(body)['status'] == 200
    method, path, headers, _body = backend.requests[-1]
    assert (method, path) == ('GET', '/health')
    assert headers['x-api-key'] == 'secret-key'


@pytest.mark.asyncio
async def test_analyze_uploads_the_buffer_as_csv_with_the_device_fs(running):
    _harness, source, backend, port = running
    source.fill(120)
    status, body = await http(port, 'POST', '/api/analyze', json.dumps({
        'route': 'analyze_cwt', 'samples': 120,
        'params': {'freq_min': '0.5', 'freq_max': '20'},
    }))
    assert status == 200
    assert json.loads(body)['task_id'] == 'task-1'

    _method, path, headers, upload = backend.requests[-1]
    assert path == '/analyze_cwt'
    # The declared length must match what was actually streamed, or a real
    # WSGI server would block waiting for the rest of the body.
    assert int(headers['content-length']) == len(upload)
    assert b'name="fs"\r\n\r\n100\r\n' in upload      # device rate, not browser
    assert b'name="freq_min"\r\n\r\n0.5\r\n' in upload
    assert upload.count(b'\n') >= 120


@pytest.mark.asyncio
async def test_analyze_rejects_routes_outside_the_allowlist(running):
    _harness, source, backend, port = running
    source.fill(10)
    status, body = await http(port, 'POST', '/api/analyze',
                              json.dumps({'route': 'shutdown'}))
    assert status == 400
    assert 'not allowed' in json.loads(body)['error']
    assert backend.requests == []


@pytest.mark.asyncio
async def test_analyze_refuses_to_upload_an_empty_buffer(running):
    _harness, _source, backend, port = running
    status, _body = await http(port, 'POST', '/api/analyze',
                               json.dumps({'route': 'analyze_cwt'}))
    assert status == 409
    assert backend.requests == []


@pytest.mark.asyncio
async def test_browser_cannot_override_fs_or_smuggle_a_file_field(running):
    _harness, source, backend, port = running
    source.fill(10)
    await http(port, 'POST', '/api/analyze', json.dumps({
        'route': 'analyze_cwt', 'params': {'fs': '9999', 'file': 'evil'},
    }))
    _method, _path, _headers, upload = backend.requests[-1]
    assert b'9999' not in upload
    assert b'name="fs"\r\n\r\n100\r\n' in upload
    assert upload.count(b'filename=') == 1


@pytest.mark.asyncio
async def test_task_status_is_relayed_whole(running):
    _harness, _source, _backend, port = running
    status, body = await http(port, 'GET', '/api/task/task-1')
    assert status == 200
    payload = json.loads(body)                      # i.e. nothing was truncated
    assert payload['status'] == 'complete'
    assert len(payload['results']['cwt_plot']) == len(BIG_PLOT)


@pytest.mark.asyncio
async def test_task_id_is_validated_before_it_reaches_the_backend(running):
    _harness, _source, backend, port = running
    status, _ = await http(port, 'GET', '/api/task/..%2f..%2fetc')
    assert status == 400
    assert backend.requests == []


@pytest.mark.asyncio
async def test_unreachable_backend_becomes_a_502_not_a_hang(running):
    harness, _source, backend, port = running
    await backend.stop()
    status, body = await http(port, 'GET', '/api/backend/health')
    assert status == 502
    assert 'cannot reach' in json.loads(body)['error']
    assert harness.last_error


@pytest.mark.asyncio
async def test_stream_emits_sse_frames_of_new_samples(running):
    _harness, source, _backend, port = running
    reader, writer = await asyncio.open_connection('127.0.0.1', port)
    writer.write(b'GET /api/stream?mark=0 HTTP/1.1\r\nHost: pico\r\n\r\n')
    await writer.drain()
    try:
        head = await asyncio.wait_for(reader.readuntil(b'\r\n\r\n'), 2)
        assert b'text/event-stream' in head

        source.fill(25)
        deadline = asyncio.get_event_loop().time() + 3
        while asyncio.get_event_loop().time() < deadline:
            line = await asyncio.wait_for(reader.readline(), 2)
            if not line.startswith(b'data: '):
                continue
            frame = json.loads(line[6:])
            if frame['counts']:
                assert frame['counts'][:3] == [0, 1, 2]
                assert frame['mark'] == len(frame['counts'])
                return
        pytest.fail('no sample frame arrived within 3s')
    finally:
        writer.close()
