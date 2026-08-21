import pytest

import proxy


@pytest.mark.parametrize('url, expected', [
    ('http://10.0.0.5:5000', ('10.0.0.5', 5000, '')),
    ('http://10.0.0.5', ('10.0.0.5', 80, '')),
    ('10.0.0.5:5000', ('10.0.0.5', 5000, '')),
    ('http://moda.lan:5000/', ('moda.lan', 5000, '')),
    ('http://moda.lan:5000/fastmoda', ('moda.lan', 5000, '/fastmoda')),
    ('  http://moda.lan:5000  ', ('moda.lan', 5000, '')),
])
def test_split_url(url, expected):
    assert proxy.split_url(url) == expected


@pytest.mark.parametrize('url', ['', None, 'http://', 'http://host:abc'])
def test_split_url_rejects_junk(url):
    with pytest.raises(proxy.BackendError):
        proxy.split_url(url)


def test_https_is_refused_with_an_explanation():
    with pytest.raises(proxy.BackendError, match='https'):
        proxy.split_url('https://moda.lan:5000')


def test_every_csv_line_is_exactly_line_bytes_wide():
    # This is what makes the streamed Content-Length exact; if it ever stops
    # holding, uploads truncate or hang instead of failing loudly.
    for volts in (0, 0.000001, 1.65, 3.3, 9.999999, -5, 1e9, 12.3456789):
        assert len(proxy.format_sample(volts)) == proxy.LINE_BYTES


def test_format_sample_clamps_rather_than_widening():
    assert proxy.format_sample(-1) == '0.000000\n'
    assert proxy.format_sample(1e6) == '9.999999\n'


def test_multipart_content_length_matches_the_bytes_actually_sent():
    body = ''.join(proxy.format_sample(v) for v in (0.1, 0.2, 0.3))
    prologue, epilogue, length = proxy.multipart(
        'BND', 'pico.csv', {'fs': '200'}, proxy.LINE_BYTES * 3)
    assert len(prologue + body + epilogue) == length


def test_multipart_carries_the_form_fields_and_filename():
    prologue, epilogue, _ = proxy.multipart(
        'BND', 'pico.csv', {'fs': '200', 'device_id': 'pico-01'}, 0)
    assert 'name="fs"\r\n\r\n200\r\n' in prologue
    assert 'name="device_id"\r\n\r\npico-01\r\n' in prologue
    assert 'filename="pico.csv"' in prologue
    assert prologue.endswith('Content-Type: text/csv\r\n\r\n')
    assert epilogue == '\r\n--BND--\r\n'


def test_request_head_is_well_formed_and_skips_blank_headers():
    head = proxy.request_head('GET', '/health', 'moda.lan',
                              {'X-API-Key': '', 'Accept': 'application/json'})
    assert head.startswith('GET /health HTTP/1.1\r\nHost: moda.lan\r\n')
    assert 'X-API-Key' not in head
    assert 'Accept: application/json' in head
    assert head.endswith('\r\n\r\n')


def test_backend_from_config():
    backend = proxy.backend_from_config(
        {'backend_url': 'http://moda.lan:5000/base', 'backend_api_key': 'k'})
    assert backend == {'host': 'moda.lan', 'port': 5000,
                       'prefix': '/base', 'api_key': 'k'}


def test_status_code_parsing():
    assert proxy._status_code(b'HTTP/1.1 200 OK\r\n') == 200
    assert proxy._status_code(b'HTTP/1.1 502 Bad Gateway\r\n') == 502
    with pytest.raises(proxy.BackendError):
        proxy._status_code(b'garbage\r\n')
