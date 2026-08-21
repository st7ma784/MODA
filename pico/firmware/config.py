"""Persistent configuration for the Pico harness.

Kept free of ``machine``/``network`` imports so it can be unit-tested on
CPython (see ``pico/tests/test_config.py``) and reused by the host simulator.
"""

try:
    import ujson as json
except ImportError:
    import json

CONFIG_PATH = '/config.json'

# Anything not present in the stored file falls back to these. Adding a key here
# is enough to make it settable from the web UI - `update()` validates against
# this table rather than a separate schema.
DEFAULTS = {
    'wifi_ssid': '',
    'wifi_password': '',
    'ap_ssid': 'moda-pico',
    'ap_password': 'modamoda',          # >= 8 chars, WPA2 minimum
    'hostname': 'moda-pico',
    'device_id': 'pico-01',
    'backend_url': 'http://192.168.1.10:5000',
    'backend_api_key': '',
    'adc_channel': 0,                   # 0..2 -> GP26/GP27/GP28
    'sample_rate': 200,                 # Hz
    'buffer_seconds': 20,
    'volts_full_scale': 3.3,            # ADC_VREF, for counts -> volts
}

# (min, max) for numeric keys. Enforced by `update()`; everything else is
# coerced to a string. Upper sample-rate bound is what a MicroPython soft-IRQ
# timer callback reliably sustains, not what the SAR ADC can do (500 kS/s).
_RANGES = {
    'adc_channel': (0, 2),
    'sample_rate': (1, 2000),
    'buffer_seconds': (1, 120),
    'volts_full_scale': (0.1, 5.0),
}

_SECRET_KEYS = ('wifi_password', 'ap_password', 'backend_api_key')


class ConfigError(ValueError):
    """Raised when a submitted config value is out of range or malformed."""


def _coerce(key, value):
    if key not in _RANGES:
        return str(value)
    lo, hi = _RANGES[key]
    try:
        num = float(value) if isinstance(lo, float) else int(value)
    except (TypeError, ValueError):
        raise ConfigError('%s must be a number' % key)
    if not (lo <= num <= hi):
        raise ConfigError('%s must be between %s and %s' % (key, lo, hi))
    return num


def load(path=CONFIG_PATH):
    """Return the stored config merged over `DEFAULTS`.

    A missing or corrupt file yields the defaults rather than raising - a Pico
    that cannot parse its own config must still come up far enough to serve the
    provisioning UI that lets you fix it.
    """
    cfg = dict(DEFAULTS)
    try:
        with open(path) as fh:
            stored = json.load(fh)
    except (OSError, ValueError):
        return cfg
    if isinstance(stored, dict):
        for key, value in stored.items():
            if key in DEFAULTS:
                try:
                    cfg[key] = _coerce(key, value)
                except ConfigError:
                    pass  # keep the default for this one key
    return cfg


def save(cfg, path=CONFIG_PATH):
    """Write only the recognised keys, so unknown junk never round-trips."""
    payload = {key: cfg[key] for key in DEFAULTS if key in cfg}
    with open(path, 'w') as fh:
        json.dump(payload, fh)
    return payload


def update(cfg, changes):
    """Validate `changes` and return a new config dict. Does not persist."""
    merged = dict(cfg)
    for key, value in changes.items():
        if key not in DEFAULTS:
            raise ConfigError('unknown setting: %s' % key)
        merged[key] = _coerce(key, value)
    return merged


def public(cfg):
    """Config safe to hand to the browser: secrets replaced by a set/unset flag."""
    out = {key: value for key, value in cfg.items() if key not in _SECRET_KEYS}
    for key in _SECRET_KEYS:
        out[key + '_set'] = bool(cfg.get(key))
    return out


def buffer_samples(cfg):
    return int(cfg['sample_rate']) * int(cfg['buffer_seconds'])
