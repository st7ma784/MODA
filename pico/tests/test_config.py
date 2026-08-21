import json

import pytest

import config


def test_defaults_when_file_missing(tmp_path):
    cfg = config.load(str(tmp_path / 'nope.json'))
    assert cfg == config.DEFAULTS


def test_corrupt_file_falls_back_rather_than_raising(tmp_path):
    path = tmp_path / 'config.json'
    path.write_text('{not json')
    assert config.load(str(path)) == config.DEFAULTS


def test_round_trip_keeps_only_known_keys(tmp_path):
    path = str(tmp_path / 'config.json')
    config.save({'wifi_ssid': 'lab', 'sample_rate': 500, 'junk': 1}, path)
    stored = json.loads(open(path).read())
    assert 'junk' not in stored
    cfg = config.load(path)
    assert cfg['wifi_ssid'] == 'lab'
    assert cfg['sample_rate'] == 500
    assert cfg['backend_url'] == config.DEFAULTS['backend_url']


def test_a_single_bad_stored_value_does_not_poison_the_rest(tmp_path):
    path = tmp_path / 'config.json'
    path.write_text(json.dumps({'sample_rate': 99999, 'wifi_ssid': 'lab'}))
    cfg = config.load(str(path))
    assert cfg['sample_rate'] == config.DEFAULTS['sample_rate']
    assert cfg['wifi_ssid'] == 'lab'


@pytest.mark.parametrize('changes', [
    {'sample_rate': 0},
    {'sample_rate': 5000},
    {'adc_channel': 3},
    {'buffer_seconds': 'many'},
    {'volts_full_scale': 12},
])
def test_out_of_range_updates_are_rejected(changes):
    with pytest.raises(config.ConfigError):
        config.update(dict(config.DEFAULTS), changes)


def test_unknown_setting_is_rejected():
    with pytest.raises(config.ConfigError):
        config.update(dict(config.DEFAULTS), {'root_shell': 'yes'})


def test_update_does_not_mutate_the_original():
    original = dict(config.DEFAULTS)
    config.update(original, {'sample_rate': 400})
    assert original['sample_rate'] == config.DEFAULTS['sample_rate']


def test_numeric_strings_from_the_form_are_coerced():
    cfg = config.update(dict(config.DEFAULTS),
                        {'sample_rate': '400', 'volts_full_scale': '3.28'})
    assert cfg['sample_rate'] == 400
    assert cfg['volts_full_scale'] == pytest.approx(3.28)


def test_public_never_leaks_secrets():
    cfg = config.update(dict(config.DEFAULTS),
                        {'wifi_password': 'hunter2', 'backend_api_key': 'k'})
    view = config.public(cfg)
    assert 'hunter2' not in json.dumps(view)
    assert 'k' not in [view.get('backend_api_key')]
    assert view['wifi_password_set'] is True
    assert view['backend_api_key_set'] is True
    assert view['ap_password_set'] is True   # non-empty default


def test_buffer_samples():
    cfg = config.update(dict(config.DEFAULTS),
                        {'sample_rate': 200, 'buffer_seconds': 15})
    assert config.buffer_samples(cfg) == 3000
