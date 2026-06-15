"""
fastmoda.storage unit tests.

storage.py derives DATA_DIR/DB_PATH/RECORDINGS_DIR/MODELS_DIR from
MODA_DATA_DIR at *import time*, so each test points MODA_DATA_DIR at a fresh
tmp_path and reloads the module to pick up that layout.

Run from repo root:
    pytest tests/test_storage.py -v
"""

import importlib
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "FastMODA"))

from fastmoda import storage as storage_module  # noqa: E402


@pytest.fixture
def storage(tmp_path, monkeypatch):
    monkeypatch.setenv('MODA_DATA_DIR', str(tmp_path))
    module = importlib.reload(storage_module)
    module.init_db()
    return module


def test_init_db_creates_layout(storage):
    assert Path(storage.DB_PATH).exists()
    assert Path(storage.RECORDINGS_DIR).is_dir()
    assert Path(storage.MODELS_DIR).is_dir()


def test_upsert_device_is_idempotent(storage):
    storage.upsert_device('device-1', label='phone')
    storage.upsert_device('device-1', label='phone-again')

    conn = storage.get_db()
    try:
        rows = conn.execute('SELECT * FROM devices').fetchall()
    finally:
        conn.close()

    assert len(rows) == 1
    assert rows[0]['label'] == 'phone'


def test_save_and_get_recording(storage):
    storage.save_recording(
        'rec-1', 'device-1', '/data/recordings/device-1/rec-1.npy',
        sampling_rate=256.0, signal_length=1024, signal_type='ecg',
    )

    rec = storage.get_recording('rec-1')
    assert rec['device_id'] == 'device-1'
    assert rec['signal_type'] == 'ecg'
    assert rec['is_baseline'] == 0

    recordings = storage.list_recordings('device-1')
    assert len(recordings) == 1
    assert recordings[0]['id'] == 'rec-1'
    assert recordings[0]['has_label'] == 0


def test_get_recording_unknown_returns_none(storage):
    assert storage.get_recording('does-not-exist') is None


def test_recording_path_creates_device_dir(storage):
    path = storage.recording_path('device-1', 'rec-1')

    assert Path(path).parent.is_dir()
    assert path.endswith('rec-1.npy')


def test_mark_recording_baseline(storage):
    storage.save_recording('rec-1', 'device-1', 'rec-1.npy', 256.0, 1024)
    storage.mark_recording_baseline('rec-1')

    rec = storage.get_recording('rec-1')
    assert rec['is_baseline'] == 1


def test_save_and_get_features(storage):
    storage.save_recording('rec-1', 'device-1', 'rec-1.npy', 256.0, 1024)
    storage.save_features('rec-1', ['a', 'b'], [1.0, 2.0])

    names, vector = storage.get_features('rec-1')
    assert names == ['a', 'b']
    assert vector == [1.0, 2.0]


def test_update_baseline_welford(storage):
    storage.upsert_device('device-1')

    storage.update_baseline('device-1', ['hr'], [1.0])
    storage.update_baseline('device-1', ['hr'], [3.0])
    baseline = storage.update_baseline('device-1', ['hr'], [5.0])

    assert baseline['n_samples'] == 3
    hr = baseline['features']['hr']
    assert hr['n'] == 3
    assert hr['mean'] == pytest.approx(3.0)
    # Welford m2 after [1, 3, 5] is 8 -> std = sqrt(m2 / n) = sqrt(8/3)
    assert hr['std'] == pytest.approx((8.0 / 3) ** 0.5)


def test_get_baseline_empty_for_unknown_device(storage):
    assert storage.get_baseline('does-not-exist') == {'n_samples': 0, 'features': {}}


def test_save_label_and_queue(storage):
    storage.save_recording('rec-1', 'device-1', 'rec-1.npy', 256.0, 1024)
    storage.save_recording('rec-2', 'device-1', 'rec-2.npy', 256.0, 1024)

    storage.save_label('rec-1', 'normal', source='self')
    queue_ids = {row['id'] for row in storage.get_label_queue()}
    # A self-report alone doesn't take a recording out of the reviewer queue.
    assert {'rec-1', 'rec-2'} <= queue_ids

    storage.save_label('rec-1', 'afib', source='reviewer', reviewer='dr-x', confidence=0.9)
    queue_ids = {row['id'] for row in storage.get_label_queue()}
    assert 'rec-1' not in queue_ids
    assert 'rec-2' in queue_ids

    labels = storage.get_labels('rec-1')
    assert len(labels) == 2


def test_save_classification_run(storage):
    storage.save_recording('rec-1', 'device-1', 'rec-1.npy', 256.0, 1024)
    storage.save_classification_run('rec-1', 'afib', 0.83, {'top_features': []}, model_version='v1')

    conn = storage.get_db()
    try:
        rows = conn.execute(
            'SELECT * FROM classification_runs WHERE recording_id = ?', ('rec-1',)
        ).fetchall()
    finally:
        conn.close()

    assert len(rows) == 1
    assert rows[0]['condition'] == 'afib'
    assert rows[0]['probability'] == pytest.approx(0.83)
