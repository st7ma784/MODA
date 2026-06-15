"""SQLite-backed persistence for recordings, baselines, labels and classification runs.

Data layout (under MODA_DATA_DIR, default "<repo>/FastMODA/data"):
    moda.db                 - sqlite database (devices, recordings, features,
                               baseline_stats, labels, classification_runs)
    recordings/<device_id>/<recording_id>.npy
    models/<condition>.joblib, models/meta.json
"""

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

DATA_DIR = os.environ.get(
    'MODA_DATA_DIR',
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'),
)
DB_PATH = os.path.join(DATA_DIR, 'moda.db')
RECORDINGS_DIR = os.path.join(DATA_DIR, 'recordings')
MODELS_DIR = os.path.join(DATA_DIR, 'models')

SCHEMA = """
CREATE TABLE IF NOT EXISTS devices (
    id TEXT PRIMARY KEY,
    label TEXT,
    created_at TEXT NOT NULL,
    consent INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS recordings (
    id TEXT PRIMARY KEY,
    device_id TEXT NOT NULL,
    filepath TEXT NOT NULL,
    sampling_rate REAL,
    signal_length INTEGER,
    signal_type TEXT,
    recorded_at TEXT,
    uploaded_at TEXT NOT NULL,
    is_baseline INTEGER DEFAULT 0,
    FOREIGN KEY (device_id) REFERENCES devices(id)
);
CREATE INDEX IF NOT EXISTS idx_recordings_device ON recordings(device_id);

CREATE TABLE IF NOT EXISTS features (
    recording_id TEXT PRIMARY KEY,
    feature_names TEXT NOT NULL,
    feature_vector TEXT NOT NULL,
    computed_at TEXT NOT NULL,
    FOREIGN KEY (recording_id) REFERENCES recordings(id)
);

CREATE TABLE IF NOT EXISTS baseline_stats (
    device_id TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    n INTEGER NOT NULL,
    mean REAL NOT NULL,
    m2 REAL NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (device_id, feature_name)
);

CREATE TABLE IF NOT EXISTS labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recording_id TEXT NOT NULL,
    condition TEXT NOT NULL,
    severity TEXT,
    source TEXT NOT NULL,
    reviewer TEXT,
    confidence REAL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (recording_id) REFERENCES recordings(id)
);
CREATE INDEX IF NOT EXISTS idx_labels_recording ON labels(recording_id);

CREATE TABLE IF NOT EXISTS classification_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recording_id TEXT NOT NULL,
    condition TEXT NOT NULL,
    probability REAL NOT NULL,
    explanation TEXT,
    model_version TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (recording_id) REFERENCES recordings(id)
);
CREATE INDEX IF NOT EXISTS idx_classification_runs_recording ON classification_runs(recording_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_db() -> sqlite3.Connection:
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA journal_mode=WAL')
    return conn


def init_db() -> None:
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    conn = get_db()
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


def recording_path(device_id: str, recording_id: str) -> str:
    """Return (and ensure the directory exists for) the .npy path for a recording."""
    device_dir = os.path.join(RECORDINGS_DIR, device_id)
    os.makedirs(device_dir, exist_ok=True)
    return os.path.join(device_dir, f'{recording_id}.npy')


def _upsert_device(conn: sqlite3.Connection, device_id: str, label: Optional[str] = None) -> None:
    conn.execute(
        'INSERT INTO devices (id, label, created_at) VALUES (?, ?, ?) '
        'ON CONFLICT(id) DO NOTHING',
        (device_id, label, _now()),
    )


def upsert_device(device_id: str, label: Optional[str] = None) -> None:
    conn = get_db()
    try:
        _upsert_device(conn, device_id, label)
        conn.commit()
    finally:
        conn.close()


def save_recording(recording_id: str, device_id: str, filepath: str,
                    sampling_rate: float, signal_length: int,
                    signal_type: Optional[str] = None,
                    recorded_at: Optional[str] = None,
                    is_baseline: bool = False) -> None:
    now = _now()
    conn = get_db()
    try:
        _upsert_device(conn, device_id)
        conn.execute(
            'INSERT INTO recordings '
            '(id, device_id, filepath, sampling_rate, signal_length, signal_type, '
            ' recorded_at, uploaded_at, is_baseline) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (recording_id, device_id, filepath, sampling_rate, signal_length,
             signal_type, recorded_at or now, now, int(bool(is_baseline))),
        )
        conn.commit()
    finally:
        conn.close()


def get_recording(recording_id: str) -> Optional[Dict]:
    conn = get_db()
    try:
        row = conn.execute('SELECT * FROM recordings WHERE id = ?', (recording_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_recordings(device_id: str) -> List[Dict]:
    conn = get_db()
    try:
        rows = conn.execute(
            'SELECT r.*, '
            '       EXISTS(SELECT 1 FROM labels l WHERE l.recording_id = r.id) AS has_label '
            'FROM recordings r WHERE r.device_id = ? ORDER BY r.uploaded_at DESC',
            (device_id,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def mark_recording_baseline(recording_id: str) -> None:
    conn = get_db()
    try:
        conn.execute('UPDATE recordings SET is_baseline = 1 WHERE id = ?', (recording_id,))
        conn.commit()
    finally:
        conn.close()


def save_features(recording_id: str, feature_names: Sequence[str], feature_vector: Sequence[float]) -> None:
    names_json = json.dumps(list(feature_names))
    vector_json = json.dumps([float(v) for v in feature_vector])
    now = _now()
    conn = get_db()
    try:
        conn.execute(
            'INSERT INTO features (recording_id, feature_names, feature_vector, computed_at) '
            'VALUES (?, ?, ?, ?) '
            'ON CONFLICT(recording_id) DO UPDATE SET '
            '  feature_names = excluded.feature_names, '
            '  feature_vector = excluded.feature_vector, '
            '  computed_at = excluded.computed_at',
            (recording_id, names_json, vector_json, now),
        )
        conn.commit()
    finally:
        conn.close()


def get_features(recording_id: str):
    conn = get_db()
    try:
        row = conn.execute(
            'SELECT feature_names, feature_vector FROM features WHERE recording_id = ?',
            (recording_id,),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row['feature_names']), json.loads(row['feature_vector'])
    finally:
        conn.close()


def get_baseline(device_id: str) -> Dict:
    """Return {'n_samples': int, 'features': {name: {mean, std, n}}}."""
    conn = get_db()
    try:
        rows = conn.execute(
            'SELECT feature_name, n, mean, m2 FROM baseline_stats WHERE device_id = ?',
            (device_id,),
        ).fetchall()
        features = {}
        n_samples = 0
        for row in rows:
            n = row['n']
            n_samples = max(n_samples, n)
            std = (row['m2'] / n) ** 0.5 if n > 0 else 0.0
            features[row['feature_name']] = {'mean': row['mean'], 'std': std, 'n': n}
        return {'n_samples': n_samples, 'features': features}
    finally:
        conn.close()


def update_baseline(device_id: str, feature_names: Sequence[str], feature_vector: Sequence[float]) -> Dict:
    """Welford-update the running mean/variance for each feature and return the new baseline."""
    now = _now()
    conn = get_db()
    try:
        _upsert_device(conn, device_id)
        for name, value in zip(feature_names, feature_vector):
            value = float(value)
            row = conn.execute(
                'SELECT n, mean, m2 FROM baseline_stats WHERE device_id = ? AND feature_name = ?',
                (device_id, name),
            ).fetchone()
            if row is None:
                n, mean, m2 = 0, 0.0, 0.0
            else:
                n, mean, m2 = row['n'], row['mean'], row['m2']

            n += 1
            delta = value - mean
            mean += delta / n
            m2 += delta * (value - mean)

            conn.execute(
                'INSERT INTO baseline_stats (device_id, feature_name, n, mean, m2, updated_at) '
                'VALUES (?, ?, ?, ?, ?, ?) '
                'ON CONFLICT(device_id, feature_name) DO UPDATE SET '
                '  n = excluded.n, mean = excluded.mean, m2 = excluded.m2, updated_at = excluded.updated_at',
                (device_id, name, n, mean, m2, now),
            )
        conn.commit()
    finally:
        conn.close()
    return get_baseline(device_id)


def save_label(recording_id: str, condition: str, severity: Optional[str] = None,
                source: str = 'self', reviewer: Optional[str] = None,
                confidence: Optional[float] = None) -> None:
    conn = get_db()
    try:
        conn.execute(
            'INSERT INTO labels (recording_id, condition, severity, source, reviewer, confidence, created_at) '
            'VALUES (?, ?, ?, ?, ?, ?, ?)',
            (recording_id, condition, severity, source, reviewer, confidence, _now()),
        )
        conn.commit()
    finally:
        conn.close()


def get_labels(recording_id: str) -> List[Dict]:
    conn = get_db()
    try:
        rows = conn.execute(
            'SELECT * FROM labels WHERE recording_id = ? ORDER BY created_at DESC',
            (recording_id,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_label_queue(limit: int = 20) -> List[Dict]:
    """Recordings that have not yet received a reviewer label, oldest first."""
    conn = get_db()
    try:
        rows = conn.execute(
            'SELECT r.id, r.device_id, r.recorded_at, r.uploaded_at, r.signal_type, r.is_baseline '
            'FROM recordings r '
            "WHERE NOT EXISTS (SELECT 1 FROM labels l WHERE l.recording_id = r.id AND l.source = 'reviewer') "
            'ORDER BY r.uploaded_at ASC LIMIT ?',
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def save_classification_run(recording_id: str, condition: str, probability: float,
                              explanation, model_version: Optional[str] = None) -> None:
    conn = get_db()
    try:
        conn.execute(
            'INSERT INTO classification_runs (recording_id, condition, probability, explanation, model_version, created_at) '
            'VALUES (?, ?, ?, ?, ?, ?)',
            (recording_id, condition, float(probability), json.dumps(explanation), model_version, _now()),
        )
        conn.commit()
    finally:
        conn.close()
