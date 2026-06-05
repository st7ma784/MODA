"""Real-time microphone capture with FFT; broadcasts frames to SSE subscribers."""
import threading
import queue
import time
import numpy as np
from typing import List, Optional

SAMPLE_RATE  = 44100
FFT_SIZE     = 2048
HOP_SIZE     = 1024   # blocksize → ~43 candidate frames/s from sounddevice
TARGET_FPS   = 15     # actual frames sent per second
MAX_FREQ_HZ  = 8000   # upper display limit

_lock: threading.Lock = threading.Lock()
_subscribers: List[queue.Queue] = []
_thread: Optional[threading.Thread] = None
_running: bool = False

_FREQS: Optional[np.ndarray] = None
_FREQ_MASK: Optional[np.ndarray] = None


def _freq_axis():
    global _FREQS, _FREQ_MASK
    if _FREQS is None:
        all_freqs  = np.fft.rfftfreq(FFT_SIZE, 1.0 / SAMPLE_RATE)
        _FREQ_MASK = all_freqs <= MAX_FREQ_HZ
        _FREQS     = all_freqs[_FREQ_MASK]
    return _FREQS, _FREQ_MASK


# ── public API ────────────────────────────────────────────────────────────────

def is_available() -> bool:
    try:
        import sounddevice  # noqa: F401
        return True
    except ImportError:
        return False


def subscribe() -> queue.Queue:
    """Return a frame queue; starts capture thread on first call."""
    q: queue.Queue = queue.Queue(maxsize=30)
    with _lock:
        _subscribers.append(q)
        if not _running:
            _start()
    return q


def unsubscribe(q: queue.Queue) -> None:
    with _lock:
        try:
            _subscribers.remove(q)
        except ValueError:
            pass


def status() -> dict:
    return {"available": is_available(), "running": _running,
            "subscribers": len(_subscribers),
            "sample_rate": SAMPLE_RATE, "fft_size": FFT_SIZE,
            "max_freq_hz": MAX_FREQ_HZ}


# ── internals ─────────────────────────────────────────────────────────────────

def _start() -> None:
    global _thread, _running
    _running = True
    _thread  = threading.Thread(target=_capture_loop, daemon=True,
                                 name="audio-capture")
    _thread.start()


def _broadcast(frame: dict) -> None:
    with _lock:
        dead = []
        for q in _subscribers:
            try:
                q.put_nowait(frame)
            except queue.Full:
                dead.append(q)
        for q in dead:
            _subscribers.remove(q)


def _build_frame(samples: np.ndarray) -> dict:
    freqs, mask = _freq_axis()
    window  = np.hanning(len(samples))
    spec    = np.fft.rfft(samples * window, n=FFT_SIZE)
    mags_db = 20.0 * np.log10(np.abs(spec)[mask] + 1e-10)
    rms_db  = float(20.0 * np.log10(np.sqrt(np.mean(samples ** 2)) + 1e-10))
    return {
        "frequencies":   freqs.tolist(),
        "magnitudes_db": mags_db.tolist(),
        "rms_db":        round(rms_db, 1),
        "peak_db":       round(float(np.max(mags_db)), 1),
        "timestamp":     time.time(),
    }


def _capture_loop() -> None:
    global _running
    try:
        import sounddevice as sd
    except ImportError:
        _running = False
        return

    frame_interval = 1.0 / TARGET_FPS
    buf     = np.zeros(FFT_SIZE, dtype=np.float32)
    last_tx = 0.0

    def _cb(indata, frames, _t, _status):
        nonlocal buf, last_tx
        chunk             = indata[:, 0]
        buf               = np.roll(buf, -len(chunk))
        buf[-len(chunk):] = chunk
        now = time.time()
        if now - last_tx >= frame_interval:
            last_tx = now
            _broadcast(_build_frame(buf.copy()))

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                             blocksize=HOP_SIZE, callback=_cb):
            while _running:
                time.sleep(0.05)
    except Exception as exc:
        print(f"[audio_capture] error: {exc}")
    finally:
        _running = False
