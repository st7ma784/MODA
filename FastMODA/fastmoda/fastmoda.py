"""Core FastMODA processing: sliding-window FFT + changepoint detection

Functions:
 - load_signal(path)
 - sliding_fft(x, fs, win_s, hop_s, nfft)
 - compute_band_powers(Sxx, freqs, bands)
 - detect_changepoints(features, model='l2', pen=10)
"""
import os

import numpy as np
from scipy import io
from scipy.signal import get_window
from numpy.fft import rfft, rfftfreq
import ruptures as rpt
from ruptures.exceptions import BadSegmentationParameters

# Variable names MODA and common recording tools use for the sampling rate.
# Matched case-insensitively against the keys in a .mat file.
_FS_KEYS = ('fs', 'sampling_freq', 'samplingfreq', 'samplerate', 'sample_rate',
            'srate', 'sf', 'freq', 'frequency')


def _flatten_to_1d(x):
    """Reduce a loaded array to a 1-D float signal."""
    x = np.asarray(x).squeeze()
    if x.ndim > 1:
        if x.shape[0] == 1:
            x = x[0, :]
        elif x.shape[1] == 1:
            x = x[:, 0]
        else:
            # Multi-channel: take the longest axis as time and use the first
            # channel, rather than flattening channels end-to-end (which
            # would splice unrelated recordings into one bogus signal).
            if x.shape[0] > x.shape[1]:
                x = x[:, 0]
            else:
                x = x[0, :]
    return x.astype(float)


def _pick_from_mat(entries, varname=None):
    """Choose the signal array and sampling rate from {name: array} entries.

    Returns (x, fs). fs is 1.0 when the file carries no recognisable rate.
    """
    entries = {k: v for k, v in entries.items() if not k.startswith('__')}

    fs = 1.0
    for key, val in list(entries.items()):
        if key.lower() in _FS_KEYS:
            arr = np.asarray(val).squeeze()
            if arr.size == 1 and np.isfinite(float(arr)) and float(arr) > 0:
                fs = float(arr)
                # A scalar rate is never the signal itself.
                entries.pop(key, None)

    if varname:
        if varname not in entries:
            raise ValueError(
                f"Variable '{varname}' not found in .mat file. "
                f"Available: {', '.join(sorted(entries)) or '(none)'}")
        return _flatten_to_1d(entries[varname]), fs

    # Ignore scalars and non-numeric entries; the signal is the biggest
    # numeric array left.
    cand = {k: np.asarray(v) for k, v in entries.items()
            if isinstance(v, np.ndarray) and np.issubdtype(np.asarray(v).dtype, np.number)}
    cand = {k: v for k, v in cand.items() if v.size > 1}
    if not cand:
        raise ValueError(
            'No numeric signal array found in .mat file. '
            f"Variables present: {', '.join(sorted(entries)) or '(none)'}")

    name = max(cand, key=lambda k: cand[k].size)
    return _flatten_to_1d(cand[name]), fs


def _load_mat(path, varname=None):
    """Load a .mat file, transparently handling both classic and v7.3 formats."""
    try:
        return _pick_from_mat(io.loadmat(path), varname)
    except NotImplementedError:
        # MATLAB v7.3 files are HDF5, which scipy.io.loadmat cannot read —
        # this is what MODA writes for larger datasets, so it is the common
        # case rather than an exotic one.
        pass

    try:
        import h5py
    except ImportError:
        raise ValueError(
            'This is a MATLAB v7.3 (HDF5) .mat file. Install h5py to load it, '
            "or re-save from MATLAB with: save('file.mat','var','-v7')")

    with h5py.File(path, 'r') as h5:
        entries = {}
        for key, node in h5.items():
            if isinstance(node, h5py.Dataset) and node.dtype.kind in 'fiu':
                # v7.3 stores arrays transposed relative to MATLAB's layout.
                entries[key] = np.array(node).T
        return _pick_from_mat(entries, varname)


def load_signal(path, varname=None):
    """Load a 1-D signal from .mat (incl. v7.3), .npy, .csv or .txt

    Returns: (x, fs)
    - x: 1D numpy array
    - fs: sampling rate carried by the file, else 1.0

    varname selects a specific variable from a .mat file; without it the
    largest numeric array wins and a scalar named fs/sampling_freq/etc is
    picked up as the sampling rate.
    """
    path = str(path)
    ext = os.path.splitext(path)[1].lower()   # case-insensitive: .MAT == .mat

    if ext == '.mat':
        return _load_mat(path, varname)
    elif ext == '.npy':
        return _flatten_to_1d(np.load(path)), 1.0
    elif ext == '.csv':
        return _flatten_to_1d(np.loadtxt(path, delimiter=',')), 1.0
    elif ext in ('.txt', '.dat', '.asc'):
        # Whitespace-delimited, the other format MATLAB users routinely export.
        return _flatten_to_1d(np.loadtxt(path)), 1.0
    else:
        raise ValueError(
            f"Unsupported file type '{ext or path}'. "
            'Supported: .mat, .npy, .csv, .txt, .dat, .asc')

def sliding_fft(x, fs=1.0, win_s=1.0, hop_s=None, nfft=None, window='hann'):
    """Compute sliding-window FFT magnitudes.

    Args:
      x: 1D signal
      fs: sampling frequency
      win_s: window length in seconds
      hop_s: hop length in seconds (defaults to win_s/4)
      nfft: FFT length (defaults to next pow2 of window samples)

    Returns: freqs, times, Sxx (magnitude spectrogram)
    """
    x = np.asarray(x).squeeze()
    if x.ndim > 1:
        # Try to flatten multi-dimensional arrays
        if x.shape[0] == 1:
            x = x[0, :]
        elif x.shape[1] == 1:
            x = x[:, 0]
        else:
            x = x.flatten()
    
    if x.ndim != 1:
        raise ValueError(f'x must be 1D, got shape {x.shape}')
    N = x.shape[0]
    win_n = int(round(win_s * fs))
    if hop_s is None:
        hop_n = max(1, win_n // 4)
    else:
        hop_n = int(round(hop_s * fs))
    if nfft is None:
        nfft = 1 << (win_n - 1).bit_length()
    w = get_window(window, win_n, fftbins=True)

    frames = []
    times = []
    for start in range(0, max(1, N - win_n + 1), hop_n):
        frame = x[start:start+win_n]
        if frame.shape[0] < win_n:
            frame = np.pad(frame, (0, win_n - frame.shape[0]))
        frame = frame * w
        X = rfft(frame, n=nfft)
        frames.append(np.abs(X))
        times.append((start + win_n/2) / fs)
    Sxx = np.vstack(frames).T  # freq x time
    freqs = rfftfreq(nfft, 1.0/fs)
    return freqs, np.array(times), Sxx

def compute_band_powers(Sxx, freqs, bands=None, eps=1e-12):
    """Aggregate spectral energy into bands.

    bands: list of (fmin, fmax, name) tuples. If None, returns full-band power.
    Returns: features (time x bands)
    """
    if bands is None:
        # full band
        power = np.sum(Sxx**2, axis=0)
        return power.reshape(-1,1), ['full']
    band_pows = []
    names = []
    for fmin, fmax, name in bands:
        idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]
        if idx.size == 0:
            band_pows.append(np.zeros(Sxx.shape[1]))
        else:
            band_pows.append(np.sum(Sxx[idx,:]**2, axis=0))
        names.append(name)
    feats = np.vstack(band_pows).T
    # avoid zeros
    feats = np.log(feats + eps)
    return feats, names

def detect_changepoints(features, model='l2', pen=10):
    """Detect changepoints on multivariate features using ruptures.

    Returns: indices (change locations in sample frames)
    """
    try:
        algo = rpt.Pelt(model=model).fit(features)
        # pen may need tuning; expose as parameter
        bkps = algo.predict(pen=pen)
        # ruptures returns 1-based index of last segment end; convert to zero-based positions
        return np.array(bkps[:-1], dtype=int)
    except BadSegmentationParameters:
        print(f"Too few time points ({features.shape[0]}) for changepoint detection at pen={pen}; reporting 0 changepoints")
        return np.array([], dtype=int)

def extract_instantaneous_frequency(Sxx, freqs, times):
    """Extract dominant frequency at each time point.
    
    Returns: inst_freq (array of dominant frequencies over time)
    """
    # Find peak frequency at each time
    peak_idx = np.argmax(Sxx, axis=0)
    inst_freq = freqs[peak_idx]
    return inst_freq

def extract_band_frequencies(Sxx, freqs, times, bands):
    """Extract dominant frequency for each band over time.
    
    Returns: dict mapping band_name -> (times, frequencies, amplitudes)
    """
    result = {}
    for fmin, fmax, name in bands:
        idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]
        if idx.size == 0:
            result[name] = (times, np.zeros_like(times), np.zeros_like(times))
        else:
            band_spec = Sxx[idx, :]
            peak_idx = np.argmax(band_spec, axis=0)
            band_freqs = freqs[idx[peak_idx]]
            band_amps = band_spec[peak_idx, np.arange(len(times))]
            result[name] = (times, band_freqs, band_amps)
    return result

def fit_sine_segments(x, fs, times, segments):
    """Fit sine wave to signal segments and detect parameter changes.
    
    Args:
        x: original signal
        fs: sampling rate
        times: time points from spectrogram
        segments: list of (start_idx, end_idx) tuples in time array
        
    Returns: list of dicts with {segment_idx, freq, amp, phase, fit_error}
    """
    from scipy.optimize import curve_fit
    
    def sine_model(t, amp, freq, phase):
        return amp * np.sin(2 * np.pi * freq * t + phase)
    
    results = []
    for seg_idx, (start, end) in enumerate(segments):
        if end <= start:
            continue
        t_start = times[start]
        t_end = times[end] if end < len(times) else times[-1]
        
        # Get signal segment
        idx_start = int(t_start * fs)
        idx_end = int(t_end * fs)
        if idx_end > len(x):
            idx_end = len(x)
        if idx_end <= idx_start:
            continue
            
        t_seg = np.arange(idx_start, idx_end) / fs
        x_seg = x[idx_start:idx_end]
        
        # Initial guess from FFT
        X = np.fft.rfft(x_seg)
        freqs_fft = np.fft.rfftfreq(len(x_seg), 1/fs)
        peak_idx = np.argmax(np.abs(X[1:])) + 1  # skip DC
        freq_guess = freqs_fft[peak_idx]
        amp_guess = 2 * np.abs(X[peak_idx]) / len(x_seg)
        
        try:
            popt, _ = curve_fit(
                sine_model, t_seg, x_seg,
                p0=[amp_guess, freq_guess, 0],
                bounds=([0, 0, -2*np.pi], [np.inf, fs/2, 2*np.pi]),
                maxfev=2000
            )
            fit = sine_model(t_seg, *popt)
            error = np.sqrt(np.mean((x_seg - fit)**2))
            
            results.append({
                'segment': seg_idx,
                'time_range': (t_start, t_end),
                'amplitude': popt[0],
                'frequency': popt[1],
                'phase': popt[2],
                'fit_error': error
            })
        except:
            # Fitting failed, use FFT estimates
            results.append({
                'segment': seg_idx,
                'time_range': (t_start, t_end),
                'amplitude': amp_guess,
                'frequency': freq_guess,
                'phase': 0,
                'fit_error': np.inf
            })
    
    return results

def detect_periodicity_changes(x, fs, times, cps, tolerance=0.1):
    """Detect when periodic patterns (frequency/amplitude) change significantly.
    
    Args:
        x: original signal
        fs: sampling rate
        times: time array from spectrogram
        cps: changepoint indices
        tolerance: relative change threshold for detecting breaks
        
    Returns: dict with periodicity analysis
    """
    # Create segments from changepoints
    segments = []
    starts = [0] + list(cps)
    ends = list(cps) + [len(times)]
    for s, e in zip(starts, ends):
        segments.append((s, e))
    
    # Fit sine to each segment
    sine_fits = fit_sine_segments(x, fs, times, segments)
    
    # Detect significant changes in frequency/amplitude
    freq_changes = []
    amp_changes = []
    
    for i in range(1, len(sine_fits)):
        prev = sine_fits[i-1]
        curr = sine_fits[i]
        
        if prev['frequency'] > 0:
            freq_rel_change = abs(curr['frequency'] - prev['frequency']) / prev['frequency']
            if freq_rel_change > tolerance:
                freq_changes.append({
                    'time': curr['time_range'][0],
                    'from_freq': prev['frequency'],
                    'to_freq': curr['frequency'],
                    'rel_change': freq_rel_change
                })
        
        if prev['amplitude'] > 0:
            amp_rel_change = abs(curr['amplitude'] - prev['amplitude']) / prev['amplitude']
            if amp_rel_change > tolerance:
                amp_changes.append({
                    'time': curr['time_range'][0],
                    'from_amp': prev['amplitude'],
                    'to_amp': curr['amplitude'],
                    'rel_change': amp_rel_change
                })
    
    return {
        'sine_fits': sine_fits,
        'frequency_changes': freq_changes,
        'amplitude_changes': amp_changes
    }
