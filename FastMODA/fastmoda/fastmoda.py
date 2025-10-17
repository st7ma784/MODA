"""Core FastMODA processing: sliding-window FFT + changepoint detection

Functions:
 - load_signal(path)
 - sliding_fft(x, fs, win_s, hop_s, nfft)
 - compute_band_powers(Sxx, freqs, bands)
 - detect_changepoints(features, model='l2', pen=10)
"""
import numpy as np
from scipy import io
from scipy.signal import get_window
from numpy.fft import rfft, rfftfreq
import ruptures as rpt

def load_signal(path, varname=None):
    """Load a 1-D signal from .mat, .npy or .csv

    Returns: (x, fs)
    - x: 1D numpy array
    - fs: sampling rate if not known returns 1.0
    """
    path = str(path)
    if path.endswith('.mat'):
        data = io.loadmat(path)
        # try common keys
        keys = [k for k in data.keys() if not k.startswith('__')]
        if varname and varname in data:
            x = data[varname]
        elif len(keys) == 1:
            x = data[keys[0]]
        else:
            # pick the largest array
            cand = [v for v in data.values() if isinstance(v, np.ndarray)]
            if not cand:
                raise ValueError('No suitable array found in mat file')
            x = max(cand, key=lambda a: a.size)
        
        # Ensure 1D: squeeze and flatten if needed
        x = np.asarray(x).squeeze()
        if x.ndim > 1:
            # If still multi-dimensional, take first row/column that has data
            if x.shape[0] == 1:
                x = x[0, :]
            elif x.shape[1] == 1:
                x = x[:, 0]
            else:
                # Flatten to 1D
                x = x.flatten()
        
        return x.astype(float), 1.0
    elif path.endswith('.npy'):
        x = np.load(path)
        return x.astype(float).squeeze(), 1.0
    elif path.endswith('.csv'):
        x = np.loadtxt(path, delimiter=',')
        return x.astype(float).squeeze(), 1.0
    else:
        raise ValueError('Unsupported filetype for load_signal')

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
    algo = rpt.Pelt(model=model).fit(features)
    # pen may need tuning; expose as parameter
    bkps = algo.predict(pen=pen)
    # ruptures returns 1-based index of last segment end; convert to zero-based positions
    return np.array(bkps[:-1], dtype=int)

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
