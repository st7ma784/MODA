"""Reusable feature-extraction pipeline shared by the recordings, baseline and
classification endpoints.

Promotes the single-signal analysis pattern from
``example_neural_network_usage.py::run_all_analyses`` into a function that can
be called directly with a signal + sampling rate.
"""

import numpy as np

from fastmoda import sliding_fft, compute_band_powers, detect_changepoints
from fastmoda.analysis_gpu import (
    compute_instantaneous_phase_gpu,
    stft_gpu,
    cwt_gpu,
    bispectrum_gpu,
)
from fastmoda.feature_extraction import extract_all_features

DEFAULT_BANDS = [(0.5, 4, 'delta'), (4, 8, 'theta'), (8, 13, 'alpha'),
                 (13, 30, 'beta'), (30, 100, 'gamma')]

# Single-signal modalities only - 'coherence' needs a second channel.
DEFAULT_ANALYSES = ('spectral', 'phase', 'stft', 'wavelet', 'bispectrum')


def compute_all_analyses(x, fs, analyses=None):
    """Run the single-signal analyses consumed by extract_all_features().

    Args:
        x: 1D signal array
        fs: sampling rate (Hz)
        analyses: iterable of modality names to compute, defaults to
            DEFAULT_ANALYSES (spectral, phase, stft, wavelet, bispectrum)

    Returns:
        analysis_results dict suitable for
        fastmoda.feature_extraction.extract_all_features()
    """
    x = np.asarray(x, dtype=np.float64).squeeze()
    fs = float(fs)
    if analyses is None:
        analyses = DEFAULT_ANALYSES

    results = {}

    if 'spectral' in analyses:
        freqs, times, Sxx = sliding_fft(x, fs, win_s=1.0)
        band_feats, _ = compute_band_powers(Sxx, freqs, DEFAULT_BANDS)
        cps = detect_changepoints(band_feats, pen=10)
        results['spectral'] = {
            'freqs': freqs, 'spec_data': Sxx, 'times': times,
            'changepoints': cps, 'bands': DEFAULT_BANDS,
        }

    if 'phase' in analyses:
        phase_data = compute_instantaneous_phase_gpu(x, fs=fs)
        results['phase'] = {
            'phase': phase_data['phase'],
            'amplitude': phase_data['amplitude'],
            'inst_freq': phase_data['frequency'],
            'fs': fs,
        }

    if 'stft' in analyses:
        # hop must be < window <= len(x); clamp so short recordings don't
        # trip scipy's "noverlap must be less than nperseg" check.
        window_size = max(8, min(256, len(x) // 2))
        hop_size = max(1, window_size // 2)
        freqs, times, Sxx = stft_gpu(x, fs=fs, window_size=window_size, hop_size=hop_size)
        results['stft'] = {'freqs': freqs, 'times': times, 'Sxx': Sxx}

    if 'wavelet' in analyses:
        f_max = max(1.0, min(50.0, fs / 2.0 * 0.95))
        f_min = min(0.5, f_max / 2.0)
        freqs, times, cwt_mag = cwt_gpu(x, fs=fs, freq_range=(f_min, f_max), n_freqs=50)
        results['wavelet'] = {'freqs': freqs, 'times': times, 'cwt_mag': cwt_mag}

    if 'bispectrum' in analyses:
        nfft = min(256, len(x))
        bispec = bispectrum_gpu(x, fs=fs, nfft=nfft)
        results['bispectrum'] = {
            'freqs': bispec['frequencies'],
            'bicoherence': bispec['bicoherence'],
            'bispectrum': bispec['bispectrum'],
        }

    return results


def compute_feature_vector(x, fs, analyses=None):
    """Return (feature_vector, feature_names) for a single-channel signal."""
    analysis_results = compute_all_analyses(x, fs, analyses=analyses)
    return extract_all_features(analysis_results)
