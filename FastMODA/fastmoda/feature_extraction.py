"""Feature Extraction for Multi-Modal Signal Analysis

Extracts numerical feature representations from each analysis type
for input into neural networks for automated diagnosis.

Feature Design Philosophy:
- Extract interpretable, clinically-relevant features
- Normalize to comparable scales
- Capture both temporal and frequency domain information
- Include statistical moments and distribution properties
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats, signal
from scipy.stats import entropy


# ==================== Spectral Analysis Features ====================

def extract_spectral_features(freqs: np.ndarray, Sxx: np.ndarray,
                              times: np.ndarray, cps: np.ndarray,
                              bands: List[Tuple] = None) -> Dict[str, float]:
    """
    Extract features from spectral analysis

    Args:
        freqs: Frequency array
        Sxx: Spectrogram (freq x time)
        times: Time array
        cps: Changepoint indices
        bands: Frequency bands for power computation

    Returns:
        Dictionary of spectral features
    """
    features = {}

    # Average spectrum
    avg_spectrum = np.mean(Sxx, axis=1)

    # 1. Dominant frequency
    dominant_freq_idx = np.argmax(avg_spectrum)
    features['dominant_frequency'] = freqs[dominant_freq_idx]
    features['dominant_power'] = avg_spectrum[dominant_freq_idx]

    # 2. Spectral centroid (center of mass)
    features['spectral_centroid'] = np.sum(freqs * avg_spectrum) / np.sum(avg_spectrum)

    # 3. Spectral spread (bandwidth)
    features['spectral_spread'] = np.sqrt(
        np.sum(((freqs - features['spectral_centroid']) ** 2) * avg_spectrum) /
        np.sum(avg_spectrum)
    )

    # 4. Spectral entropy (complexity measure)
    normalized_spectrum = avg_spectrum / np.sum(avg_spectrum)
    features['spectral_entropy'] = entropy(normalized_spectrum)

    # 5. Spectral flatness (tonality vs noise)
    geometric_mean = np.exp(np.mean(np.log(avg_spectrum + 1e-10)))
    arithmetic_mean = np.mean(avg_spectrum)
    features['spectral_flatness'] = geometric_mean / (arithmetic_mean + 1e-10)

    # 6. Peak prominence
    peaks, properties = signal.find_peaks(avg_spectrum, prominence=0.1*np.max(avg_spectrum))
    features['n_spectral_peaks'] = len(peaks)
    if len(peaks) > 0:
        features['max_peak_prominence'] = np.max(properties['prominences'])
        features['mean_peak_prominence'] = np.mean(properties['prominences'])
    else:
        features['max_peak_prominence'] = 0
        features['mean_peak_prominence'] = 0

    # 7. Band power features
    if bands is not None:
        for low, high, name in bands:
            mask = (freqs >= low) & (freqs <= high)
            band_power = np.mean(Sxx[mask, :])
            features[f'{name}_power'] = band_power

    # 8. Temporal features
    features['n_changepoints'] = len(cps)
    features['changepoint_density'] = len(cps) / (times[-1] - times[0]) if len(times) > 0 else 0

    # 9. Spectral variability over time
    features['spectral_std_time'] = np.mean(np.std(Sxx, axis=1))

    # 10. High/Low frequency ratio
    mid_idx = len(freqs) // 2
    low_power = np.mean(Sxx[:mid_idx, :])
    high_power = np.mean(Sxx[mid_idx:, :])
    features['high_low_ratio'] = high_power / (low_power + 1e-10)

    return features


# ==================== Phase Analysis Features ====================

def extract_phase_features(phase: np.ndarray, amplitude: np.ndarray,
                           inst_freq: np.ndarray, fs: float = 1.0) -> Dict[str, float]:
    """
    Extract features from phase analysis

    Args:
        phase: Instantaneous phase
        amplitude: Instantaneous amplitude
        inst_freq: Instantaneous frequency
        fs: Sampling frequency

    Returns:
        Dictionary of phase features
    """
    features = {}

    # 1. Frequency statistics
    features['mean_inst_frequency'] = np.mean(inst_freq)
    features['std_inst_frequency'] = np.std(inst_freq)
    features['median_inst_frequency'] = np.median(inst_freq)
    features['iqr_inst_frequency'] = np.percentile(inst_freq, 75) - np.percentile(inst_freq, 25)

    # 2. Amplitude statistics
    features['mean_inst_amplitude'] = np.mean(amplitude)
    features['std_inst_amplitude'] = np.std(amplitude)
    features['cv_inst_amplitude'] = np.std(amplitude) / (np.mean(amplitude) + 1e-10)  # Coefficient of variation

    # 3. Phase coherence (Kuramoto order parameter)
    phase_coherence = np.abs(np.mean(np.exp(1j * phase)))
    features['phase_coherence'] = phase_coherence

    # 4. Phase concentration (circular variance)
    features['phase_concentration'] = 1 - np.abs(np.mean(np.exp(1j * phase)))

    # 5. Frequency modulation index
    freq_derivative = np.diff(inst_freq) * fs
    features['freq_modulation_index'] = np.std(freq_derivative)

    # 6. Amplitude modulation index
    amp_derivative = np.diff(amplitude) * fs
    features['amp_modulation_index'] = np.std(amp_derivative)

    # 7. Phase-amplitude coupling strength
    # Correlation between instantaneous amplitude and frequency
    if len(inst_freq) == len(amplitude):
        features['phase_amp_coupling'] = np.abs(np.corrcoef(inst_freq, amplitude)[0, 1])
    else:
        features['phase_amp_coupling'] = 0

    # 8. Frequency range
    features['freq_range'] = np.max(inst_freq) - np.min(inst_freq)

    # 9. Amplitude range
    features['amp_range'] = np.max(amplitude) - np.min(amplitude)

    # 10. Phase entropy
    phase_hist, _ = np.histogram(phase, bins=50, density=True)
    features['phase_entropy'] = entropy(phase_hist + 1e-10)

    return features


# ==================== STFT Features ====================

def extract_stft_features(freqs: np.ndarray, times: np.ndarray,
                          Sxx: np.ndarray) -> Dict[str, float]:
    """
    Extract features from STFT

    Args:
        freqs: Frequency array
        times: Time array
        Sxx: STFT magnitude (freq x time)

    Returns:
        Dictionary of STFT features
    """
    features = {}

    # 1. Temporal spectral centroid
    centroids = []
    for t_idx in range(Sxx.shape[1]):
        spectrum = Sxx[:, t_idx]
        centroid = np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-10)
        centroids.append(centroid)
    centroids = np.array(centroids)

    features['mean_temporal_centroid'] = np.mean(centroids)
    features['std_temporal_centroid'] = np.std(centroids)
    features['centroid_trend'] = np.polyfit(times, centroids, 1)[0]  # Linear trend

    # 2. Temporal spectral spread
    spreads = []
    for t_idx in range(Sxx.shape[1]):
        spectrum = Sxx[:, t_idx]
        centroid = centroids[t_idx]
        spread = np.sqrt(np.sum(((freqs - centroid) ** 2) * spectrum) / (np.sum(spectrum) + 1e-10))
        spreads.append(spread)
    spreads = np.array(spreads)

    features['mean_temporal_spread'] = np.mean(spreads)
    features['std_temporal_spread'] = np.std(spreads)

    # 3. Spectral flux (change over time)
    flux = np.sqrt(np.sum(np.diff(Sxx, axis=1) ** 2, axis=0))
    features['mean_spectral_flux'] = np.mean(flux)
    features['std_spectral_flux'] = np.std(flux)

    # 4. Temporal modulation
    freq_modulation = np.std(Sxx, axis=1)  # Variance across time for each frequency
    features['temporal_modulation'] = np.mean(freq_modulation)

    # 5. Time-frequency concentration
    normalized_stft = Sxx / (np.sum(Sxx) + 1e-10)
    features['tf_concentration'] = -np.sum(normalized_stft * np.log(normalized_stft + 1e-10))

    return features


# ==================== Wavelet Features ====================

def extract_wavelet_features(freqs: np.ndarray, times: np.ndarray,
                             cwt_mag: np.ndarray) -> Dict[str, float]:
    """
    Extract features from continuous wavelet transform

    Args:
        freqs: Frequency array (log-spaced)
        times: Time array
        cwt_mag: CWT magnitude (freq x time)

    Returns:
        Dictionary of wavelet features
    """
    features = {}

    # 1. Scale-averaged power
    scale_avg_power = np.mean(cwt_mag, axis=0)
    features['mean_scale_avg_power'] = np.mean(scale_avg_power)
    features['std_scale_avg_power'] = np.std(scale_avg_power)

    # 2. Frequency-averaged power over time
    freq_avg_power = np.mean(cwt_mag, axis=1)
    features['mean_freq_avg_power'] = np.mean(freq_avg_power)
    features['std_freq_avg_power'] = np.std(freq_avg_power)

    # 3. Multi-scale entropy
    for scale_idx in [0, len(freqs)//4, len(freqs)//2, 3*len(freqs)//4, -1]:
        scale_signal = cwt_mag[scale_idx, :]
        scale_hist, _ = np.histogram(scale_signal, bins=20, density=True)
        features[f'scale_{scale_idx}_entropy'] = entropy(scale_hist + 1e-10)

    # 4. Ridge strength (coherent structures)
    # Approximate as max power concentration
    max_power_per_time = np.max(cwt_mag, axis=0)
    total_power_per_time = np.sum(cwt_mag, axis=0)
    ridge_strength = max_power_per_time / (total_power_per_time + 1e-10)
    features['mean_ridge_strength'] = np.mean(ridge_strength)

    # 5. Dominant scale at each time
    dominant_scales = np.argmax(cwt_mag, axis=0)
    features['mean_dominant_scale'] = np.mean(freqs[dominant_scales])
    features['std_dominant_scale'] = np.std(freqs[dominant_scales])

    # 6. Energy distribution across scales
    energy_per_scale = np.sum(cwt_mag, axis=1)
    normalized_energy = energy_per_scale / (np.sum(energy_per_scale) + 1e-10)
    features['scale_energy_entropy'] = entropy(normalized_energy)

    # 7. Low/Mid/High scale energy ratio
    n_scales = len(freqs)
    low_energy = np.sum(cwt_mag[:n_scales//3, :])
    mid_energy = np.sum(cwt_mag[n_scales//3:2*n_scales//3, :])
    high_energy = np.sum(cwt_mag[2*n_scales//3:, :])
    total = low_energy + mid_energy + high_energy + 1e-10

    features['low_scale_energy_ratio'] = low_energy / total
    features['mid_scale_energy_ratio'] = mid_energy / total
    features['high_scale_energy_ratio'] = high_energy / total

    return features


# ==================== Coherence Features ====================

def extract_coherence_features(freqs: np.ndarray, times: np.ndarray,
                               coherence: np.ndarray) -> Dict[str, float]:
    """
    Extract features from coherence analysis

    Args:
        freqs: Frequency array
        times: Time array
        coherence: Coherence values (freq x time)

    Returns:
        Dictionary of coherence features
    """
    features = {}

    # 1. Average coherence
    features['mean_coherence'] = np.mean(coherence)
    features['std_coherence'] = np.std(coherence)
    features['max_coherence'] = np.max(coherence)

    # 2. High coherence fraction
    features['high_coherence_fraction'] = np.mean(coherence > 0.7)

    # 3. Peak coherence location
    max_idx = np.unravel_index(np.argmax(coherence), coherence.shape)
    features['peak_coherence_freq'] = freqs[max_idx[0]]

    # 4. Frequency band coherence
    freq_bands = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 50)]
    for low, high in freq_bands:
        mask = (freqs >= low) & (freqs <= high)
        if np.any(mask):
            band_coherence = np.mean(coherence[mask, :])
            features[f'coherence_{low}_{high}Hz'] = band_coherence

    # 5. Temporal coherence variability
    coherence_over_time = np.mean(coherence, axis=0)
    features['coherence_temporal_std'] = np.std(coherence_over_time)

    return features


# ==================== Bispectrum Features ====================

def extract_bispectrum_features(freqs: np.ndarray, bicoherence: np.ndarray) -> Dict[str, float]:
    """
    Extract features from bispectrum analysis

    Args:
        freqs: Frequency array
        bicoherence: Bicoherence matrix (freq x freq)

    Returns:
        Dictionary of bispectrum features
    """
    features = {}

    # 1. Maximum bicoherence
    features['max_bicoherence'] = np.max(bicoherence)
    features['mean_bicoherence'] = np.mean(bicoherence)
    features['std_bicoherence'] = np.std(bicoherence)

    # 2. Strong coupling fraction
    features['strong_coupling_fraction'] = np.mean(bicoherence > 0.5)

    # 3. Peak coupling location
    max_idx = np.unravel_index(np.argmax(bicoherence), bicoherence.shape)
    features['peak_coupling_f1'] = freqs[max_idx[0]]
    features['peak_coupling_f2'] = freqs[max_idx[1]]
    features['peak_coupling_f_sum'] = features['peak_coupling_f1'] + features['peak_coupling_f2']

    # 4. Diagonal bicoherence (self-coupling)
    diagonal = np.diag(bicoherence)
    features['mean_diagonal_bicoherence'] = np.mean(diagonal)

    # 5. Off-diagonal bicoherence (cross-coupling)
    mask = np.ones_like(bicoherence, dtype=bool)
    np.fill_diagonal(mask, False)
    features['mean_offdiagonal_bicoherence'] = np.mean(bicoherence[mask])

    # 6. Coupling entropy (how distributed is the coupling)
    normalized_bic = bicoherence / (np.sum(bicoherence) + 1e-10)
    features['coupling_entropy'] = -np.sum(normalized_bic * np.log(normalized_bic + 1e-10))

    return features


# ==================== Combined Feature Extraction ====================

def extract_all_features(analysis_results: Dict) -> Tuple[np.ndarray, List[str]]:
    """
    Extract features from all analysis types and concatenate

    Args:
        analysis_results: Dictionary containing results from all analyses
            Keys: 'spectral', 'phase', 'stft', 'wavelet', 'coherence', 'bispectrum'

    Returns:
        feature_vector: 1D numpy array of all features
        feature_names: List of feature names (same order as vector)
    """
    all_features = {}

    # Extract from each analysis if present
    if 'spectral' in analysis_results:
        spectral = analysis_results['spectral']
        spectral_features = extract_spectral_features(
            spectral['freqs'], spectral['spec_data'],
            spectral['times'], spectral['changepoints'],
            spectral.get('bands', None)
        )
        all_features.update({f'spectral_{k}': v for k, v in spectral_features.items()})

    if 'phase' in analysis_results:
        phase = analysis_results['phase']
        phase_features = extract_phase_features(
            phase['phase'], phase['amplitude'],
            phase['inst_freq'], phase.get('fs', 1.0)
        )
        all_features.update({f'phase_{k}': v for k, v in phase_features.items()})

    if 'stft' in analysis_results:
        stft = analysis_results['stft']
        stft_features = extract_stft_features(
            stft['freqs'], stft['times'], stft['Sxx']
        )
        all_features.update({f'stft_{k}': v for k, v in stft_features.items()})

    if 'wavelet' in analysis_results:
        wavelet = analysis_results['wavelet']
        wavelet_features = extract_wavelet_features(
            wavelet['freqs'], wavelet['times'], wavelet['cwt_mag']
        )
        all_features.update({f'wavelet_{k}': v for k, v in wavelet_features.items()})

    if 'coherence' in analysis_results:
        coherence = analysis_results['coherence']
        coherence_features = extract_coherence_features(
            coherence['freqs'], coherence['times'], coherence['coherence']
        )
        all_features.update({f'coherence_{k}': v for k, v in coherence_features.items()})

    if 'bispectrum' in analysis_results:
        bispec = analysis_results['bispectrum']
        bispec_features = extract_bispectrum_features(
            bispec['freqs'], bispec['bicoherence']
        )
        all_features.update({f'bispectrum_{k}': v for k, v in bispec_features.items()})

    # Convert to arrays
    feature_names = sorted(all_features.keys())  # Consistent ordering
    feature_vector = np.array([all_features[name] for name in feature_names])

    # Handle NaN and Inf
    feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1e10, neginf=-1e10)

    return feature_vector, feature_names


def normalize_features(features: np.ndarray, mean: np.ndarray = None,
                       std: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize features using z-score normalization

    Args:
        features: Feature vector or matrix (n_samples x n_features)
        mean: Pre-computed mean (for test data)
        std: Pre-computed std (for test data)

    Returns:
        normalized_features: Z-score normalized features
        mean: Feature means
        std: Feature standard deviations
    """
    if mean is None:
        mean = np.mean(features, axis=0 if features.ndim > 1 else None)
    if std is None:
        std = np.std(features, axis=0 if features.ndim > 1 else None)

    std = np.where(std == 0, 1, std)  # Avoid division by zero
    normalized = (features - mean) / std

    return normalized, mean, std
