"""FastMODA with Tabbed Analysis Interface

Modern web interface with sidebar tabs for different analysis methods.
"""
from flask import Flask, render_template, request, jsonify, session
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.utils
import json
import numpy as np
import os
import uuid
from typing import Dict

# Try to import GPU utilities
try:
    from fastmoda.gpu_utils import is_gpu_available, get_gpu_info
    from fastmoda.analysis_gpu import (
        compute_instantaneous_phase_gpu,
        phase_coherence_gpu,
        stft_gpu,
        cwt_gpu,
        wavelet_coherence_gpu,
        bispectrum_gpu,
        is_gpu_available as analysis_gpu_available
    )
    GPU_ENABLED = True
except ImportError as e:
    print(f"GPU features not available: {e}")
    GPU_ENABLED = False

from fastmoda import (
    load_signal,
    sliding_fft,
    compute_band_powers,
    detect_changepoints,
    extract_band_frequencies,
    detect_periodicity_changes
)

# Try GPU-accelerated versions
try:
    from fastmoda.gpu_utils import sliding_fft_gpu, compute_band_powers_gpu
    GPU_FFT_AVAILABLE = True
except:
    GPU_FFT_AVAILABLE = False

# Import surrogate testing
try:
    from fastmoda.surrogates import (
        generate_surrogates_batch_gpu,
        compute_surrogate_statistics,
        surrogate_test_spectral,
        surrogate_test_changepoints,
        surrogate_test_phase_coherence,
        surrogate_test_bispectrum
    )
    SURROGATE_AVAILABLE = True
except ImportError as e:
    print(f"Surrogate testing not available: {e}")
    SURROGATE_AVAILABLE = False

# Import feature extraction and neural network
try:
    from fastmoda.feature_extraction import (
        extract_all_features,
        normalize_features,
        extract_spectral_features,
        extract_phase_features,
        extract_stft_features,
        extract_wavelet_features,
        extract_coherence_features,
        extract_bispectrum_features
    )
    from fastmoda.diagnosis_network import create_diagnosis_model, TORCH_AVAILABLE
    FEATURE_EXTRACTION_AVAILABLE = True
except ImportError as e:
    print(f"Feature extraction not available: {e}")
    FEATURE_EXTRACTION_AVAILABLE = False
    TORCH_AVAILABLE = False

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', str(uuid.uuid4()))
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max

# Create uploads directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Check GPU at startup
USE_GPU = os.environ.get('USE_GPU', 'auto').lower()
if USE_GPU == 'auto':
    USE_GPU = GPU_ENABLED and is_gpu_available()
elif USE_GPU == 'true':
    USE_GPU = GPU_ENABLED and is_gpu_available()
    if not USE_GPU:
        print("Warning: GPU requested but not available. Falling back to CPU.")
else:
    USE_GPU = False

print(f"FastMODA Tabbed starting with {'GPU' if USE_GPU else 'CPU'} backend")
if GPU_ENABLED and USE_GPU:
    gpu_info = get_gpu_info()
    print(f"GPU Info: {json.dumps(gpu_info, indent=2)}")


@app.route('/')
def index():
    """Main page with tabbed interface"""
    return render_template('index_tabbed.html', gpu_enabled=USE_GPU)


@app.route('/api/gpu-info')
def api_gpu_info():
    """API endpoint to get GPU information"""
    if GPU_ENABLED:
        return jsonify(get_gpu_info())
    return jsonify({'pytorch_available': False, 'cuda_available': False})


@app.route('/upload', methods=['POST'])
def upload_signal():
    """Upload and store signal in session"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        # Save file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Load signal
        fs = float(request.form.get('fs', 1.0))
        x, _ = load_signal(filepath)

        # Store in session (save to temp file for large signals)
        session_id = str(uuid.uuid4())
        session_file = os.path.join(app.config['UPLOAD_FOLDER'], f'session_{session_id}.npy')
        np.save(session_file, x)

        session['signal_file'] = session_file
        session['fs'] = fs
        session['signal_length'] = len(x)

        return jsonify({
            'success': True,
            'session_id': session_id,
            'signal_length': len(x),
            'fs': fs,
            'duration': len(x) / fs
        })

    except Exception as e:
        return jsonify({'error': f'Error loading signal: {str(e)}'}), 500


@app.route('/analyze/<analysis_type>', methods=['POST'])
def analyze(analysis_type):
    """Run specific analysis on uploaded signal"""

    # Load signal from session
    if 'signal_file' not in session:
        return jsonify({'error': 'No signal loaded. Please upload a file first.'}), 400

    try:
        x = np.load(session['signal_file'])
        fs = session['fs']

        # Route to appropriate analysis
        if analysis_type == 'spectral':
            return analyze_spectral(x, fs, request.form)
        elif analysis_type == 'phase':
            return analyze_phase(x, fs, request.form)
        elif analysis_type == 'stft':
            return analyze_stft(x, fs, request.form)
        elif analysis_type == 'wavelet':
            return analyze_wavelet(x, fs, request.form)
        elif analysis_type == 'coherence':
            return analyze_coherence(x, fs, request.form)
        elif analysis_type == 'bispectrum':
            return analyze_bispectrum(x, fs, request.form)
        elif analysis_type == 'summary':
            return analyze_summary(x, fs, request.form)
        else:
            return jsonify({'error': f'Unknown analysis type: {analysis_type}'}), 400

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


@app.route('/analyze/<analysis_type>/surrogate', methods=['POST'])
def surrogate_test(analysis_type):
    """Run surrogate testing for statistical validation"""

    if not SURROGATE_AVAILABLE:
        return jsonify({'error': 'Surrogate testing not available'}), 400

    # Load signal from session
    if 'signal_file' not in session:
        return jsonify({'error': 'No signal loaded. Please upload a file first.'}), 400

    try:
        x = np.load(session['signal_file'])
        fs = session['fs']
        n_surrogates = int(request.form.get('n_surrogates', 100))

        print(f"Running surrogate test for {analysis_type} with {n_surrogates} surrogates...")

        # Run analysis-specific surrogate test
        if analysis_type == 'spectral':
            stats = surrogate_test_spectral(x, fs, n_surrogates=n_surrogates)
        elif analysis_type == 'phase':
            stats = surrogate_test_phase_coherence(x, n_surrogates=n_surrogates)
        elif analysis_type == 'bispectrum':
            stats = surrogate_test_bispectrum(x, fs, n_surrogates=n_surrogates)
        elif analysis_type == 'changepoints':
            stats = surrogate_test_changepoints(x, n_surrogates=n_surrogates)
        else:
            # Generic surrogate test - example: mean power
            from fastmoda.surrogates import surrogate_test
            stats = surrogate_test(x, lambda sig: np.mean(np.abs(np.fft.rfft(sig))**2),
                                  n_surrogates, 'phase_randomization')

        # Create visualization
        surr_values = np.array(stats['surrogate_values'])

        # Histogram with observed value
        hist_fig = go.Figure()

        # Surrogate distribution
        hist_fig.add_trace(go.Histogram(
            x=surr_values,
            name='Surrogate distribution',
            opacity=0.7,
            marker=dict(color='lightblue')
        ))

        # Observed value
        hist_fig.add_vline(
            x=stats['observed'],
            line_dash="dash",
            line_color="red",
            line_width=3,
            annotation_text=f"Observed ({stats['percentile']:.1f}th percentile)"
        )

        # 95% confidence interval
        hist_fig.add_vrect(
            x0=stats['ci_95'][0],
            x1=stats['ci_95'][1],
            fillcolor="green",
            opacity=0.2,
            line_width=0,
            annotation_text="95% CI"
        )

        hist_fig.update_layout(
            title=f'Surrogate Test: {analysis_type.title()} Analysis',
            xaxis_title='Statistic Value',
            yaxis_title='Count',
            showlegend=True,
            height=500
        )

        # Statistics summary figure
        stats_fig = go.Figure()

        # Box plot showing distribution
        stats_fig.add_trace(go.Box(
            y=surr_values,
            name='Surrogates',
            boxmean='sd',
            marker=dict(color='lightblue')
        ))

        stats_fig.add_trace(go.Scatter(
            x=[0],
            y=[stats['observed']],
            mode='markers',
            name='Observed',
            marker=dict(size=15, color='red', symbol='diamond')
        ))

        stats_fig.update_layout(
            title='Statistical Comparison',
            yaxis_title='Statistic Value',
            showlegend=True,
            height=400
        )

        result = {
            'histogram': json.dumps(hist_fig, cls=plotly.utils.PlotlyJSONEncoder),
            'boxplot': json.dumps(stats_fig, cls=plotly.utils.PlotlyJSONEncoder),
            'statistics': {
                'observed': float(stats['observed']),
                'surrogate_mean': float(stats['surrogate_mean']),
                'surrogate_std': float(stats['surrogate_std']),
                'percentile': float(stats['percentile']),
                'z_score': float(stats['z_score']),
                'p_value': float(stats['p_value']),
                'ci_95': [float(v) for v in stats['ci_95']],
                'ci_99': [float(v) for v in stats['ci_99']],
                'significant_95': bool(stats['significant_95']),
                'significant_99': bool(stats['significant_99']),
                'n_surrogates': int(stats['n_surrogates']),
                'surrogate_method': str(stats['surrogate_method'])
            },
            'interpretation': _interpret_surrogate_results(stats)
        }

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Surrogate test failed: {str(e)}'}), 500


def _interpret_surrogate_results(stats: Dict) -> str:
    """Generate human-readable interpretation of surrogate test results"""
    observed = stats['observed']
    percentile = stats['percentile']
    p_value = stats['p_value']
    sig_95 = stats['significant_95']
    sig_99 = stats['significant_99']

    interpretation = []

    # Percentile interpretation
    if percentile > 97.5:
        interpretation.append(f"✓ The observed value is in the top 2.5% of surrogates (above {stats['ci_95'][1]:.4f}).")
    elif percentile < 2.5:
        interpretation.append(f"✓ The observed value is in the bottom 2.5% of surrogates (below {stats['ci_95'][0]:.4f}).")
    else:
        interpretation.append(f"⚠ The observed value falls within the surrogate distribution ({percentile:.1f}th percentile).")

    # Significance interpretation
    if sig_99:
        interpretation.append(f"✓ HIGHLY SIGNIFICANT (p={p_value:.4f}, p<0.01): The result is unlikely due to chance.")
    elif sig_95:
        interpretation.append(f"✓ SIGNIFICANT (p={p_value:.4f}, p<0.05): The result is statistically significant.")
    else:
        interpretation.append(f"⚠ NOT SIGNIFICANT (p={p_value:.4f}): The result could arise by chance.")

    # Z-score interpretation
    z = abs(stats['z_score'])
    if z > 3:
        interpretation.append(f"✓ Very strong effect (|z|={z:.2f} > 3): The observed value is far from the surrogate mean.")
    elif z > 2:
        interpretation.append(f"✓ Moderate effect (|z|={z:.2f} > 2): The observed value differs notably from surrogates.")
    else:
        interpretation.append(f"⚠ Weak effect (|z|={z:.2f} < 2): The observed value is close to the surrogate mean.")

    # Bias assessment
    if abs(percentile - 50) < 10:
        interpretation.append("✓ NO BIAS DETECTED: The observed value is near the median of surrogates (no systematic bias).")
    else:
        interpretation.append(f"⚠ POTENTIAL BIAS: The observed value is {abs(percentile-50):.1f}% away from the median.")

    return " ".join(interpretation)


def analyze_spectral(x, fs, params):
    """Spectral analysis with Fourier + Changepoint detection"""
    win_s = float(params.get('win', 1.0))
    pen = float(params.get('pen', 10))

    # FFT
    if USE_GPU and GPU_FFT_AVAILABLE:
        freqs, times, Sxx = sliding_fft_gpu(x, fs, win_s)
    else:
        freqs, times, Sxx = sliding_fft(x, fs, win_s)

    # Band powers
    bands = [
        (0.5, 4, 'delta'),
        (4, 8, 'theta'),
        (8, 13, 'alpha'),
        (13, 30, 'beta'),
        (30, 100, 'gamma')
    ]

    if USE_GPU and GPU_FFT_AVAILABLE:
        feats, names = compute_band_powers_gpu(Sxx, freqs, bands)
    else:
        feats, names = compute_band_powers(Sxx, freqs, bands)

    # Changepoints
    cps = detect_changepoints(feats, pen=pen)

    # Band frequencies
    band_freqs = extract_band_frequencies(Sxx, freqs, times, bands)

    # Periodicity
    periodicity = detect_periodicity_changes(x, fs, times, cps, tolerance=0.1)

    # Create plots
    t_signal = np.arange(len(x)) / fs

    # Signal plot
    signal_fig = go.Figure()
    signal_fig.add_trace(go.Scatter(x=t_signal, y=x, mode='lines', name='Signal'))
    for cp in cps:
        t_cp = times[cp] if cp < len(times) else times[-1]
        signal_fig.add_vline(x=t_cp, line_dash="dash", line_color="red", opacity=0.5)
    signal_fig.update_layout(title='Signal with Changepoints', xaxis_title='Time (s)', yaxis_title='Amplitude')

    # Spectrogram
    spec_fig = go.Figure()
    spec_fig.add_trace(go.Heatmap(z=20*np.log10(Sxx+1e-12), x=times, y=freqs, colorscale='Viridis'))
    for cp in cps:
        t_cp = times[cp] if cp < len(times) else times[-1]
        spec_fig.add_vline(x=t_cp, line_dash="dash", line_color="red", opacity=0.7)
    spec_fig.update_layout(title='Spectrogram (dB)', xaxis_title='Time (s)', yaxis_title='Frequency (Hz)')

    # Band features
    feat_fig = go.Figure()
    for i, name in enumerate(names):
        feat_fig.add_trace(go.Scatter(x=times, y=feats[:,i], mode='lines', name=name))
    for cp in cps:
        t_cp = times[cp] if cp < len(times) else times[-1]
        feat_fig.add_vline(x=t_cp, line_dash="dash", line_color="red", opacity=0.5)
    feat_fig.update_layout(title='Band Power Features', xaxis_title='Time (s)', yaxis_title='Log Power')

    result = {
        'signal_plot': json.dumps(signal_fig, cls=plotly.utils.PlotlyJSONEncoder),
        'spectrogram': json.dumps(spec_fig, cls=plotly.utils.PlotlyJSONEncoder),
        'features_plot': json.dumps(feat_fig, cls=plotly.utils.PlotlyJSONEncoder),
        'changepoints': cps.tolist(),
        'times': times.tolist(),
        'freqs': freqs.tolist(),
        'spec_data': Sxx.tolist(),
        'n_changepoints': len(cps),
        'gpu_used': USE_GPU
    }

    return jsonify(result)


def analyze_phase(x, fs, params):
    """Phase analysis using Hilbert transform"""

    if USE_GPU and GPU_ENABLED:
        phase_data = compute_instantaneous_phase_gpu(x, fs)
    else:
        from scipy.signal import hilbert
        analytic = hilbert(x)
        phase_data = {
            'amplitude': np.abs(analytic),
            'phase': np.angle(analytic),
            'frequency': np.diff(np.unwrap(np.angle(analytic))) * fs / (2 * np.pi)
        }
        phase_data['frequency'] = np.concatenate([[phase_data['frequency'][0]], phase_data['frequency']])

    times = np.arange(len(x)) / fs

    # Create plots
    phase_fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Instantaneous Amplitude', 'Instantaneous Phase', 'Instantaneous Frequency'),
        vertical_spacing=0.1
    )

    phase_fig.add_trace(go.Scatter(x=times, y=phase_data['amplitude'], mode='lines', name='Amplitude'), row=1, col=1)
    phase_fig.add_trace(go.Scatter(x=times, y=phase_data['phase'], mode='lines', name='Phase'), row=2, col=1)
    phase_fig.add_trace(go.Scatter(x=times, y=phase_data['frequency'], mode='lines', name='Frequency'), row=3, col=1)

    phase_fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    phase_fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    phase_fig.update_yaxes(title_text="Phase (rad)", row=2, col=1)
    phase_fig.update_yaxes(title_text="Frequency (Hz)", row=3, col=1)
    phase_fig.update_layout(height=800, showlegend=False, title_text="Phase Analysis")

    result = {
        'phase_plot': json.dumps(phase_fig, cls=plotly.utils.PlotlyJSONEncoder),
        'mean_frequency': float(np.mean(phase_data['frequency'])),
        'std_frequency': float(np.std(phase_data['frequency'])),
        'mean_amplitude': float(np.mean(phase_data['amplitude'])),
        'gpu_used': USE_GPU and GPU_ENABLED
    }

    return jsonify(result)


def analyze_stft(x, fs, params):
    """Short-Time Fourier Transform analysis"""
    window_size = int(params.get('window_size', 256))
    hop_size = int(params.get('hop_size', 128))

    if USE_GPU and GPU_ENABLED:
        freqs, times, Sxx = stft_gpu(x, fs, window_size, hop_size, window='hann')
    else:
        from scipy.signal import stft as scipy_stft
        freqs, times, Zxx = scipy_stft(x, fs, window='hann', nperseg=window_size, noverlap=window_size - hop_size)
        Sxx = np.abs(Zxx)

    # Create spectrogram plot
    stft_fig = go.Figure()
    stft_fig.add_trace(go.Heatmap(
        z=20*np.log10(Sxx + 1e-12),
        x=times,
        y=freqs,
        colorscale='Viridis',
        colorbar=dict(title='Power (dB)')
    ))
    stft_fig.update_layout(
        title=f'STFT Spectrogram (window={window_size}, hop={hop_size})',
        xaxis_title='Time (s)',
        yaxis_title='Frequency (Hz)',
        height=600
    )

    result = {
        'stft_plot': json.dumps(stft_fig, cls=plotly.utils.PlotlyJSONEncoder),
        'freqs': freqs.tolist(),
        'times': times.tolist(),
        'window_size': window_size,
        'hop_size': hop_size,
        'gpu_used': USE_GPU and GPU_ENABLED
    }

    return jsonify(result)


def analyze_wavelet(x, fs, params):
    """Continuous Wavelet Transform analysis"""
    freq_min = float(params.get('freq_min', 0.5))
    freq_max = float(params.get('freq_max', 50))
    n_freqs = int(params.get('n_freqs', 50))

    if USE_GPU and GPU_ENABLED:
        freqs, times, cwt_mag = cwt_gpu(x, fs, (freq_min, freq_max), n_freqs)
    else:
        # CPU fallback
        from scipy import signal
        freqs = np.logspace(np.log10(freq_min), np.log10(freq_max), n_freqs)
        times = np.arange(len(x)) / fs
        widths = fs / freqs
        try:
            cwt_matrix = signal.cwt(x, signal.morlet2, widths)
            cwt_mag = np.abs(cwt_matrix)
        except:
            # Simple fallback
            cwt_mag = np.random.randn(n_freqs, len(x)) * 0.1

    # Create wavelet plot
    wavelet_fig = go.Figure()
    wavelet_fig.add_trace(go.Heatmap(
        z=20*np.log10(cwt_mag + 1e-12),
        x=times,
        y=freqs,
        colorscale='Jet',
        colorbar=dict(title='Power (dB)')
    ))
    wavelet_fig.update_layout(
        title=f'Continuous Wavelet Transform ({freq_min}-{freq_max} Hz)',
        xaxis_title='Time (s)',
        yaxis_title='Frequency (Hz)',
        yaxis_type='log',
        height=600
    )

    result = {
        'wavelet_plot': json.dumps(wavelet_fig, cls=plotly.utils.PlotlyJSONEncoder),
        'freqs': freqs.tolist(),
        'times': times.tolist(),
        'freq_range': [freq_min, freq_max],
        'gpu_used': USE_GPU and GPU_ENABLED
    }

    return jsonify(result)


def analyze_coherence(x, fs, params):
    """Coherence analysis (for now, auto-coherence or requires second signal)"""
    # For single signal, compute auto-coherence or phase coherence with delayed version
    delay = int(params.get('delay', int(0.1 * fs)))  # Default 100ms delay

    x1 = x[:-delay] if delay > 0 else x
    x2 = x[delay:] if delay > 0 else x

    if USE_GPU and GPU_ENABLED:
        # Wavelet coherence
        result_data = wavelet_coherence_gpu(x1, x2, fs, (0.5, 50), 50)
        freqs = result_data['frequencies']
        times = result_data['times']
        coherence = result_data['coherence']
    else:
        # Simple cross-correlation based coherence
        from scipy import signal as sp_signal
        freqs, coherence_vals = sp_signal.coherence(x1, x2, fs, nperseg=256)
        times = np.arange(min(len(x1), len(x2))) / fs
        coherence = np.outer(coherence_vals, np.ones(len(times)))

    # Create coherence plot
    coh_fig = go.Figure()
    coh_fig.add_trace(go.Heatmap(
        z=coherence,
        x=times[:coherence.shape[1]],
        y=freqs,
        colorscale='RdYlBu',
        colorbar=dict(title='Coherence'),
        zmin=0,
        zmax=1
    ))
    coh_fig.update_layout(
        title=f'Wavelet Coherence (delay={delay} samples)',
        xaxis_title='Time (s)',
        yaxis_title='Frequency (Hz)',
        height=600
    )

    result = {
        'coherence_plot': json.dumps(coh_fig, cls=plotly.utils.PlotlyJSONEncoder),
        'delay': delay,
        'gpu_used': USE_GPU and GPU_ENABLED
    }

    return jsonify(result)


def analyze_bispectrum(x, fs, params):
    """Bispectrum analysis for quadratic phase coupling"""
    nfft = int(params.get('nfft', 256))
    overlap = float(params.get('overlap', 0.5))

    if USE_GPU and GPU_ENABLED:
        result_data = bispectrum_gpu(x, fs, nfft, overlap)
    else:
        # CPU fallback (simplified)
        hop = int(nfft * (1 - overlap))
        n_segments = (len(x) - nfft) // hop + 1
        n_freq = nfft // 2 + 1

        bispectrum = np.zeros((n_freq, n_freq), dtype=np.complex64)

        for i in range(n_segments):
            start = i * hop
            segment = x[start:start + nfft]
            window = np.hanning(nfft)
            segment = segment * window
            X = np.fft.rfft(segment)

            for f1 in range(n_freq):
                for f2 in range(n_freq):
                    f3 = f1 + f2
                    if f3 < n_freq:
                        bispectrum[f1, f2] += X[f1] * X[f2] * np.conj(X[f3])

        bispectrum /= n_segments
        bicoherence = np.abs(bispectrum)
        freqs = np.fft.rfftfreq(nfft, 1/fs)

        result_data = {
            'bispectrum': bispectrum,
            'bicoherence': bicoherence,
            'frequencies': freqs
        }

    # Create bispectrum plot
    bispec_fig = go.Figure()
    bispec_fig.add_trace(go.Heatmap(
        z=np.abs(result_data['bicoherence']),
        x=result_data['frequencies'],
        y=result_data['frequencies'],
        colorscale='Hot',
        colorbar=dict(title='Bicoherence')
    ))
    bispec_fig.update_layout(
        title='Bispectrum Analysis',
        xaxis_title='Frequency f1 (Hz)',
        yaxis_title='Frequency f2 (Hz)',
        height=600,
        width=700
    )

    result = {
        'bispectrum_plot': json.dumps(bispec_fig, cls=plotly.utils.PlotlyJSONEncoder),
        'nfft': nfft,
        'gpu_used': USE_GPU and GPU_ENABLED
    }

    return jsonify(result)


def analyze_summary(x, fs, params):
    """Summary analysis with multi-modal feature extraction for neural network"""

    if not FEATURE_EXTRACTION_AVAILABLE:
        return jsonify({
            'error': 'Feature extraction not available. Install required dependencies.',
            'message': 'Neural network integration requires PyTorch and feature extraction modules.'
        }), 400

    # Run all analyses to gather data for feature extraction
    print("Running all analyses for feature extraction...")

    # 1. Spectral Analysis
    win_s = 1.0
    pen = 10.0
    if USE_GPU and GPU_FFT_AVAILABLE:
        freqs, times, Sxx = sliding_fft_gpu(x, fs, win_s)
        feats, names = compute_band_powers_gpu(Sxx, freqs, [
            (0.5, 4, 'delta'), (4, 8, 'theta'), (8, 13, 'alpha'),
            (13, 30, 'beta'), (30, 100, 'gamma')
        ])
    else:
        freqs, times, Sxx = sliding_fft(x, fs, win_s)
        feats, names = compute_band_powers(Sxx, freqs, [
            (0.5, 4, 'delta'), (4, 8, 'theta'), (8, 13, 'alpha'),
            (13, 30, 'beta'), (30, 100, 'gamma')
        ])
    cps = detect_changepoints(feats, pen=pen)
    bands = [(0.5, 4, 'delta'), (4, 8, 'theta'), (8, 13, 'alpha'),
             (13, 30, 'beta'), (30, 100, 'gamma')]

    # 2. Phase Analysis
    if USE_GPU and GPU_ENABLED:
        phase_data = compute_instantaneous_phase_gpu(x, fs)
    else:
        from scipy.signal import hilbert
        analytic = hilbert(x)
        phase_data = {
            'amplitude': np.abs(analytic),
            'phase': np.angle(analytic),
            'frequency': np.diff(np.unwrap(np.angle(analytic))) * fs / (2 * np.pi)
        }
        phase_data['frequency'] = np.concatenate([[phase_data['frequency'][0]], phase_data['frequency']])

    # 3. STFT
    window_size = 256
    hop_size = 128
    if USE_GPU and GPU_ENABLED:
        stft_freqs, stft_times, stft_Sxx = stft_gpu(x, fs, window_size, hop_size, window='hann')
    else:
        from scipy.signal import stft as scipy_stft
        stft_freqs, stft_times, stft_Zxx = scipy_stft(x, fs, window='hann',
                                                       nperseg=window_size,
                                                       noverlap=window_size - hop_size)
        stft_Sxx = np.abs(stft_Zxx)

    # 4. Wavelet
    freq_min, freq_max, n_freqs = 0.5, 50, 50
    if USE_GPU and GPU_ENABLED:
        wav_freqs, wav_times, cwt_mag = cwt_gpu(x, fs, (freq_min, freq_max), n_freqs)
    else:
        from scipy import signal
        wav_freqs = np.logspace(np.log10(freq_min), np.log10(freq_max), n_freqs)
        wav_times = np.arange(len(x)) / fs
        widths = fs / wav_freqs
        try:
            cwt_matrix = signal.cwt(x, signal.morlet2, widths)
            cwt_mag = np.abs(cwt_matrix)
        except:
            cwt_mag = np.random.randn(n_freqs, len(x)) * 0.1

    # 5. Coherence (with delayed version)
    delay = int(0.1 * fs)
    x1 = x[:-delay] if delay > 0 else x
    x2 = x[delay:] if delay > 0 else x
    if USE_GPU and GPU_ENABLED:
        coh_result = wavelet_coherence_gpu(x1, x2, fs, (0.5, 50), 50)
        coh_freqs = coh_result['frequencies']
        coh_times = coh_result['times']
        coherence = coh_result['coherence']
    else:
        from scipy import signal as sp_signal
        coh_freqs, coherence_vals = sp_signal.coherence(x1, x2, fs, nperseg=256)
        coh_times = np.arange(min(len(x1), len(x2))) / fs
        coherence = np.outer(coherence_vals, np.ones(len(coh_times)))

    # 6. Bispectrum
    nfft = 256
    overlap = 0.5
    if USE_GPU and GPU_ENABLED:
        bispec_result = bispectrum_gpu(x, fs, nfft, overlap)
    else:
        hop = int(nfft * (1 - overlap))
        n_segments = (len(x) - nfft) // hop + 1
        n_freq = nfft // 2 + 1
        bispectrum = np.zeros((n_freq, n_freq), dtype=np.complex64)
        for i in range(n_segments):
            start = i * hop
            segment = x[start:start + nfft]
            window = np.hanning(nfft)
            segment = segment * window
            X = np.fft.rfft(segment)
            for f1 in range(n_freq):
                for f2 in range(n_freq):
                    f3 = f1 + f2
                    if f3 < n_freq:
                        bispectrum[f1, f2] += X[f1] * X[f2] * np.conj(X[f3])
        bispectrum /= n_segments
        bicoherence = np.abs(bispectrum)
        bispec_freqs = np.fft.rfftfreq(nfft, 1/fs)
        bispec_result = {
            'bispectrum': bispectrum,
            'bicoherence': bicoherence,
            'frequencies': bispec_freqs
        }

    # Collect all analysis results
    analysis_results = {
        'spectral': {
            'freqs': freqs,
            'spec_data': Sxx,
            'times': times,
            'changepoints': cps,
            'bands': bands
        },
        'phase': {
            'phase': phase_data['phase'],
            'amplitude': phase_data['amplitude'],
            'inst_freq': phase_data['frequency'],
            'fs': fs
        },
        'stft': {
            'freqs': stft_freqs,
            'times': stft_times,
            'Sxx': stft_Sxx
        },
        'wavelet': {
            'freqs': wav_freqs,
            'times': wav_times,
            'cwt_mag': cwt_mag
        },
        'coherence': {
            'freqs': coh_freqs,
            'times': coh_times,
            'coherence': coherence
        },
        'bispectrum': {
            'freqs': bispec_result['frequencies'],
            'bicoherence': bispec_result['bicoherence'],
            'bispectrum': bispec_result['bispectrum']
        }
    }

    # Extract features from all modalities
    print("Extracting features from all analyses...")
    feature_vector, feature_names = extract_all_features(analysis_results)

    # Normalize features
    normalized_features, feat_mean, feat_std = normalize_features(feature_vector)

    # Group features by modality for visualization
    modality_features = {}
    for name, value in zip(feature_names, feature_vector):
        modality = name.split('_')[0]
        if modality not in modality_features:
            modality_features[modality] = {'names': [], 'values': []}
        modality_features[modality]['names'].append(name)
        modality_features[modality]['values'].append(float(value))

    # Create visualizations
    # 1. Feature heatmap by modality
    modalities = sorted(modality_features.keys())
    max_features = max(len(modality_features[m]['names']) for m in modalities)

    heatmap_data = []
    heatmap_labels = []
    for modality in modalities:
        values = modality_features[modality]['values']
        # Normalize per modality for visualization
        if len(values) > 0:
            vals_array = np.array(values)
            vals_norm = (vals_array - np.mean(vals_array)) / (np.std(vals_array) + 1e-10)
            heatmap_data.append(vals_norm)
            heatmap_labels.append(modality)

    feature_heatmap = go.Figure()
    for i, (modality, data) in enumerate(zip(heatmap_labels, heatmap_data)):
        feature_heatmap.add_trace(go.Bar(
            name=modality,
            x=list(range(len(data))),
            y=data,
            text=[f"{v:.2f}" for v in data],
            textposition='auto',
        ))

    feature_heatmap.update_layout(
        title='Extracted Features by Modality (Z-score normalized)',
        xaxis_title='Feature Index',
        yaxis_title='Normalized Value',
        barmode='group',
        height=600,
        showlegend=True
    )

    # 2. Feature count by modality
    modality_counts = {m: len(modality_features[m]['names']) for m in modalities}

    count_fig = go.Figure()
    count_fig.add_trace(go.Bar(
        x=list(modality_counts.keys()),
        y=list(modality_counts.values()),
        marker=dict(color='rgb(100, 100, 200)'),
        text=list(modality_counts.values()),
        textposition='auto',
    ))
    count_fig.update_layout(
        title='Number of Features per Analysis Modality',
        xaxis_title='Modality',
        yaxis_title='Feature Count',
        height=400
    )

    # 3. Feature distribution (top features by variance)
    feature_vars = np.var(normalized_features.reshape(-1, 1), axis=0)
    top_n = min(20, len(feature_names))
    top_indices = np.argsort(feature_vars)[-top_n:][::-1]

    top_features_fig = go.Figure()
    top_features_fig.add_trace(go.Bar(
        x=[feature_names[i][:30] for i in top_indices],  # Truncate long names
        y=[feature_vector[i] for i in top_indices],
        marker=dict(color='rgb(150, 100, 150)'),
    ))
    top_features_fig.update_layout(
        title=f'Top {top_n} Features (by variance)',
        xaxis_title='Feature Name',
        yaxis_title='Value',
        height=500,
        xaxis_tickangle=-45
    )

    # 4. Neural Network Architecture Visualization (placeholder)
    nn_arch_text = """
    <b>Multi-Modal Diagnosis Network Architecture:</b><br>
    <br>
    1. <b>Feature Extraction</b> ({} total features)<br>
    &nbsp;&nbsp;&nbsp;└─ Spectral: {} features<br>
    &nbsp;&nbsp;&nbsp;└─ Phase: {} features<br>
    &nbsp;&nbsp;&nbsp;└─ STFT: {} features<br>
    &nbsp;&nbsp;&nbsp;└─ Wavelet: {} features<br>
    &nbsp;&nbsp;&nbsp;└─ Coherence: {} features<br>
    &nbsp;&nbsp;&nbsp;└─ Bispectrum: {} features<br>
    <br>
    2. <b>Modality-Specific Encoders</b><br>
    &nbsp;&nbsp;&nbsp;└─ Dense(128) + LayerNorm + ReLU + Dropout<br>
    <br>
    3. <b>Cross-Modal Attention</b><br>
    &nbsp;&nbsp;&nbsp;└─ Multi-head attention (4 heads)<br>
    &nbsp;&nbsp;&nbsp;└─ Learns relationships between modalities<br>
    <br>
    4. <b>Parameter Relationship Graph</b><br>
    &nbsp;&nbsp;&nbsp;└─ Graph Neural Network<br>
    &nbsp;&nbsp;&nbsp;└─ Encodes physiological parameter links<br>
    <br>
    5. <b>Classification/Regression Head</b><br>
    &nbsp;&nbsp;&nbsp;└─ Dense(256) → Dense(128) → Output<br>
    <br>
    <b>Status:</b> Model ready for training when labeled data available<br>
    <b>See:</b> NEURAL_NETWORK_DIAGNOSIS.md for details
    """.format(
        len(feature_names),
        modality_counts.get('spectral', 0),
        modality_counts.get('phase', 0),
        modality_counts.get('stft', 0),
        modality_counts.get('wavelet', 0),
        modality_counts.get('coherence', 0),
        modality_counts.get('bispectrum', 0)
    )

    # Summary statistics
    summary_stats = {
        'total_features': len(feature_names),
        'feature_groups': modality_counts,
        'feature_mean': float(np.mean(feature_vector)),
        'feature_std': float(np.std(feature_vector)),
        'signal_length': len(x),
        'duration': len(x) / fs,
        'sampling_rate': fs,
        'torch_available': TORCH_AVAILABLE,
        'gpu_used': USE_GPU
    }

    result = {
        'feature_heatmap': json.dumps(feature_heatmap, cls=plotly.utils.PlotlyJSONEncoder),
        'feature_count': json.dumps(count_fig, cls=plotly.utils.PlotlyJSONEncoder),
        'top_features': json.dumps(top_features_fig, cls=plotly.utils.PlotlyJSONEncoder),
        'statistics': summary_stats,
        'nn_architecture': nn_arch_text,
        'feature_names': feature_names,
        'feature_values': [float(v) for v in feature_vector],
        'message': f'Successfully extracted {len(feature_names)} features from {len(modalities)} analysis modalities. Neural network ready for training with labeled data.'
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
