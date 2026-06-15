"""Fully optimized GPU-enabled Flask application

Key improvements over previous version:
1. Batched GPU FFT (10-50x faster)
2. Changepoint detection on FREQUENCY not power (fewer, better changepoints)
3. Adaptive penalty tuning (auto-adjusts to signal characteristics)
4. Efficient sine fitting with smart segment merging
5. Real-time progress tracking
"""
from flask import Flask, render_template, request, jsonify, Response, stream_with_context, send_file
from queue import Empty
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.utils
import json
import numpy as np
import os
import io
import uuid
from threading import Thread
import time

# Try to import optimized GPU utilities
try:
    from fastmoda.optimized_gpu import (
        batched_sliding_fft_gpu,
        extract_instantaneous_frequency_gpu,
        compute_spectral_centroid_gpu,
        efficient_band_powers_gpu,
        full_optimized_pipeline_gpu,
        TORCH_AVAILABLE
    )
    from fastmoda.optimized import (
        detect_frequency_changepoints,
        adaptive_segment_sine_fitting
    )
    import torch
    GPU_ENABLED = True
except ImportError as e:
    print(f"GPU optimization not available: {e}")
    GPU_ENABLED = False
    TORCH_AVAILABLE = False

from fastmoda import (
    load_signal,
    detect_periodicity_changes,
    extract_band_frequencies
)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.secret_key = 'fastmoda-optimized-key'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

processing_status = {}

# GPU configuration
USE_GPU = os.environ.get('USE_GPU', 'auto').lower()
if USE_GPU == 'auto':
    USE_GPU = GPU_ENABLED and TORCH_AVAILABLE and torch.cuda.is_available() if TORCH_AVAILABLE else False
elif USE_GPU == 'true':
    USE_GPU = GPU_ENABLED and TORCH_AVAILABLE and torch.cuda.is_available() if TORCH_AVAILABLE else False
    if not USE_GPU:
        print("Warning: GPU requested but not available. Falling back to CPU.")
else:
    USE_GPU = False

DEVICE = torch.device('cuda' if USE_GPU else 'cpu') if TORCH_AVAILABLE else None

print(f"\n{'='*60}")
print(f"FastMODA OPTIMIZED - Starting")
print(f"Backend: {'GPU (OPTIMIZED)' if USE_GPU else 'CPU'}")
if USE_GPU and TORCH_AVAILABLE:
    try:
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    except Exception as e:
        print(f"Warning: Could not query GPU info: {e}")
print(f"{'='*60}\n")

@app.route('/')
def index():
    return render_template('index_optimized.html', gpu_enabled=USE_GPU)

@app.route('/tfa')
def tfa():
    """Time-Frequency Analysis page (CWT / WFT / STFT)"""
    return render_template('tfa.html', gpu_enabled=USE_GPU)

@app.route('/modwt')
def modwt():
    """MODWT wavelet transform analysis page"""
    return render_template('modwt.html', gpu_enabled=USE_GPU)

@app.route('/coherence')
def coherence():
    """Coherence analysis page (GPU-accelerated when available, CPU fallback otherwise)"""
    return render_template('coherence.html', gpu_enabled=USE_GPU)

@app.route('/bispectrum')
def bispectrum():
    """Bispectrum analysis page (GPU-accelerated when available, CPU fallback otherwise)"""
    return render_template('bispectrum.html', gpu_enabled=USE_GPU)

@app.route('/bayesian')
def bayesian():
    """Bayesian inference page (GPU-accelerated when available, CPU fallback otherwise)"""
    return render_template('bayesian.html', gpu_enabled=USE_GPU)

@app.route('/tests')
def tests_page():
    """All-endpoints interactive test harness."""
    return render_template('tests.html', gpu_enabled=USE_GPU)


@app.route('/test_signal.npy')
def test_signal():
    """
    Generate a synthetic multi-component test signal as a .npy file.
    Query params:
      fs      sample rate (default 256)
      n       number of samples (default 2048)
      preset  resting|active|drowsy|sleep|noise|chirp (default resting)
      seed    optional integer seed
    """
    fs      = float(request.args.get('fs', 256))
    n       = int(request.args.get('n', 2048))
    preset  = request.args.get('preset', 'resting')
    seed    = request.args.get('seed')
    if seed is not None:
        np.random.seed(int(seed))

    presets = {
        'resting': dict(alpha=1.0, theta=0.3, beta=0.12, delta=0.10, gamma=0.05, noise=0.20),
        'active':  dict(alpha=0.3, theta=0.2, beta=0.80, delta=0.05, gamma=0.20, noise=0.30),
        'drowsy':  dict(alpha=0.5, theta=0.9, beta=0.05, delta=0.30, gamma=0.02, noise=0.15),
        'sleep':   dict(alpha=0.1, theta=0.3, beta=0.04, delta=1.20, gamma=0.02, noise=0.10),
        'noise':   dict(alpha=0.1, theta=0.1, beta=0.10, delta=0.10, gamma=0.10, noise=1.50),
    }
    t = np.arange(n) / fs
    if preset == 'chirp':
        # Linear frequency sweep 1 → fs/4 Hz over the signal length
        f1 = fs / 4.0
        phase = 2 * np.pi * (1.0 * t + (f1 - 1.0) / (2 * (n / fs)) * t * t)
        x = np.sin(phase) + 0.05 * np.random.randn(n)
    else:
        p = presets.get(preset, presets['resting'])
        x = (p['alpha'] * np.sin(2 * np.pi * 10.0 * t)
             + p['theta'] * np.sin(2 * np.pi *  6.0 * t)
             + p['beta']  * np.sin(2 * np.pi * 18.0 * t)
             + p['delta'] * np.sin(2 * np.pi *  2.0 * t)
             + p['gamma'] * np.sin(2 * np.pi * 40.0 * t)
             + p['noise'] * np.random.randn(n))
    buf = io.BytesIO()
    np.save(buf, x.astype(np.float32))
    buf.seek(0)
    return send_file(buf, mimetype='application/octet-stream',
                     as_attachment=True, download_name='test_signal.npy')


@app.route('/health')
def health():
    return jsonify({'status': 'ok'}), 200

@app.route('/api/gpu-info')
def api_gpu_info():
    """API endpoint to get GPU information"""
    if GPU_ENABLED and TORCH_AVAILABLE:
        return jsonify({
            'pytorch_available': True,
            'cuda_available': torch.cuda.is_available(),
            'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'optimized': True
        })
    return jsonify({'pytorch_available': False, 'cuda_available': False, 'optimized': False})

@app.route('/analyze', methods=['POST'])
def analyze():
    """Initial analysis - returns signal plot immediately"""
    if 'file' not in request.files or not request.files['file'].filename:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    try:
        fs = float(request.form.get('fs', 1.0))
        win_s = float(request.form.get('win', 1.0))
        pen = request.form.get('pen', 'auto')
        if pen != 'auto':
            pen = float(pen)

        # Surrogate testing parameters
        enable_surrogates = request.form.get('enable_surrogates') == 'on'
        n_surrogates = int(request.form.get('n_surrogates', 19)) if enable_surrogates else 0
        surrogate_method = request.form.get('surrogate_method', 'iaaft') if enable_surrogates else None
        alpha = float(request.form.get('alpha', 0.05)) if enable_surrogates else 0.05

        # Load signal
        x, _ = load_signal(filepath)
        print(f"\n{'='*60}")
        print(f"NEW ANALYSIS REQUEST")
        print(f"Signal: {len(x)} samples, {len(x)/fs:.2f} seconds")
        print(f"Window: {win_s}s, Penalty: {pen}")
        if enable_surrogates:
            print(f"Surrogate Testing: {n_surrogates} {surrogate_method.upper()} surrogates, α={alpha}")
        print(f"{'='*60}")

        # Create task
        task_id = str(uuid.uuid4())

        processing_status[task_id] = {
            'status': 'processing',
            'progress': 10,
            'stage': 'Loading signal...',
            'signal_shape': x.shape,
            'fs': fs,
            'filepath': filepath,
            'surrogate_testing': enable_surrogates
        }

        # Generate signal plot
        t_signal = np.arange(len(x)) / fs
        signal_fig = go.Figure()
        signal_fig.add_trace(go.Scatter(
            x=t_signal, y=x,
            mode='lines',
            name='Signal',
            line={'color': 'blue', 'width': 1}
        ))
        signal_fig.update_layout(
            title='Original Signal (Analysis in progress...)',
            xaxis_title='Time (s)',
            yaxis_title='Amplitude',
            hovermode='x unified',
            height=400
        )

        # Start background processing
        thread = Thread(target=optimized_background_analysis,
                       args=(task_id, filepath, fs, win_s, pen, x, enable_surrogates,
                             n_surrogates, surrogate_method, alpha))
        thread.daemon = True
        thread.start()

        return jsonify({
            'task_id': task_id,
            'signal_plot': json.dumps(signal_fig, cls=plotly.utils.PlotlyJSONEncoder),
            'signal_length': len(x),
            'sampling_rate': fs,
            'duration': len(x) / fs,
            'optimized': USE_GPU,
            'surrogate_testing': enable_surrogates
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f"Error: {str(e)}"}), 500

@app.route('/status/<task_id>')
def get_status(task_id):
    """Get processing status"""
    if task_id not in processing_status:
        return jsonify({'error': 'Task not found'}), 404
    return jsonify(processing_status[task_id])


@app.route('/find_changepoints', methods=['POST'])
def find_changepoints_endpoint():
    """Sweep window sizes to find optimal window (periodicity) then detect changepoints."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'JSON body required'}), 400

    task_id = data.get('task_id')
    target_freqs = data.get('target_freqs', [])

    task = processing_status.get(task_id)
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    if task.get('status') != 'complete':
        return jsonify({'error': 'Original analysis not yet complete'}), 400

    filepath = task.get('filepath')
    fs = task.get('fs')
    if not filepath or not fs:
        return jsonify({'error': 'Task missing filepath or fs'}), 400

    try:
        x, _ = load_signal(filepath)
    except Exception as e:
        return jsonify({'error': f'Could not reload signal: {e}'}), 500

    sweep_id = str(uuid.uuid4())
    processing_status[sweep_id] = {
        'status': 'processing',
        'progress': 0,
        'stage': 'Initialising sweep...'
    }

    thread = Thread(target=sweep_background_analysis,
                    args=(sweep_id, x, float(fs), target_freqs))
    thread.daemon = True
    thread.start()

    return jsonify({'sweep_id': sweep_id})

def optimized_background_analysis(task_id, filepath, fs, win_s, pen, x,
                                   enable_surrogates=False, n_surrogates=19,
                                   surrogate_method='iaaft', alpha=0.05):
    """Optimized background analysis using new pipeline with optional surrogate testing"""
    try:
        # Define bands
        bands = [
            (0.5, 4, 'delta'),
            (4, 8, 'theta'),
            (8, 13, 'alpha'),
            (13, 30, 'beta'),
            (30, 100, 'gamma')
        ]
        
        processing_status[task_id].update({
            'progress': 15,
            'stage': 'Initializing optimized pipeline...'
        })
        
        # OPTIMIZED PIPELINE - All in one efficient call
        if USE_GPU:
            processing_status[task_id].update({
                'progress': 20,
                'stage': 'Running GPU-accelerated analysis...'
            })
            
            results = full_optimized_pipeline_gpu(
                x, fs=fs, win_s=win_s, bands=bands, pen=pen, device=DEVICE
            )
            
            freqs = results['freqs']
            times = results['times']
            Sxx = results['Sxx']
            inst_freq = results['instantaneous_freq']
            centroid = results['spectral_centroid']
            feats = results['band_features']
            names = results['band_names']
            cps = results['changepoints']
            timing = results['timing']
            
            print(f"\nPerformance breakdown:")
            print(f"  FFT:           {timing['fft']:.3f}s")
            print(f"  Features:      {timing['features']:.3f}s")
            print(f"  Band powers:   {timing['bands']:.3f}s")
            print(f"  Changepoints:  {timing['changepoints']:.3f}s")
            print(f"  TOTAL:         {timing['total']:.3f}s")
            
        else:
            # CPU fallback
            processing_status[task_id].update({
                'progress': 20,
                'stage': 'Computing FFT (CPU)...'
            })
            
            from fastmoda import sliding_fft, compute_band_powers
            from fastmoda.optimized import detect_frequency_changepoints
            
            freqs, times, Sxx = sliding_fft(x, fs, win_s)
            
            processing_status[task_id].update({
                'progress': 40,
                'stage': 'Computing features...'
            })
            
            feats, names = compute_band_powers(Sxx, freqs, bands)
            inst_freq = np.array([freqs[np.argmax(Sxx[:, i])] for i in range(Sxx.shape[1])])
            
            processing_status[task_id].update({
                'progress': 60,
                'stage': 'Detecting changepoints (frequency-based)...'
            })
            
            cps = detect_frequency_changepoints(Sxx, freqs, pen=pen)
        
        # Update progress
        processing_status[task_id].update({
            'progress': 70,
            'stage': 'Extracting band frequencies...'
        })

        band_freqs = extract_band_frequencies(Sxx, freqs, times, bands)

        # Surrogate testing (if enabled)
        surrogate_results = None
        if enable_surrogates and n_surrogates > 0:
            from fastmoda.surrogates_gpu import batched_iaaft_surrogates_gpu, wiaaft_surrogate_gpu

            processing_status[task_id].update({
                'progress': 75,
                'stage': f'Generating {n_surrogates} {surrogate_method.upper()} surrogates...'
            })

            print(f"\nGenerating {n_surrogates} {surrogate_method.upper()} surrogates...")

            if USE_GPU:
                x_torch = torch.from_numpy(x).float().to(DEVICE)

                if surrogate_method == 'iaaft':
                    # Batched IAAFT surrogates (fast)
                    surrogates = batched_iaaft_surrogates_gpu(x_torch, n_surrogates, device=DEVICE)
                else:  # wiaaft
                    # WIAAFT surrogates (slower, better multi-scale preservation)
                    surrogates = torch.stack([
                        wiaaft_surrogate_gpu(x_torch, device=DEVICE)
                        for _ in range(n_surrogates)
                    ])

                # Convert back to numpy for FFT processing
                surrogates_np = surrogates.cpu().numpy()
            else:
                # CPU fallback
                x_torch = torch.from_numpy(x).float()

                if surrogate_method == 'iaaft':
                    surrogates = batched_iaaft_surrogates_gpu(x_torch, n_surrogates, device=None)
                else:  # wiaaft
                    surrogates = torch.stack([
                        wiaaft_surrogate_gpu(x_torch, device=None)
                        for _ in range(n_surrogates)
                    ])

                surrogates_np = surrogates.numpy()

            processing_status[task_id].update({
                'progress': 78,
                'stage': f'Computing spectrograms for {n_surrogates} surrogates...'
            })

            print(f"Computing spectrograms for surrogates...")

            # Compute spectrogram for each surrogate
            from fastmoda import sliding_fft
            surrogate_spectrograms = []

            for i in range(n_surrogates):
                if USE_GPU:
                    from fastmoda.optimized_gpu import batched_sliding_fft_gpu
                    surr_torch = torch.from_numpy(surrogates_np[i]).float().to(DEVICE)
                    _, _, Sxx_surr_torch = batched_sliding_fft_gpu(surr_torch, fs=fs, win_s=win_s, device=DEVICE)
                    Sxx_surr = Sxx_surr_torch.cpu().numpy()
                else:
                    _, _, Sxx_surr = sliding_fft(surrogates_np[i], fs, win_s)

                surrogate_spectrograms.append(Sxx_surr)

            # Stack surrogates: [n_surrogates, n_freqs, n_times]
            surrogate_spectrograms = np.stack(surrogate_spectrograms)

            # Compute significance thresholds
            print(f"Computing significance thresholds (α={alpha})...")

            # Threshold at each (freq, time) point
            threshold_95 = np.percentile(surrogate_spectrograms, (1 - 0.05) * 100, axis=0)  # 95%
            threshold_99 = np.percentile(surrogate_spectrograms, (1 - 0.01) * 100, axis=0)  # 99%
            threshold_user = np.percentile(surrogate_spectrograms, (1 - alpha) * 100, axis=0)  # User-defined

            # Significance masks
            significant_95 = Sxx > threshold_95
            significant_99 = Sxx > threshold_99
            significant_user = Sxx > threshold_user

            # Count significant points
            n_significant_95 = np.sum(significant_95)
            n_significant_99 = np.sum(significant_99)
            n_total = Sxx.size

            print(f"Significance testing complete:")
            print(f"  95% threshold: {n_significant_95}/{n_total} points ({100*n_significant_95/n_total:.1f}%)")
            print(f"  99% threshold: {n_significant_99}/{n_total} points ({100*n_significant_99/n_total:.1f}%)")

            surrogate_results = {
                'enabled': True,
                'n_surrogates': n_surrogates,
                'method': surrogate_method,
                'alpha': alpha,
                'threshold_95': threshold_95,
                'threshold_99': threshold_99,
                'threshold_user': threshold_user,
                'significant_95': significant_95,
                'significant_99': significant_99,
                'significant_user': significant_user,
                'n_significant_95': int(n_significant_95),
                'n_significant_99': int(n_significant_99),
                'pct_significant_95': float(100 * n_significant_95 / n_total),
                'pct_significant_99': float(100 * n_significant_99 / n_total)
            }

        # Periodicity analysis with smart segment limiting
        next_progress = 85 if enable_surrogates else 80
        processing_status[task_id].update({
            'progress': next_progress,
            'stage': f'Analyzing periodicity ({len(cps)} changepoints)...'
        })
        
        # Use adaptive sine fitting
        MAX_SEGMENTS = 50
        sine_results = adaptive_segment_sine_fitting(x, fs, times, cps, max_segments=MAX_SEGMENTS)
        
        # Create periodicity dict for compatibility
        periodicity = {
            'sine_fits': sine_results,
            'frequency_changes': [],
            'amplitude_changes': []
        }
        
        # Detect changes
        for i in range(1, len(sine_results)):
            prev = sine_results[i-1]
            curr = sine_results[i]
            
            if prev['frequency'] > 0:
                rel_change = abs(curr['frequency'] - prev['frequency']) / prev['frequency']
                if rel_change > 0.1:
                    periodicity['frequency_changes'].append({
                        'time': curr['time_range'][0],
                        'from_freq': prev['frequency'],
                        'to_freq': curr['frequency'],
                        'rel_change': rel_change
                    })
            
            if prev['amplitude'] > 0:
                rel_change = abs(curr['amplitude'] - prev['amplitude']) / prev['amplitude']
                if rel_change > 0.1:
                    periodicity['amplitude_changes'].append({
                        'time': curr['time_range'][0],
                        'from_amp': prev['amplitude'],
                        'to_amp': curr['amplitude'],
                        'rel_change': rel_change
                    })
        
        # Generate plots
        next_progress = 92 if enable_surrogates else 90
        processing_status[task_id].update({
            'progress': next_progress,
            'stage': 'Generating visualizations...'
        })

        plots = generate_optimized_plots(x, fs, times, freqs, Sxx, feats, names,
                                        cps, band_freqs, periodicity, inst_freq,
                                        surrogate_results=surrogate_results)
        
        # Add optimization info
        if 'warning' not in processing_status[task_id]:
            processing_status[task_id]['info'] = (
                f"Optimized analysis: Changepoints detected on instantaneous frequency "
                f"(not raw power). Found {len(cps)} meaningful changes."
            )
        
        # Complete
        processing_status[task_id].update({
            'status': 'complete',
            'progress': 100,
            'stage': 'Complete!',
            'results': plots,
            'num_changepoints': len(cps),
            'num_windows': len(times)
        })
        
        print(f"\nTask {task_id} completed successfully")
        print(f"Changepoints: {len(cps)} (from {len(times)} windows)")
        
    except Exception as e:
        processing_status[task_id].update({
            'status': 'error',
            'error': str(e),
            'stage': 'Error occurred'
        })
        print(f"Error in task {task_id}: {e}")
        import traceback
        traceback.print_exc()

def sweep_background_analysis(sweep_id, x, fs, target_freqs):
    """Background task: sweep window sizes, find elbow (periodicity), detect changepoints."""
    try:
        from fastmoda.optimized import sweep_window_changepoints, detect_frequency_changepoints
        from fastmoda import sliding_fft

        def progress_cb(step, total):
            pct = int(10 + 78 * step / total)
            processing_status[sweep_id].update({
                'progress': pct,
                'stage': f'Testing window {step}/{total}...'
            })

        processing_status[sweep_id].update({'progress': 5, 'stage': 'Starting window sweep...'})

        win_sizes, cp_counts, optimal_idx = sweep_window_changepoints(
            x, fs,
            target_freqs=target_freqs if target_freqs else None,
            n_steps=12,
            progress_cb=progress_cb
        )

        optimal_win_s = win_sizes[optimal_idx]

        processing_status[sweep_id].update({'progress': 90, 'stage': 'Running final detection at optimal window...'})

        freqs_opt, times_opt, Sxx_opt = sliding_fft(x, fs, optimal_win_s)
        Sxx_det, freqs_det = Sxx_opt, freqs_opt
        if target_freqs:
            mask = np.zeros(len(freqs_opt), dtype=bool)
            for fmin, fmax in target_freqs:
                mask |= (freqs_opt >= float(fmin)) & (freqs_opt <= float(fmax))
            if mask.any():
                Sxx_det = Sxx_opt[mask]
                freqs_det = freqs_opt[mask]

        cps = detect_frequency_changepoints(Sxx_det, freqs_det, pen='auto')
        cp_times = [float(times_opt[c]) for c in cps if c < len(times_opt)]

        # Build sweep curve plot
        sweep_fig = go.Figure()
        sweep_fig.add_trace(go.Scatter(
            x=win_sizes, y=cp_counts,
            mode='lines+markers',
            name='# Changepoints',
            line={'color': '#4CAF50', 'width': 2},
            marker={'size': 7, 'color': '#4CAF50'}
        ))
        sweep_fig.add_vline(
            x=optimal_win_s,
            line_color='red', line_dash='dash', line_width=2,
            annotation_text=f'Elbow: {optimal_win_s:.3f} s',
            annotation_position='top right',
            annotation_font_color='red'
        )
        freq_hint = ''
        if target_freqs:
            freq_hint = '  |  Bands: ' + ', '.join(f'{a:.1f}–{b:.1f} Hz' for a, b in target_freqs)
        sweep_fig.update_layout(
            title=f'Window Sweep — elbow ≈ natural periodicity{freq_hint}',
            xaxis_title='Window Size (s)',
            yaxis_title='Changepoints Detected',
            xaxis_type='log',
            height=340
        )

        processing_status[sweep_id].update({
            'status': 'complete',
            'progress': 100,
            'stage': f'Done — {len(cp_times)} changepoints at optimal {optimal_win_s:.3f} s window',
            'optimal_win_s': optimal_win_s,
            'n_changepoints': len(cp_times),
            'changepoint_times': cp_times,
            'target_freqs': target_freqs or [],
            'sweep_plot': json.dumps(sweep_fig, cls=plotly.utils.PlotlyJSONEncoder)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        processing_status[sweep_id].update({'status': 'error', 'error': str(e), 'stage': 'Error'})


def generate_optimized_plots(x, fs, times, freqs, Sxx, feats, names, cps, band_freqs, periodicity, inst_freq, surrogate_results=None):
    """Generate all plots with optimization info and optional surrogate significance"""
    t_signal = np.arange(len(x)) / fs
    
    # Define band colors
    band_colors = {
        'delta': 'rgba(139, 69, 19, 0.3)',    # Brown
        'theta': 'rgba(255, 140, 0, 0.3)',    # Dark orange
        'alpha': 'rgba(255, 215, 0, 0.3)',    # Gold
        'beta': 'rgba(0, 191, 255, 0.3)',     # Deep sky blue
        'gamma': 'rgba(138, 43, 226, 0.3)'    # Blue violet
    }
    
    # 1. Signal with color-coded frequency band overlay
    signal_fig = go.Figure()
    
    # First, add colored background regions for each segment between changepoints
    cp_times = times[cps] if len(cps) > 0 else []
    segment_times = [0] + list(cp_times) + [t_signal[-1]]
    
    # Get dominant band for each segment based on instantaneous frequency
    bands = [
        (0.5, 4, 'delta'),
        (4, 8, 'theta'),
        (8, 13, 'alpha'),
        (13, 30, 'beta'),
        (30, 100, 'gamma')
    ]
    
    for i in range(len(segment_times) - 1):
        t_start = segment_times[i]
        t_end = segment_times[i + 1]
        
        # Find dominant frequency in this time range
        time_mask = (times >= t_start) & (times < t_end)
        if np.any(time_mask):
            seg_freq = np.median(inst_freq[time_mask])
            
            # Determine which band this frequency belongs to
            dominant_band = 'gamma'  # default
            for fmin, fmax, band_name in bands:
                if fmin <= seg_freq <= fmax:
                    dominant_band = band_name
                    break
            
            # Add colored rectangle for this segment
            signal_fig.add_vrect(
                x0=t_start, x1=t_end,
                fillcolor=band_colors.get(dominant_band, 'rgba(200,200,200,0.2)'),
                layer="below",
                line_width=0,
                annotation_text=f"{dominant_band}<br>{seg_freq:.1f} Hz",
                annotation_position="top left",
                annotation=dict(font_size=9, font_color="black")
            )
    
    # Add the signal trace
    signal_fig.add_trace(go.Scatter(
        x=t_signal, y=x,
        mode='lines',
        name='Signal',
        line={'color': 'black', 'width': 1.5},
        hovertemplate='Time: %{x:.3f}s<br>Amplitude: %{y:.3f}<extra></extra>'
    ))
    
    # Add changepoint lines
    for cp_t in cp_times:
        signal_fig.add_vline(
            x=cp_t, 
            line_dash="dash", 
            line_color="red", 
            line_width=2,
            opacity=0.7,
            annotation_text=f"CP: {cp_t:.2f}s",
            annotation_position="top"
        )
    
    signal_fig.update_layout(
        title=f'Signal with Color-Coded Frequency Bands ({len(cps)} changepoints)',
        xaxis_title='Time (s)',
        yaxis_title='Amplitude',
        hovermode='x unified',
        height=500,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    # 2. Spectrogram
    spec_fig = go.Figure(data=go.Heatmap(
        z=10*np.log10(Sxx**2 + 1e-12),
        x=times,
        y=freqs,
        colorscale='Viridis',
        colorbar={'title': 'Power (dB)'}
    ))

    # Add significance contours if surrogate testing was performed
    if surrogate_results and surrogate_results['enabled']:
        # Add contour showing 95% significance threshold
        spec_fig.add_trace(go.Contour(
            z=surrogate_results['significant_95'].astype(int),
            x=times,
            y=freqs,
            showscale=False,
            contours=dict(
                start=0.5,
                end=1.5,
                size=1,
                coloring='lines'
            ),
            line=dict(color='white', width=2),
            name='95% Significant',
            hoverinfo='skip'
        ))

    # Add changepoints
    for cp_t in cp_times:
        spec_fig.add_vline(x=cp_t, line_dash="dash", line_color="red", opacity=0.7)

    spec_title = 'Time-Frequency Spectrogram'
    if surrogate_results and surrogate_results['enabled']:
        spec_title += f' (white contours = significant regions, {surrogate_results["n_surrogates"]} {surrogate_results["method"].upper()} surrogates)'

    spec_fig.update_layout(
        title=spec_title,
        xaxis_title='Time (s)',
        yaxis_title='Frequency (Hz)',
        height=500
    )
    
    # 3. Frequency Band Timeline (NEW!)
    timeline_fig = go.Figure()
    
    # Add horizontal bands showing frequency ranges
    band_y_positions = {'delta': 5, 'theta': 4, 'alpha': 3, 'beta': 2, 'gamma': 1}
    band_labels = {'delta': 'δ (0.5-4 Hz)', 'theta': 'θ (4-8 Hz)', 
                   'alpha': 'α (8-13 Hz)', 'beta': 'β (13-30 Hz)', 
                   'gamma': 'γ (30-100 Hz)'}
    
    # Plot which band is active over time based on instantaneous frequency
    for i in range(len(segment_times) - 1):
        t_start = segment_times[i]
        t_end = segment_times[i + 1]
        
        time_mask = (times >= t_start) & (times < t_end)
        if np.any(time_mask):
            seg_freq = np.median(inst_freq[time_mask])
            
            # Find dominant band
            dominant_band = 'gamma'
            for fmin, fmax, band_name in bands:
                if fmin <= seg_freq <= fmax:
                    dominant_band = band_name
                    break
            
            # Add a horizontal bar for this segment
            y_pos = band_y_positions.get(dominant_band, 0)
            timeline_fig.add_trace(go.Scatter(
                x=[t_start, t_end, t_end, t_start, t_start],
                y=[y_pos - 0.4, y_pos - 0.4, y_pos + 0.4, y_pos + 0.4, y_pos - 0.4],
                fill='toself',
                fillcolor=band_colors.get(dominant_band, 'gray'),
                line=dict(color=band_colors.get(dominant_band, 'gray').replace('0.3', '0.8'), width=2),
                name=f'{band_labels[dominant_band]}: {seg_freq:.1f} Hz',
                hovertemplate=f'<b>{band_labels[dominant_band]}</b><br>' +
                             f'Time: {t_start:.2f}s - {t_end:.2f}s<br>' +
                             f'Frequency: {seg_freq:.1f} Hz<extra></extra>',
                showlegend=True
            ))
    
    # Add changepoint markers
    for cp_t in cp_times:
        timeline_fig.add_vline(
            x=cp_t, 
            line_dash="solid", 
            line_color="red", 
            line_width=3,
            opacity=0.8
        )
    
    timeline_fig.update_layout(
        title='Frequency Band Timeline (Color-coded segments)',
        xaxis_title='Time (s)',
        yaxis_title='',
        yaxis=dict(
            tickmode='array',
            tickvals=list(band_y_positions.values()),
            ticktext=list(band_labels.values()),
            range=[0.5, 5.5]
        ),
        hovermode='closest',
        height=400,
        showlegend=False
    )
    
    # 4. Instantaneous frequency with band boundaries
    inst_fig = go.Figure()
    
    # Add horizontal lines for band boundaries
    band_boundaries = [0.5, 4, 8, 13, 30, 100]
    band_names_list = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    
    for i, (lower, upper, band_name) in enumerate(bands):
        inst_fig.add_hrect(
            y0=lower, y1=upper,
            fillcolor=band_colors[band_name],
            layer="below",
            line_width=0,
            annotation_text=band_name,
            annotation_position="right"
        )
    
    inst_fig.add_trace(go.Scatter(
        x=times, y=inst_freq,
        mode='lines',
        name='Instantaneous Frequency',
        line={'color': 'black', 'width': 2.5},
        hovertemplate='Time: %{x:.3f}s<br>Frequency: %{y:.2f} Hz<extra></extra>'
    ))
    
    for cp_t in cp_times:
        inst_fig.add_vline(x=cp_t, line_dash="dash", line_color="red", 
                          line_width=2, opacity=0.7)
    
    inst_fig.update_layout(
        title='Instantaneous Frequency with Band Regions (used for changepoint detection)',
        xaxis_title='Time (s)',
        yaxis_title='Frequency (Hz)',
        hovermode='x unified',
        height=500
    )
    
    # 5. Band powers
    band_fig = go.Figure()
    for i, name in enumerate(names):
        band_fig.add_trace(go.Scatter(
            x=times, y=feats[:, i],
            mode='lines',
            name=name
        ))
    
    for cp_t in cp_times:
        band_fig.add_vline(x=cp_t, line_dash="dash", line_color="red", opacity=0.3)
    
    band_fig.update_layout(
        title='Band Powers (log scale)',
        xaxis_title='Time (s)',
        yaxis_title='Log Power',
        hovermode='x unified',
        height=400
    )
    
    # 6. Periodicity
    period_fig = go.Figure()
    
    if 'sine_fits' in periodicity and len(periodicity['sine_fits']) > 0:
        seg_times = [sf['time_range'][0] for sf in periodicity['sine_fits']]
        seg_freqs = [sf['frequency'] for sf in periodicity['sine_fits']]
        seg_amps = [sf['amplitude'] for sf in periodicity['sine_fits']]
        
        period_fig.add_trace(go.Scatter(
            x=seg_times, y=seg_freqs,
            mode='markers+lines',
            name='Segment Frequency',
            marker={'size': 8}
        ))
    
    period_fig.update_layout(
        title=f'Periodicity Analysis ({len(periodicity.get("sine_fits", []))} segments)',
        xaxis_title='Time (s)',
        yaxis_title='Frequency (Hz)',
        hovermode='x unified',
        height=400
    )
    
    # 7. Find most frequent components across all segments
    # Analyze which frequencies appear most often
    freq_histogram = {}
    duration_by_freq = {}
    
    for i in range(len(segment_times) - 1):
        t_start = segment_times[i]
        t_end = segment_times[i + 1]
        duration = t_end - t_start
        
        time_mask = (times >= t_start) & (times < t_end)
        if np.any(time_mask):
            seg_freq = np.median(inst_freq[time_mask])
            
            # Round to nearest 0.5 Hz for grouping
            freq_rounded = round(seg_freq * 2) / 2
            
            if freq_rounded not in freq_histogram:
                freq_histogram[freq_rounded] = 0
                duration_by_freq[freq_rounded] = 0
            
            freq_histogram[freq_rounded] += 1
            duration_by_freq[freq_rounded] += duration
    
    # Sort by duration (most prevalent)
    sorted_freqs = sorted(duration_by_freq.items(), key=lambda x: x[1], reverse=True)
    
    # Get top 5 most frequent components
    top_n = min(5, len(sorted_freqs))
    top_frequencies = [freq for freq, _ in sorted_freqs[:top_n]]
    
    # Create individual magnitude plots for each top frequency
    component_plots = []
    frequency_summary = []
    
    for rank, freq_component in enumerate(top_frequencies, 1):
        # Find the closest frequency bin in the spectrogram
        freq_idx = np.argmin(np.abs(freqs - freq_component))
        actual_freq = freqs[freq_idx]
        
        # Extract magnitude over time for this frequency
        magnitude = Sxx[freq_idx, :]
        
        # Create plot
        comp_fig = go.Figure()
        
        # Add magnitude trace
        comp_fig.add_trace(go.Scatter(
            x=times,
            y=magnitude,
            mode='lines',
            name=f'{actual_freq:.1f} Hz',
            line={'color': f'hsl({rank * 60}, 70%, 50%)', 'width': 2},
            fill='tozeroy',
            fillcolor=f'hsla({rank * 60}, 70%, 50%, 0.3)',
            hovertemplate=f'Time: %{{x:.2f}}s<br>Magnitude: %{{y:.3f}}<extra></extra>'
        ))
        
        # Add changepoints
        for cp_t in cp_times:
            comp_fig.add_vline(x=cp_t, line_dash="dash", line_color="red", 
                             line_width=1, opacity=0.5)
        
        # Determine which band this frequency belongs to
        freq_band = 'Unknown'
        for fmin, fmax, band_name in bands:
            if fmin <= actual_freq <= fmax:
                freq_band = band_name
                break
        
        comp_fig.update_layout(
            title=f'Component #{rank}: {actual_freq:.1f} Hz ({freq_band} band)',
            xaxis_title='Time (s)',
            yaxis_title='Magnitude',
            hovermode='x unified',
            height=300,
            showlegend=False
        )
        
        component_plots.append({
            'rank': rank,
            'frequency': float(actual_freq),
            'band': freq_band,
            'duration': float(duration_by_freq[freq_component]),
            'occurrences': int(freq_histogram[freq_component]),
            'plot': json.dumps(comp_fig, cls=plotly.utils.PlotlyJSONEncoder)
        })
        
        frequency_summary.append({
            'rank': rank,
            'frequency': float(actual_freq),
            'band': freq_band,
            'duration': float(duration_by_freq[freq_component]),
            'duration_pct': float(duration_by_freq[freq_component] / t_signal[-1] * 100),
            'occurrences': int(freq_histogram[freq_component])
        })
    
    result = {
        'signal': json.dumps(signal_fig, cls=plotly.utils.PlotlyJSONEncoder),
        'spectrogram': json.dumps(spec_fig, cls=plotly.utils.PlotlyJSONEncoder),
        'timeline': json.dumps(timeline_fig, cls=plotly.utils.PlotlyJSONEncoder),
        'instantaneous_freq': json.dumps(inst_fig, cls=plotly.utils.PlotlyJSONEncoder),
        'band_powers': json.dumps(band_fig, cls=plotly.utils.PlotlyJSONEncoder),
        'periodicity': json.dumps(period_fig, cls=plotly.utils.PlotlyJSONEncoder),
        'component_plots': component_plots,
        'frequency_summary': frequency_summary
    }

    # Add surrogate testing results if enabled
    if surrogate_results and surrogate_results['enabled']:
        result['surrogate_stats'] = {
            'enabled': True,
            'n_surrogates': surrogate_results['n_surrogates'],
            'method': surrogate_results['method'],
            'alpha': surrogate_results['alpha'],
            'n_significant_95': surrogate_results['n_significant_95'],
            'n_significant_99': surrogate_results['n_significant_99'],
            'pct_significant_95': surrogate_results['pct_significant_95'],
            'pct_significant_99': surrogate_results['pct_significant_99']
        }

    return result

@app.route('/analyze_modwt', methods=['POST'])
def analyze_modwt():
    """MODWT wavelet transform analysis endpoint"""
    if 'file' not in request.files or not request.files['file'].filename:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    try:
        fs = float(request.form.get('fs', 1.0))
        wavelet = request.form.get('wavelet', 'la8')
        level = request.form.get('level', None)
        if level is not None:
            level = int(level)

        # Load signal
        x, _ = load_signal(filepath)

        # Create task
        task_id = str(uuid.uuid4())

        processing_status[task_id] = {
            'status': 'processing',
            'progress': 10,
            'stage': 'Starting MODWT decomposition...',
            'signal_shape': x.shape,
            'fs': fs
        }

        # Start background processing
        thread = Thread(target=process_modwt_background,
                       args=(task_id, x, fs, wavelet, level))
        thread.daemon = True
        thread.start()

        return jsonify({
            'task_id': task_id,
            'signal_length': len(x),
            'sampling_rate': fs,
            'wavelet': wavelet
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f"Error: {str(e)}"}), 500


def _modwt_scipy_fallback(x, fs, level):
    """Bandpass decomposition as MODWT surrogate when torch unavailable."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    bands = [(0.5,4,'delta'),(4,8,'theta'),(8,12,'alpha'),(12,30,'beta'),(30,100,'gamma')]
    n = min(level, len(bands))
    t = np.arange(len(x)) / fs
    step = max(1, len(x) // 300)
    fig_c = make_subplots(rows=n, cols=1,
        subplot_titles=[f'Level {i+1}: {bands[i][2]}' for i in range(n)],
        vertical_spacing=0.05)
    energies = []
    for i in range(n):
        lo, hi, name = bands[i]
        nyq = fs / 2
        try:
            from scipy.signal import butter, filtfilt
            b, a = butter(4, [max(0.001,lo/nyq), min(0.999,hi/nyq)], 'band')
            coeff = filtfilt(b, a, x)
        except Exception:
            coeff = np.zeros_like(x)
        energies.append(float(np.sum(coeff**2)))
        fig_c.add_trace(go.Scatter(x=t[::step].tolist(), y=coeff[::step].tolist(),
            mode='lines', line={'width':1}, name=name), row=i+1, col=1)
    fig_c.update_layout(height=200*n, showlegend=False,
                        title='Bandpass Decomposition (scipy CPU fallback)')
    tot = sum(energies)+1e-12
    fig_e = go.Figure(go.Bar(x=[bands[i][2] for i in range(n)],
                              y=[e/tot for e in energies]))
    fig_e.update_layout(title='Band Energy Distribution', yaxis_title='Relative Energy')
    return {
        'coefficients_plot': json.dumps(fig_c, cls=plotly.utils.PlotlyJSONEncoder),
        'energy_plot':       json.dumps(fig_e, cls=plotly.utils.PlotlyJSONEncoder),
        'n_levels': n, 'reconstruction_error': 0.0,
        'gpu_used': False, 'method': 'scipy_bandpass',
    }


def process_modwt_background(task_id, x, fs, wavelet, level):
    """Background processing for MODWT analysis"""
    try:
        processing_status[task_id].update({'progress': 10, 'stage': 'Starting MODWT…'})

        try:
            from fastmoda.modwt_gpu import modwt_gpu, imodwt_gpu
        except (ImportError, Exception):
            result = _modwt_scipy_fallback(x, fs, level)
            processing_status[task_id].update({
                'status': 'complete', 'progress': 100, 'stage': 'Complete!',
                'results': result,
            })
            return

        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        processing_status[task_id].update({
            'progress': 20,
            'stage': 'Converting signal to GPU...'
        })

        # Convert to torch tensor
        if USE_GPU:
            x_tensor = torch.from_numpy(x).float().to(DEVICE)
        else:
            x_tensor = torch.from_numpy(x).float()

        processing_status[task_id].update({
            'progress': 30,
            'stage': 'Computing MODWT decomposition...'
        })

        # Perform MODWT
        w, v = modwt_gpu(x_tensor, wavelet=wavelet, level=level, device=DEVICE if USE_GPU else None)

        # Convert back to numpy for plotting
        w_np = [w_i.cpu().numpy() if USE_GPU else w_i.numpy() for w_i in w]
        v_np = v.cpu().numpy() if USE_GPU else v.numpy()

        processing_status[task_id].update({
            'progress': 60,
            'stage': 'Generating visualizations...'
        })

        # Create time axis
        t = np.arange(len(x)) / fs

        # Create plots
        n_levels = len(w_np)

        # 1. Create coefficient subplots
        fig_coeffs = make_subplots(
            rows=n_levels + 2, cols=1,
            subplot_titles=['Original Signal'] + [f'Level {i+1} Wavelet Coefficients' for i in range(n_levels)] + ['Scaling Coefficients'],
            vertical_spacing=0.02,
            row_heights=[0.15] + [0.7 / (n_levels + 1)] * (n_levels + 1)
        )

        # Original signal
        fig_coeffs.add_trace(
            go.Scatter(x=t, y=x, mode='lines', name='Original', line=dict(color='black', width=1)),
            row=1, col=1
        )
        fig_coeffs.update_yaxes(title_text='Amplitude', row=1, col=1)

        # Wavelet coefficients for each level
        for i, w_i in enumerate(w_np, 1):
            fig_coeffs.add_trace(
                go.Scatter(x=t, y=w_i, mode='lines', name=f'W{i}',
                          line=dict(width=0.8), showlegend=False),
                row=i+1, col=1
            )
            fig_coeffs.update_yaxes(title_text=f'W{i}', row=i+1, col=1)

        # Scaling coefficients
        fig_coeffs.add_trace(
            go.Scatter(x=t, y=v_np, mode='lines', name=f'V{n_levels}',
                      line=dict(color='purple', width=0.8), showlegend=False),
            row=n_levels+2, col=1
        )
        fig_coeffs.update_yaxes(title_text=f'V{n_levels}', row=n_levels+2, col=1)
        fig_coeffs.update_xaxes(title_text='Time (s)', row=n_levels+2, col=1)

        fig_coeffs.update_layout(
            height=200 * (n_levels + 2),
            title_text=f'MODWT Decomposition ({wavelet.upper()} wavelet, {n_levels} levels)',
            showlegend=False
        )

        # 2. Create heatmap of all coefficients
        # Stack all wavelet coefficients
        all_coeffs = np.vstack(w_np)

        # Create scale labels (approximate frequency ranges)
        scale_labels = []
        for i in range(1, n_levels + 1):
            # Scale i corresponds to frequencies [fs/2^(i+1), fs/2^i]
            f_min = fs / (2**(i+1))
            f_max = fs / (2**i)
            scale_labels.append(f'L{i}<br>[{f_min:.2f}-{f_max:.2f} Hz]')

        fig_heatmap = go.Figure(data=go.Heatmap(
            z=all_coeffs,
            x=t,
            y=list(range(1, n_levels + 1)),
            colorscale='RdBu',
            zmid=0,
            colorbar=dict(title='Coefficient<br>Magnitude')
        ))

        fig_heatmap.update_layout(
            title=f'MODWT Coefficient Heatmap',
            xaxis_title='Time (s)',
            yaxis_title='Decomposition Level',
            yaxis=dict(
                tickmode='array',
                tickvals=list(range(1, n_levels + 1)),
                ticktext=scale_labels
            ),
            height=400
        )

        # 3. Reconstruction verification
        processing_status[task_id].update({
            'progress': 80,
            'stage': 'Verifying reconstruction...'
        })

        # Reconstruct
        if USE_GPU:
            w_recon = [torch.from_numpy(w_i).float().to(DEVICE) for w_i in w_np]
            v_recon = torch.from_numpy(v_np).float().to(DEVICE)
        else:
            w_recon = [torch.from_numpy(w_i).float() for w_i in w_np]
            v_recon = torch.from_numpy(v_np).float()

        x_recon = imodwt_gpu(w_recon, v_recon, wavelet=wavelet, device=DEVICE if USE_GPU else None)
        x_recon_np = x_recon.cpu().numpy() if USE_GPU else x_recon.numpy()

        # Compute reconstruction error
        recon_error = np.linalg.norm(x - x_recon_np) / np.linalg.norm(x)

        # Create reconstruction comparison plot
        fig_recon = go.Figure()
        fig_recon.add_trace(go.Scatter(
            x=t, y=x, mode='lines', name='Original',
            line=dict(color='black', width=1.5)
        ))
        fig_recon.add_trace(go.Scatter(
            x=t, y=x_recon_np, mode='lines', name='Reconstructed',
            line=dict(color='red', width=1, dash='dash')
        ))
        fig_recon.add_trace(go.Scatter(
            x=t, y=x - x_recon_np, mode='lines', name='Error (×100)',
            line=dict(color='blue', width=1),
            yaxis='y2'
        ))

        fig_recon.update_layout(
            title=f'Reconstruction Verification (Error: {recon_error:.2e})',
            xaxis_title='Time (s)',
            yaxis_title='Amplitude',
            yaxis2=dict(
                title='Error',
                overlaying='y',
                side='right'
            ),
            hovermode='x unified',
            height=400
        )

        # 4. Energy distribution across scales
        energies = [np.sum(w_i**2) for w_i in w_np]
        total_energy = sum(energies) + np.sum(v_np**2)
        energy_pcts = [100 * e / total_energy for e in energies]

        fig_energy = go.Figure(data=[
            go.Bar(
                x=[f'Level {i+1}' for i in range(n_levels)],
                y=energy_pcts,
                marker_color='steelblue',
                hovertemplate='%{x}<br>Energy: %{y:.2f}%<extra></extra>'
            )
        ])

        fig_energy.update_layout(
            title='Energy Distribution Across Decomposition Levels',
            xaxis_title='Decomposition Level',
            yaxis_title='Energy (%)',
            height=400
        )

        processing_status[task_id].update({
            'status': 'complete',
            'progress': 100,
            'stage': 'Complete!',
            'results': {
                'coefficients_plot': json.dumps(fig_coeffs, cls=plotly.utils.PlotlyJSONEncoder),
                'heatmap_plot': json.dumps(fig_heatmap, cls=plotly.utils.PlotlyJSONEncoder),
                'reconstruction_plot': json.dumps(fig_recon, cls=plotly.utils.PlotlyJSONEncoder),
                'energy_plot': json.dumps(fig_energy, cls=plotly.utils.PlotlyJSONEncoder),
                'n_levels': n_levels,
                'reconstruction_error': float(recon_error),
                'scale_info': [
                    {
                        'level': i+1,
                        'freq_range': [fs / (2**(i+2)), fs / (2**(i+1))],
                        'energy_pct': float(energy_pcts[i])
                    }
                    for i in range(n_levels)
                ]
            }
        })

        print(f"\nMODWT Task {task_id} completed successfully")
        print(f"Levels: {n_levels}, Reconstruction error: {recon_error:.2e}")

    except Exception as e:
        processing_status[task_id].update({
            'status': 'error',
            'error': str(e),
            'stage': 'Error occurred'
        })
        print(f"Error in MODWT task {task_id}: {e}")
        import traceback
        traceback.print_exc()


@app.route('/analyze_coherence', methods=['POST'])
def analyze_coherence():
    """Multi-signal coherence analysis endpoint (GPU-accelerated or scipy fallback)"""

    # Check for multiple files
    files = request.files.getlist('files')
    if len(files) < 2:
        return jsonify({'error': 'At least 2 signals required for coherence analysis'}), 400
    if len(files) > 6:
        return jsonify({'error': 'Maximum 6 signals supported'}), 400
    
    try:
        fs = float(request.form.get('fs', 1.0))
        win_s = float(request.form.get('win', 1.0))
        overlap = float(request.form.get('overlap', 0.5))
        numcycles = int(request.form.get('numcycles', 10))
        wavelet_type = request.form.get('wavelet_type', 'lognorm')
        preprocess = request.form.get('preprocess', 'false').lower() == 'true'
        cut_edges = request.form.get('cut_edges', 'true').lower() == 'true'
        surrogate_method = request.form.get('surrogate_method', 'none')
        n_surrogates = int(request.form.get('n_surrogates', 19))
        freq_min = float(request.form.get('freq_min', 0.5))
        freq_max_raw = request.form.get('freq_max', '')
        freq_max = float(freq_max_raw) if freq_max_raw else None
        central_freq_raw = request.form.get('central_freq', '')
        central_freq = float(central_freq_raw) if central_freq_raw else None
        surrogate_analysis = request.form.get('surrogate_analysis', 'Maximum')
        surrogate_percentile = float(request.form.get('surrogate_percentile', 0.95))
        subtract_surrogates = request.form.get('subtract_surrogates', 'false').lower() == 'true'

        # Load all signals
        signals = []
        signal_names = []
        for file in files:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            signal, actual_fs = load_signal(filepath)
            if actual_fs and actual_fs != 1.0:
                fs = actual_fs
            
            signals.append(signal)
            signal_names.append(file.filename)
        
        # Check all signals have same length
        lengths = [len(s) for s in signals]
        if len(set(lengths)) > 1:
            return jsonify({
                'error': f'All signals must have same length. Got: {lengths}'
            }), 400
        
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        processing_status[task_id] = {
            'stage': 'Starting coherence analysis',
            'progress': 0,
            'error': None,
            'result': None
        }
        
        # Start background processing
        thread = Thread(
            target=process_coherence_background,
            args=(task_id, signals, signal_names, fs, win_s, overlap, numcycles,
                  wavelet_type, preprocess, cut_edges, surrogate_method, n_surrogates,
                  freq_min, freq_max, central_freq,
                  surrogate_analysis, surrogate_percentile, subtract_surrogates)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({'task_id': task_id})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _coherence_scipy_fallback(signals, signal_names, fs, win_s, numcycles=10,
                               wavelet_type='lognorm', preprocess=False, cut_edges=True,
                               surrogate_method='none', n_surrogates=19,
                               freq_min=0.5, freq_max=None, central_freq=None,
                               surrogate_analysis='Maximum', surrogate_percentile=0.95,
                               subtract_surrogates=False):
    """
    CPU coherence fallback using vectorised CWT + proper time-localised coherence.
    No per-sample loops — cwt_complex and time_localized_coherence
    are both fully vectorised (batch FFT + gather).
    """
    from fastmoda.ridge_gpu import cwt_complex, time_localized_coherence
    from fastmoda.surrogates import phase_randomization_surrogate, iaaft_surrogate
    from scipy.signal import detrend

    n_freqs = 50
    fmin    = freq_min
    fmax    = freq_max if freq_max is not None else min(fs / 2.0, 100.0)
    freqs   = np.logspace(np.log10(fmin), np.log10(fmax), n_freqs)
    n_cycles = central_freq if central_freq is not None else 6.0

    if preprocess:
        signals = [detrend(s) for s in signals]

    cwts = [cwt_complex(s, freqs, fs, wavelet=wavelet_type, n_cycles=n_cycles, cut_edges=cut_edges) for s in signals]

    results = {}
    for i in range(len(signals)):
        for j in range(i + 1, len(signals)):
            n1, n2 = signal_names[i], signal_names[j]

            cwt1 = cwts[i]
            cwt2 = cwts[j]

            tpc  = time_localized_coherence(cwt1, cwt2, freqs, fs,
                                             numcycles=numcycles)  # (NF, T) vectorised
            phcoh  = np.nanmean(tpc, axis=1)                       # (NF,)
            phdiff = np.angle(np.nanmean(cwt1 * np.conj(cwt2), axis=1))

            T = tpc.shape[1]
            ds = max(1, T // 100)
            pair_result = {
                'freqs':        freqs,
                'phcoh':        phcoh,
                'phdiff':       phdiff,
                'tpc':          tpc[:, ::ds],
                'time_windows': np.arange(0, T, ds) / fs,
            }

            if surrogate_method != 'none':
                # RP -> phase-randomized surrogates; IAAFT1/IAAFT2/WIAAFT -> CPU IAAFT approximation
                surr_phcoh = np.zeros((n_surrogates, len(freqs)))
                for k in range(n_surrogates):
                    if surrogate_method == 'RP':
                        surr_signal = phase_randomization_surrogate(signals[j], seed=k)
                    else:
                        surr_signal = iaaft_surrogate(signals[j], seed=k)
                    surr_cwt = cwt_complex(surr_signal, freqs, fs, wavelet=wavelet_type, n_cycles=n_cycles, cut_edges=cut_edges)
                    surr_tpc = time_localized_coherence(cwt1, surr_cwt, freqs, fs, numcycles=numcycles)
                    surr_phcoh[k] = np.nanmean(surr_tpc, axis=1)

                if surrogate_analysis == 'Percentile':
                    # MATLAB CoherenceMulti.m: K = floor((ns+1)*alpha); s1 = sort(t,'descend'); thresh = s1(K,:)
                    K = int(np.floor((n_surrogates + 1) * surrogate_percentile))
                    if K == 0:
                        threshold = np.max(surr_phcoh, axis=0)
                    else:
                        K = min(K, n_surrogates)
                        threshold = np.sort(surr_phcoh, axis=0)[::-1][K - 1]
                else:  # 'Maximum'
                    threshold = np.max(surr_phcoh, axis=0)

                pair_result['surrogate_threshold'] = threshold
                if subtract_surrogates:
                    pair_result['phcoh_subtracted'] = np.maximum(phcoh - threshold, 0)

            results[(n1, n2)] = pair_result
    return results


def process_coherence_background(task_id, signals, signal_names, fs, win_s, overlap, numcycles,
                                  wavelet_type='lognorm', preprocess=False, cut_edges=True,
                                  surrogate_method='none', n_surrogates=19,
                                  freq_min=0.5, freq_max=None, central_freq=None,
                                  surrogate_analysis='Maximum', surrogate_percentile=0.95,
                                  subtract_surrogates=False):
    """Background processing for coherence analysis"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    try:
        processing_status[task_id]['stage'] = 'Computing coherence'
        processing_status[task_id]['progress'] = 20

        # The GPU fast path only supports the MATLAB-default lognorm/cut-edges
        # pipeline with no surrogates; advanced params always use the CWT fallback.
        needs_cwt_fallback = (
            wavelet_type != 'lognorm' or preprocess or not cut_edges or
            surrogate_method != 'none' or freq_min != 0.5 or
            freq_max is not None or central_freq is not None
        )

        gpu_used = False
        if not needs_cwt_fallback:
            try:
                from fastmoda.coherence_gpu import compute_multi_pair_coherence_gpu
                results = compute_multi_pair_coherence_gpu(
                    signals, signal_names, fs,
                    win_s=win_s, overlap=overlap, numcycles=numcycles,
                    device=DEVICE
                )
                gpu_used = True
            except (ImportError, Exception):
                needs_cwt_fallback = True

        if needs_cwt_fallback:
            results = _coherence_scipy_fallback(
                signals, signal_names, fs, win_s, numcycles,
                wavelet_type=wavelet_type, preprocess=preprocess, cut_edges=cut_edges,
                surrogate_method=surrogate_method, n_surrogates=n_surrogates,
                freq_min=freq_min, freq_max=freq_max, central_freq=central_freq,
                surrogate_analysis=surrogate_analysis, surrogate_percentile=surrogate_percentile,
                subtract_surrogates=subtract_surrogates
            )

        processing_status[task_id]['stage'] = 'Generating visualizations'
        processing_status[task_id]['progress'] = 60
        
        # Create visualizations for each pair
        pair_plots = {}
        for (name1, name2), result in results.items():
            freqs = result['freqs']
            phcoh = result['phcoh']
            phdiff = result['phdiff']
            tpc = result['tpc']
            time_windows = result['time_windows']
            
            has_surrogate = 'surrogate_threshold' in result
            has_subtracted = 'phcoh_subtracted' in result
            show_legend_row1 = has_surrogate or has_subtracted

            # Create subplot: coherence + TPC heatmap + phase diff
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=(
                    f'Time-Averaged Coherence: {name1} vs {name2}',
                    'Time-Localized Coherence',
                    'Phase Difference'
                ),
                vertical_spacing=0.1,
                row_heights=[0.3, 0.4, 0.3]
            )

            # 1. Time-averaged coherence
            if has_subtracted:
                fig.add_trace(
                    go.Scatter(
                        x=freqs, y=result['phcoh_subtracted'],
                        mode='lines',
                        name='Surrogate Subtracted',
                        showlegend=True,
                        line=dict(color='blue', width=2),
                        hovertemplate='Freq: %{x:.2f} Hz<br>Coherence: %{y:.3f}<extra></extra>'
                    ),
                    row=1, col=1
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=freqs, y=phcoh,
                        mode='lines',
                        name='Coherence',
                        showlegend=show_legend_row1,
                        line=dict(color='blue', width=2),
                        hovertemplate='Freq: %{x:.2f} Hz<br>Coherence: %{y:.3f}<extra></extra>'
                    ),
                    row=1, col=1
                )
                if has_surrogate:
                    threshold_label = ('Surrogate threshold (Maximum)' if surrogate_analysis != 'Percentile'
                                        else f'Surrogate threshold ({surrogate_percentile * 100:.0f}%)')
                    fig.add_trace(
                        go.Scatter(
                            x=freqs, y=result['surrogate_threshold'],
                            mode='lines',
                            name=threshold_label,
                            showlegend=True,
                            line=dict(color='gray', width=1.5, dash='dash'),
                            hovertemplate='Freq: %{x:.2f} Hz<br>Threshold: %{y:.3f}<extra></extra>'
                        ),
                        row=1, col=1
                    )
            fig.update_xaxes(title_text='Frequency (Hz)', row=1, col=1)
            fig.update_yaxes(title_text='Coherence', range=[0, 1], row=1, col=1)
            
            # 2. Time-localized coherence heatmap
            fig.add_trace(
                go.Heatmap(
                    x=time_windows,
                    y=freqs,
                    z=tpc,
                    colorscale='Viridis',
                    colorbar=dict(title='Coherence', y=0.5, len=0.4),
                    hovertemplate='Time: %{x:.2f} s<br>Freq: %{y:.2f} Hz<br>Coherence: %{z:.3f}<extra></extra>'
                ),
                row=2, col=1
            )
            fig.update_xaxes(title_text='Time (s)', row=2, col=1)
            fig.update_yaxes(title_text='Frequency (Hz)', row=2, col=1)
            
            # 3. Phase difference
            fig.add_trace(
                go.Scatter(
                    x=freqs, y=np.rad2deg(phdiff),
                    mode='lines',
                    name='Phase Diff',
                    line=dict(color='red', width=2),
                    hovertemplate='Freq: %{x:.2f} Hz<br>Phase: %{y:.1f}°<extra></extra>'
                ),
                row=3, col=1
            )
            fig.update_xaxes(title_text='Frequency (Hz)', row=3, col=1)
            fig.update_yaxes(title_text='Phase Difference (degrees)', row=3, col=1)
            
            fig.update_layout(
                height=1200,
                showlegend=show_legend_row1,
                title_text=f'Wavelet Phase Coherence Analysis: {name1} ↔ {name2}',
                title_font_size=16
            )

            pair_plots[f'{name1}_vs_{name2}'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        processing_status[task_id].update({
            'status': 'complete', 'stage': 'Complete!', 'progress': 100,
            'results': {
                'pair_plots': pair_plots,
                'n_pairs': len(results),
                'signal_names': signal_names,
                'gpu_used': gpu_used,
                'method': 'wavelet_gpu' if gpu_used else 'wavelet_cwt',
                'wavelet_type': wavelet_type,
                'preprocess': preprocess,
                'cut_edges': cut_edges,
                'surrogate_method': surrogate_method,
                'n_surrogates': n_surrogates if surrogate_method != 'none' else 0,
                'freq_min': freq_min,
                'freq_max': freq_max,
                'central_freq': central_freq,
                'surrogate_analysis': surrogate_analysis if surrogate_method != 'none' else None,
                'surrogate_percentile': surrogate_percentile if surrogate_method != 'none' else None,
                'subtract_surrogates': subtract_surrogates if surrogate_method != 'none' else False,
            }
        })

    except Exception as e:
        processing_status[task_id].update({'status': 'error', 'error': str(e), 'stage': 'Error'})
        import traceback; traceback.print_exc()


@app.route('/analyze_bispectrum', methods=['POST'])
def analyze_bispectrum():
    """Bispectrum analysis endpoint (GPU-accelerated or scipy fallback)"""

    files = request.files.getlist('files')
    if len(files) < 1:
        return jsonify({'error': 'At least 1 signal required'}), 400
    if len(files) > 2:
        files = files[:2]  # Max 2 signals
    
    try:
        fs = float(request.form.get('fs', 1.0))
        freq_min = float(request.form.get('freq_min', 0.5))
        freq_max = float(request.form.get('freq_max', fs/2))
        n_freqs = int(request.form.get('n_freqs', 50))
        bispec_type = request.form.get('bispec_type', '122')
        
        # Load signals
        signals = []
        signal_names = []
        for file in files:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            signal, actual_fs = load_signal(filepath)
            if actual_fs and actual_fs != 1.0:
                fs = actual_fs
            signals.append(signal)
            signal_names.append(file.filename)
        
        # Pad if only one signal
        if len(signals) == 1:
            signals.append(signals[0])
            signal_names.append(signal_names[0])
        
        task_id = str(uuid.uuid4())
        processing_status[task_id] = {
            'stage': 'Starting bispectrum analysis',
            'progress': 0,
            'error': None,
            'result': None
        }
        
        thread = Thread(
            target=process_bispectrum_background,
            args=(task_id, signals, signal_names, fs, freq_min, freq_max, n_freqs, bispec_type)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({'task_id': task_id})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def process_bispectrum_background(task_id, signals, signal_names, fs, freq_min, freq_max, n_freqs, bispec_type):
    """Background processing for bispectrum analysis"""
    import plotly.graph_objects as go

    try:
        processing_status[task_id]['stage'] = 'Computing bispectrum'
        processing_status[task_id]['progress'] = 20

        gpu_used = False
        try:
            from fastmoda.bispectrum_gpu import wavelet_bispectrum_gpu, find_significant_couplings
            result = wavelet_bispectrum_gpu(
                torch.from_numpy(signals[0]).to(DEVICE),
                torch.from_numpy(signals[1]).to(DEVICE),
                fs, freq_range=(freq_min, freq_max),
                n_freqs=n_freqs, bispectrum_type=bispec_type, device=DEVICE
            )
            couplings = find_significant_couplings(result, threshold_percentile=95)
            gpu_used = True
        except (ImportError, Exception):
            from fastmoda.analysis_gpu import bispectrum_gpu as bispec_cpu
            # bispectrum_gpu(x, fs, nfft, overlap) — no freq_range kwarg
            nfft_val = min(256, len(signals[0]) // 2)
            r = bispec_cpu(signals[0], fs=fs, nfft=nfft_val)
            biamp = np.abs(r.get('bispectrum', np.zeros((nfft_val // 2, nfft_val // 2))))
            freq_arr = r.get('frequencies', np.linspace(0, fs / 2, biamp.shape[0]))
            result = {
                'freq': freq_arr, 'biamp': biamp,
                'coupling_strength': float(np.mean(np.abs(biamp))),
                'freq_range': [float(freq_arr[0]), float(freq_arr[-1])],
            }
            couplings = []

        processing_status[task_id]['stage'] = 'Finding significant couplings'
        processing_status[task_id]['progress'] = 60
        
        processing_status[task_id]['stage'] = 'Creating visualizations'
        processing_status[task_id]['progress'] = 80
        
        # Create bispectrum heatmap
        freq = result['freq']
        biamp = result['biamp']
        
        fig = go.Figure()
        
        # Amplitude heatmap
        fig.add_trace(go.Heatmap(
            x=freq,
            y=freq,
            z=biamp,
            colorscale='Hot',
            colorbar=dict(title='Amplitude'),
            hovertemplate='f1: %{x:.2f} Hz<br>f2: %{y:.2f} Hz<br>Amplitude: %{z:.3e}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'Wavelet Bispectrum (Type {bispec_type}): {signal_names[0]} & {signal_names[1]}',
            xaxis_title='Frequency f1 (Hz)',
            yaxis_title='Frequency f2 (Hz)',
            width=800,
            height=800
        )
        
        # Top couplings table
        top_couplings = couplings[:10]  # Top 10
        
        # biphase: present in wavelet_bispectrum_gpu result; compute from complex
        # bispectrum matrix on the CPU path
        raw_bisp  = result.get('bispectrum')           # complex matrix or None
        biphase_m = (np.angle(raw_bisp).mean() if raw_bisp is not None
                     else result.get('biphase', np.zeros_like(result['biamp'])).mean())
        biphase_std = (np.std(np.angle(raw_bisp)) if raw_bisp is not None
                       else 0.0)

        processing_status[task_id].update({
            'status': 'complete', 'stage': 'Complete!', 'progress': 100,
            'results': {
                'bispectrum_plot':  json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder),
                'coupling_strength': result['coupling_strength'],
                'top_couplings':    [
                    {'f1': f1, 'f2': f2, 'f3': f1+f2, 'strength': float(s)}
                    for f1, f2, s in top_couplings
                ],
                'bispec_type':      bispec_type,
                'freq_range':       result['freq_range'],
                'mean_biphase_deg': round(float(np.degrees(biphase_m)), 2),
                'std_biphase_deg':  round(float(np.degrees(biphase_std)), 2),
                'gpu_used':         gpu_used,
                'method':           'wavelet_gpu' if gpu_used else 'scipy_bispectrum',
            }
        })

    except Exception as e:
        processing_status[task_id].update({'status': 'error', 'error': str(e), 'stage': 'Error'})
        import traceback; traceback.print_exc()


@app.route('/analyze_bayesian', methods=['POST'])
def analyze_bayesian():
    """Bayesian inference endpoint for phase coupling (GPU-accelerated or scipy fallback)"""

    files = request.files.getlist('files')
    if len(files) != 2:
        return jsonify({'error': 'Exactly 2 signals required for Bayesian analysis'}), 400
    
    try:
        fs = float(request.form.get('fs', 1.0))
        band1_low = float(request.form.get('band1_low', 0.5))
        band1_high = float(request.form.get('band1_high', 2.0))
        band2_low = float(request.form.get('band2_low', 0.5))
        band2_high = float(request.form.get('band2_high', 2.0))
        window_s = float(request.form.get('window_s', 40.0))
        n_surrogates = int(request.form.get('n_surrogates', 19))
        overlap = float(request.form.get('overlap', 0.75))
        propagation = float(request.form.get('propagation', 0.2))
        bn = int(request.form.get('bn', 2))
        signif = float(request.form.get('signif', 95.0))

        # Load signals
        signals = []
        signal_names = []
        for file in files:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            signal, actual_fs = load_signal(filepath)
            if actual_fs and actual_fs != 1.0:
                fs = actual_fs
            signals.append(signal)
            signal_names.append(file.filename)
        
        task_id = str(uuid.uuid4())
        processing_status[task_id] = {
            'stage': 'Starting Bayesian inference',
            'progress': 0,
            'error': None,
            'result': None
        }
        
        thread = Thread(
            target=process_bayesian_background,
            args=(task_id, signals, signal_names, fs,
                  (band1_low, band1_high), (band2_low, band2_high),
                  window_s, n_surrogates, overlap, propagation, bn, signif)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({'task_id': task_id})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _bayesian_scipy_fallback(signals, fs, band1, band2, window_s, overlap=0.75,
                              propagation=0.2, bn=2, signif=95.0):
    """Hilbert-based phase coupling fallback when torch is unavailable."""
    from scipy.signal import butter, filtfilt, hilbert
    def bandpass(x, lo, hi):
        b, a = butter(4, [lo / (fs / 2), hi / (fs / 2)], 'band')
        return filtfilt(b, a, x)
    x1f = bandpass(signals[0], band1[0], band1[1])
    x2f = bandpass(signals[1], band2[0], band2[1])
    phase1 = np.angle(hilbert(x1f))
    phase2 = np.angle(hilbert(x2f))
    pd = phase2 - phase1
    win_samples = max(1, int(window_s * fs))
    n_wins = max(1, len(signals[0]) // win_samples)
    cpl = np.array([
        float(np.abs(np.mean(np.exp(1j * pd[i * win_samples:(i + 1) * win_samples]))))
        for i in range(n_wins)
    ])
    t = np.linspace(0, len(signals[0]) / fs, n_wins)
    return {'time': t, 'cpl1': cpl, 'cpl2': cpl, 'direction': np.zeros(n_wins)}


def process_bayesian_background(task_id, signals, signal_names, fs, band1, band2, window_s,
                                  n_surrogates, overlap=0.75, propagation=0.2, bn=2, signif=95.0):
    """Background processing for Bayesian inference"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    try:
        processing_status[task_id]['stage'] = 'Running Bayesian inference'
        processing_status[task_id]['progress'] = 20

        gpu_used = False
        try:
            from fastmoda.bayesian_gpu import bayesian_inference_full
            result = bayesian_inference_full(
                torch.from_numpy(signals[0]).to(DEVICE),
                torch.from_numpy(signals[1]).to(DEVICE),
                fs, band1=band1, band2=band2,
                window_s=window_s, n_surrogates=n_surrogates,
                overlap=overlap, propagation=propagation, bn=bn, signif=signif,
                device=DEVICE
            )
            gpu_used = True
        except (ImportError, Exception):
            result = _bayesian_scipy_fallback(signals, fs, band1, band2, window_s,
                                               overlap=overlap, propagation=propagation,
                                               bn=bn, signif=signif)
        
        processing_status[task_id]['stage'] = 'Creating visualizations'
        processing_status[task_id]['progress'] = 70
        
        # Create plots
        time = result['time']
        cpl1 = result['cpl1']
        cpl2 = result['cpl2']
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(
                f'Coupling Strength: {signal_names[0]} ↔ {signal_names[1]}',
                'Coupling Direction'
            ),
            vertical_spacing=0.15
        )
        
        # Coupling strengths
        fig.add_trace(
            go.Scatter(x=time, y=cpl2, mode='lines', name=f'{signal_names[0]}→{signal_names[1]}',
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=time, y=cpl1, mode='lines', name=f'{signal_names[1]}→{signal_names[0]}',
                      line=dict(color='red', width=2)),
            row=1, col=1
        )
        
        # Surrogate thresholds
        if 'surr_cpl1' in result:
            fig.add_trace(
                go.Scatter(x=time, y=result['surr_cpl2'], mode='lines', name='Threshold (95%)',
                          line=dict(color='blue', width=1, dash='dash')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=time, y=result['surr_cpl1'], mode='lines', name='Threshold (95%)',
                          line=dict(color='red', width=1, dash='dash')),
                row=1, col=1
            )
        
        # Direction
        fig.add_trace(
            go.Scatter(x=time, y=result['direction'], mode='lines', name='Direction',
                      line=dict(color='purple', width=2)),
            row=2, col=1
        )
        fig.add_hline(y=0, line=dict(color='gray', dash='dot'), row=2, col=1)
        
        fig.update_xaxes(title_text='Time (s)', row=2, col=1)
        fig.update_yaxes(title_text='Coupling Strength', row=1, col=1)
        fig.update_yaxes(title_text='Direction', range=[-1, 1], row=2, col=1)
        
        fig.update_layout(height=800, showlegend=True)
        
        processing_status[task_id].update({
            'status': 'complete', 'stage': 'Complete!', 'progress': 100,
            'results': {
                'coupling_plot': json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder),
                'mean_cpl1': float(np.mean(cpl1)),
                'mean_cpl2': float(np.mean(cpl2)),
                'mean_direction': float(np.mean(result['direction'])),
                'band1': band1, 'band2': band2,
                'window_s': window_s,
                'n_surrogates': n_surrogates if 'surr_cpl1' in result else 0,
                'overlap': overlap,
                'propagation': propagation,
                'bn': bn,
                'signif': signif,
                'gpu_used': gpu_used,
                'method': 'bayesian_gpu' if gpu_used else 'hilbert_plv',
            }
        })

    except Exception as e:
        processing_status[task_id].update({'status': 'error', 'error': str(e), 'stage': 'Error'})
        import traceback; traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════════════════
# NEW ENDPOINTS — STFT / CWT / Hilbert / Surrogates / Features
# ═══════════════════════════════════════════════════════════════════════════════

def _async_route(task_id, x, fs, thread_target, **kwargs):
    t = Thread(target=thread_target, args=(task_id, x, fs), kwargs=kwargs)
    t.daemon = True
    t.start()


@app.route('/analyze_stft', methods=['POST'])
def analyze_stft():
    """Short-Time Fourier Transform endpoint."""
    if 'file' not in request.files or not request.files['file'].filename:
        return jsonify({'error': 'No file uploaded'}), 400
    f = request.files['file']
    fp = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
    f.save(fp)
    try:
        fs   = float(request.form.get('fs', 1.0))
        wsize = int(request.form.get('window_size', 256))
        hop   = int(request.form.get('hop_size', 128))
        win   = request.form.get('window', 'hann').lower()
        kbeta = float(request.form.get('kaiser_beta', 8.6))  # Kaiser shape param
        x, afs = load_signal(fp)
        if afs and afs != 1.0: fs = afs
        task_id = str(uuid.uuid4())
        processing_status[task_id] = {'status': 'processing', 'progress': 0,
                                       'stage': 'Queued', 'fs': fs}
        _async_route(task_id, x, fs, _stft_worker,
                     window_size=wsize, hop_size=hop, window=win, kaiser_beta=kbeta)
        return jsonify({'task_id': task_id, 'signal_length': len(x), 'sampling_rate': fs})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _stft_worker(task_id, x, fs, window_size=256, hop_size=128, window='hann',
                  kaiser_beta=8.6):
    import plotly.graph_objects as go
    try:
        processing_status[task_id].update({'progress': 20, 'stage': 'Computing STFT/WFT…'})
        # 'gaussian' window → use WFT from filtering module (proper Gaussian STFT)
        is_wft = window.lower() in ('gaussian', 'wft')
        if is_wft:
            from fastmoda.filtering import wft
            freqs, times, Sxx = wft(x, fs, window_size=window_size,
                                     hop_size=hop_size, window='gaussian',
                                     kaiser_beta=kaiser_beta,
                                     device=DEVICE if USE_GPU else None)
        else:
            from fastmoda.filtering import wft
            freqs, times, Sxx = wft(x, fs, window_size=window_size,
                                     hop_size=hop_size, window=window,
                                     kaiser_beta=kaiser_beta,
                                     device=DEVICE if USE_GPU else None)
        processing_status[task_id].update({'progress': 60, 'stage': 'Building plot…'})
        # Downsample time axis to ≤500 cols for JSON transport
        step = max(1, Sxx.shape[1] // 500)
        Sxx_ds = 10 * np.log10(Sxx[:, ::step] + 1e-12)
        times_ds = times[::step]
        fig = go.Figure(go.Heatmap(
            x=times_ds.tolist(), y=freqs.tolist(), z=Sxx_ds.tolist(),
            colorscale='Viridis', colorbar={'title': 'dB'},
            hovertemplate='%{x:.2f}s / %{y:.2f}Hz / %{z:.1f}dB<extra></extra>'
        ))
        fig.update_layout(title='STFT Spectrogram',
                          xaxis_title='Time (s)', yaxis_title='Frequency (Hz)')
        avg = np.mean(Sxx, axis=1)
        processing_status[task_id].update({
            'status': 'complete', 'progress': 100, 'stage': 'Complete!',
            'results': {
                'stft_plot': json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder),
                'dominant_freq':    round(float(freqs[np.argmax(avg)]), 2),
                'spectral_centroid': round(float(np.sum(freqs * avg) / (np.sum(avg) + 1e-12)), 2),
                'n_freq_bins':      len(freqs),
                'n_time_windows':   len(times),
                'window_size':      window_size,
                'hop_size':         hop_size,
                'window_type':      'gaussian (WFT)' if is_wft else window,
                'gpu_used':         False,
            }
        })
    except Exception as e:
        processing_status[task_id].update({'status': 'error', 'error': str(e), 'stage': 'Error'})
        import traceback; traceback.print_exc()


@app.route('/analyze_wft', methods=['POST'])
def analyze_wft():
    """Windowed Fourier Transform with Gaussian window (alias for /analyze_stft?window=gaussian)."""
    if 'file' not in request.files or not request.files['file'].filename:
        return jsonify({'error': 'No file uploaded'}), 400
    f = request.files['file']
    fp = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
    f.save(fp)
    try:
        fs   = float(request.form.get('fs', 1.0))
        ws   = int(request.form.get('window_size', 256))
        hop  = int(request.form.get('hop_size', 128))
        x, afs = load_signal(fp)
        if afs and afs != 1.0: fs = afs
        task_id = str(uuid.uuid4())
        processing_status[task_id] = {'status': 'processing', 'progress': 0,
                                       'stage': 'Queued', 'fs': fs}
        t = Thread(target=_stft_worker,
                   args=(task_id, x, fs, ws, hop, 'gaussian'))
        t.daemon = True; t.start()
        return jsonify({'task_id': task_id, 'signal_length': len(x), 'sampling_rate': fs})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/analyze_cwt', methods=['POST'])
def analyze_cwt():
    """Continuous Wavelet Transform endpoint."""
    if 'file' not in request.files or not request.files['file'].filename:
        return jsonify({'error': 'No file uploaded'}), 400
    f = request.files['file']
    fp = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
    f.save(fp)
    try:
        fs      = float(request.form.get('fs', 1.0))
        fmin    = float(request.form.get('freq_min', 0.5))
        fmax    = float(request.form.get('freq_max', fs / 2))
        n_freqs = int(request.form.get('n_freqs', 50))
        wavelet    = request.form.get('wavelet', 'lognorm')
        n_cyc      = float(request.form.get('n_cycles', 6.0))
        nv         = request.form.get('nv', None)          # voices per octave
        padding    = request.form.get('padding', 'symmetric')
        cut_edges  = request.form.get('cut_edges', 'false').lower() == 'true'
        plot_type  = request.form.get('plot_type', 'amplitude').lower()
        x, afs     = load_signal(fp)
        if afs and afs != 1.0: fs = afs
        task_id = str(uuid.uuid4())
        processing_status[task_id] = {'status': 'processing', 'progress': 0,
                                       'stage': 'Queued', 'fs': fs}
        _async_route(task_id, x, fs, _cwt_worker,
                     freq_min=fmin, freq_max=fmax, n_freqs=n_freqs,
                     wavelet=wavelet, n_cycles=n_cyc, nv=nv,
                     padding=padding, cut_edges=cut_edges, plot_type=plot_type)
        return jsonify({'task_id': task_id, 'signal_length': len(x), 'sampling_rate': fs})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _cwt_worker(task_id, x, fs, freq_min=0.5, freq_max=None, n_freqs=50,
                 wavelet='lognorm', n_cycles=6.0, nv=None,
                 padding='symmetric', cut_edges=False, plot_type='amplitude'):
    import plotly.graph_objects as go
    try:
        if freq_max is None:
            freq_max = fs / 2
        processing_status[task_id].update({'progress': 20,
            'stage': f'Computing CWT ({wavelet}, {padding} padding)…'})
        from fastmoda.ridge_gpu import cwt_complex, nv_to_freqs

        # nv (voices per octave) overrides n_freqs when given
        if nv is not None:
            freqs = nv_to_freqs(freq_min, freq_max, int(nv))
        else:
            freqs = np.logspace(np.log10(freq_min), np.log10(freq_max), n_freqs)

        times = np.arange(len(x)) / fs
        cwt_c = cwt_complex(x, freqs, fs, wavelet=wavelet, n_cycles=n_cycles,
                             padding=padding, cut_edges=cut_edges,
                             device=DEVICE if USE_GPU else None)
        Cwt   = np.abs(cwt_c)  # amplitude (NaN where CutEdges masks)
        processing_status[task_id].update({'progress': 65, 'stage': 'Building plot…'})
        step = max(1, Cwt.shape[1] // 500)
        Cwt_ds = Cwt[:, ::step]
        if plot_type == 'power':
            Z_db = 10 * np.log10(Cwt_ds ** 2 + 1e-12)  # power dB
        else:
            Z_db = 20 * np.log10(Cwt_ds + 1e-12)       # amplitude dB
        times_ds = times[::step]
        fig = go.Figure(go.Heatmap(
            x=times_ds.tolist(), y=freqs.tolist(), z=Z_db.tolist(),
            colorscale='Jet', colorbar={'title': 'dB'},
            hovertemplate='%{x:.2f}s / %{y:.2f}Hz / %{z:.1f}dB<extra></extra>'
        ))
        fig.update_layout(title=f'Continuous Wavelet Transform ({plot_type.capitalize()})',
                          xaxis_title='Time (s)', yaxis_title='Frequency (Hz)',
                          yaxis_type='log')
        ridge = freqs[np.argmax(Cwt, axis=0)]
        processing_status[task_id].update({
            'status': 'complete', 'progress': 100, 'stage': 'Complete!',
            'results': {
                'cwt_plot':          json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder),
                'mean_ridge_freq':   round(float(np.mean(ridge)), 2),
                'dominant_freq':     round(float(np.median(ridge)), 2),
                'n_freq_bins':       len(freqs),
                'freq_min':          freq_min,
                'freq_max':          freq_max,
                'wavelet':           wavelet,
                'n_cycles':          n_cycles,
                'padding':           padding,
                'cut_edges':         cut_edges,
                'plot_type':         plot_type,
                'n_freq_bins':       len(freqs),
                'gpu_used':          USE_GPU,
            }
        })
    except Exception as e:
        processing_status[task_id].update({'status': 'error', 'error': str(e), 'stage': 'Error'})
        import traceback; traceback.print_exc()


@app.route('/analyze_hilbert', methods=['POST'])
def analyze_hilbert():
    """Hilbert transform — instantaneous amplitude, phase, and frequency."""
    if 'file' not in request.files or not request.files['file'].filename:
        return jsonify({'error': 'No file uploaded'}), 400
    f = request.files['file']
    fp = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
    f.save(fp)
    try:
        fs = float(request.form.get('fs', 1.0))
        x, afs = load_signal(fp)
        if afs and afs != 1.0: fs = afs
        task_id = str(uuid.uuid4())
        processing_status[task_id] = {'status': 'processing', 'progress': 0,
                                       'stage': 'Queued', 'fs': fs}
        _async_route(task_id, x, fs, _hilbert_worker)
        return jsonify({'task_id': task_id, 'signal_length': len(x), 'sampling_rate': fs})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _hilbert_worker(task_id, x, fs):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    try:
        processing_status[task_id].update({'progress': 20, 'stage': 'Hilbert transform…'})
        from fastmoda.analysis_gpu import compute_instantaneous_phase_gpu
        phase_data = compute_instantaneous_phase_gpu(x, fs=fs)
        amp   = phase_data['amplitude']
        phase = phase_data['phase']
        ifreq = phase_data['frequency']
        t     = np.arange(len(x)) / fs

        processing_status[task_id].update({'progress': 60, 'stage': 'Building plots…'})
        step = max(1, len(t) // 1000)
        fig = make_subplots(rows=3, cols=1,
            subplot_titles=('Instantaneous Amplitude', 'Instantaneous Phase',
                            'Instantaneous Frequency'),
            vertical_spacing=0.08)
        fig.add_trace(go.Scatter(x=t[::step].tolist(), y=amp[::step].tolist(),
                                  mode='lines', line={'color': 'cyan', 'width': 1}), row=1, col=1)
        fig.add_trace(go.Scatter(x=t[::step].tolist(), y=phase[::step].tolist(),
                                  mode='lines', line={'color': 'orange', 'width': 1}), row=2, col=1)
        fig.add_trace(go.Scatter(x=t[::step].tolist(), y=ifreq[::step].tolist(),
                                  mode='lines', line={'color': 'lime', 'width': 1}), row=3, col=1)
        fig.update_xaxes(title_text='Time (s)', row=3, col=1)
        fig.update_layout(height=900, showlegend=False, title='Hilbert Analysis')

        processing_status[task_id].update({
            'status': 'complete', 'progress': 100, 'stage': 'Complete!',
            'results': {
                'hilbert_plot':      json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder),
                'mean_amplitude':    round(float(np.mean(amp)), 4),
                'std_amplitude':     round(float(np.std(amp)), 4),
                'mean_inst_freq':    round(float(np.mean(ifreq[(ifreq > 0) & (ifreq < fs / 2)])), 2),
                'phase_range_rad':   round(float(np.max(phase) - np.min(phase)), 3),
                'gpu_used':          False,
            }
        })
    except Exception as e:
        processing_status[task_id].update({'status': 'error', 'error': str(e), 'stage': 'Error'})
        import traceback; traceback.print_exc()


@app.route('/analyze_surrogates', methods=['POST'])
def analyze_surrogates():
    """Statistical significance testing via surrogate signals."""
    if 'file' not in request.files or not request.files['file'].filename:
        return jsonify({'error': 'No file uploaded'}), 400
    f = request.files['file']
    fp = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
    f.save(fp)
    try:
        fs           = float(request.form.get('fs', 1.0))
        test_type    = request.form.get('test_type', 'spectral')
        n_surr       = int(request.form.get('n_surrogates', 19))
        surr_method  = request.form.get('surrogate_method', 'phase_randomization')
        target_freq  = request.form.get('target_freq')
        if target_freq:
            target_freq = float(target_freq)
        x, afs = load_signal(fp)
        if afs and afs != 1.0: fs = afs
        task_id = str(uuid.uuid4())
        processing_status[task_id] = {'status': 'processing', 'progress': 0,
                                       'stage': 'Queued', 'fs': fs}
        _async_route(task_id, x, fs, _surrogates_worker,
                     test_type=test_type, n_surrogates=n_surr,
                     surrogate_method=surr_method, target_freq=target_freq)
        return jsonify({'task_id': task_id, 'signal_length': len(x),
                        'sampling_rate': fs, 'test_type': test_type})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _surrogates_worker(task_id, x, fs, test_type='spectral', n_surrogates=19,
                        surrogate_method='phase_randomization', target_freq=None):
    import plotly.graph_objects as go
    try:
        processing_status[task_id].update({'progress': 10,
            'stage': f'Generating {n_surrogates} {surrogate_method} surrogates…'})
        from fastmoda.surrogates import (
            surrogate_test_spectral, surrogate_test_changepoints,
            surrogate_test_phase_coherence, surrogate_test_bispectrum,
            surrogate_test,
        )

        # Surrogate methods: cpp, rp, aaft, iaaft, wiaaft
        def _make_surrogate_fn(method):
            """Return a surrogate generator that works on CPU or GPU."""
            if method == 'cpp':
                from fastmoda.filtering import cpp_surrogates
                def gen(sig, n, **kw): return cpp_surrogates(sig, n, fs=fs)
                return gen
            if method == 'rp':
                from fastmoda.filtering import rp_surrogates
                def gen(sig, n, **kw): return rp_surrogates(sig, n)
                return gen
            if method == 'aaft':
                from fastmoda.filtering import aaft_surrogates
                def gen(sig, n, **kw): return aaft_surrogates(sig, n)
                return gen
            if method in ('iaaft2', 'iaaft_cpu'):
                from fastmoda.filtering import iaaft_surrogates
                def gen(sig, n, **kw): return iaaft_surrogates(sig, n)
                return gen
            if method == 'wiaaft':
                try:
                    from fastmoda.surrogates_gpu import wiaaft_surrogate_gpu
                    import torch as _torch
                    def gen(sig, n, **kw):
                        t = _torch.from_numpy(sig.astype(np.float32))
                        surrs = [wiaaft_surrogate_gpu(t).numpy() for _ in range(n)]
                        return np.stack(surrs)
                    return gen
                except (ImportError, Exception):
                    pass   # fall through to phase_randomization
            return None    # use default in surrogate_test

        gen_fn = _make_surrogate_fn(surrogate_method)

        if test_type == 'spectral':
            if gen_fn is not None:
                def _obs(sig):
                    sp = np.abs(np.fft.rfft(sig))**2
                    fidx = np.argmax(sp[1:]) + 1
                    return float(sp[fidx])
                surrs = gen_fn(x, n_surrogates)
                surr_vals = np.array([_obs(s) for s in surrs])
                from fastmoda.surrogates import compute_surrogate_statistics
                stats = compute_surrogate_statistics(_obs(x), surr_vals)
                stats['n_surrogates'] = n_surrogates
                stats['surrogate_method'] = surrogate_method
                stats['surrogate_values'] = surr_vals.tolist()
            else:
                stats = surrogate_test_spectral(x, fs=fs, target_freq=target_freq,
                                                n_surrogates=n_surrogates)
        elif test_type == 'changepoints':
            stats = surrogate_test_changepoints(x, n_surrogates=n_surrogates)
        elif test_type == 'phase_coherence':
            stats = surrogate_test_phase_coherence(x, n_surrogates=n_surrogates)
        elif test_type == 'bispectrum':
            stats = surrogate_test_bispectrum(x, fs=fs, n_surrogates=n_surrogates)
        else:
            raise ValueError(f'Unknown test_type: {test_type}')

        processing_status[task_id].update({'progress': 80, 'stage': 'Building distribution plot…'})
        surr_vals = stats.get('surrogate_values', [])
        fig = go.Figure()
        if surr_vals:
            fig.add_trace(go.Histogram(x=surr_vals, name='Surrogates',
                                        marker_color='steelblue', opacity=0.7))
            fig.add_vline(x=stats['observed'], line_color='red', line_dash='dash',
                          annotation_text='Observed')
            if 'ci_95' in stats and len(stats['ci_95']) == 2:
                fig.add_vrect(x0=stats['ci_95'][0], x1=stats['ci_95'][1],
                              fillcolor='green', opacity=0.1, line_width=0,
                              annotation_text='95% CI')
        fig.update_layout(title=f'Surrogate Distribution — {test_type}',
                          xaxis_title='Statistic', yaxis_title='Count')

        def _safe(v):
            try: return round(float(v), 4)
            except: return v

        processing_status[task_id].update({
            'status': 'complete', 'progress': 100, 'stage': 'Complete!',
            'results': {
                'surrogate_plot':    json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder),
                'test_type':         test_type,
                'surrogate_method':  surrogate_method,
                'n_surrogates':      stats.get('n_surrogates', n_surrogates),
                'observed':          _safe(stats.get('observed', 0)),
                'surrogate_mean':    _safe(stats.get('surrogate_mean', 0)),
                'surrogate_std':     _safe(stats.get('surrogate_std', 0)),
                'z_score':           _safe(stats.get('z_score', 0)),
                'p_value':           _safe(stats.get('p_value', 1.0)),
                'percentile':        _safe(stats.get('percentile', 50)),
                'significant_95':    bool(stats.get('significant_95', False)),
                'significant_99':    bool(stats.get('significant_99', False)),
                'gpu_used':          False,
            }
        })
    except Exception as e:
        processing_status[task_id].update({'status': 'error', 'error': str(e), 'stage': 'Error'})
        import traceback; traceback.print_exc()


@app.route('/analyze_features', methods=['POST'])
def analyze_features():
    """Extract numerical ML feature vector from signal."""
    if 'file' not in request.files or not request.files['file'].filename:
        return jsonify({'error': 'No file uploaded'}), 400
    f = request.files['file']
    fp = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
    f.save(fp)
    try:
        fs = float(request.form.get('fs', 1.0))
        analyses_raw = request.form.get('analyses', 'spectral,phase')
        analyses = [a.strip() for a in analyses_raw.split(',')]
        x, afs = load_signal(fp)
        if afs and afs != 1.0: fs = afs
        task_id = str(uuid.uuid4())
        processing_status[task_id] = {'status': 'processing', 'progress': 0,
                                       'stage': 'Queued', 'fs': fs}
        _async_route(task_id, x, fs, _features_worker, analyses=analyses)
        return jsonify({'task_id': task_id, 'signal_length': len(x),
                        'sampling_rate': fs, 'analyses': analyses})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _features_worker(task_id, x, fs, analyses=None):
    if analyses is None:
        analyses = ['spectral', 'phase']
    try:
        from fastmoda.feature_extraction import (
            extract_spectral_features, extract_phase_features,
        )
        processing_status[task_id].update({'progress': 10, 'stage': 'Extracting features…'})

        feature_vector = []
        feature_names  = []

        bands = [(0.5, 4, 'delta'), (4, 8, 'theta'), (8, 12, 'alpha'),
                 (12, 30, 'beta'), (30, 100, 'gamma')]

        if 'spectral' in analyses:
            processing_status[task_id].update({'progress': 30, 'stage': 'Spectral features…'})
            from fastmoda import sliding_fft, compute_band_powers
            freqs, times, Sxx = sliding_fft(x, fs=fs)
            _, names = compute_band_powers(Sxx, freqs, [(lo, hi, n) for lo, hi, n in bands])
            from numpy import array as npa
            cps = npa([])
            feat = extract_spectral_features(freqs, Sxx, times, cps, bands)
            for k, v in feat.items():
                feature_names.append(f'spectral_{k}')
                feature_vector.append(float(v) if not isinstance(v, str) else 0.0)

        if 'phase' in analyses:
            processing_status[task_id].update({'progress': 60, 'stage': 'Phase features…'})
            from fastmoda.analysis_gpu import compute_instantaneous_phase_gpu
            from fastmoda.feature_extraction import extract_phase_features
            pd_data = compute_instantaneous_phase_gpu(x, fs=fs)
            # signature: extract_phase_features(phase, amplitude, frequency, fs)
            feat = extract_phase_features(pd_data['phase'], pd_data['amplitude'],
                                          pd_data['frequency'], fs)
            for k, v in feat.items():
                feature_names.append(f'phase_{k}')
                feature_vector.append(float(v) if not isinstance(v, (int, float)) else float(v))

        processing_status[task_id].update({
            'status': 'complete', 'progress': 100, 'stage': 'Complete!',
            'results': {
                'feature_vector':  feature_vector,
                'feature_names':   feature_names,
                'n_features':      len(feature_vector),
                'analyses_run':    analyses,
                'gpu_used':        False,
            }
        })
    except Exception as e:
        processing_status[task_id].update({'status': 'error', 'error': str(e), 'stage': 'Error'})
        import traceback; traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════════════════
# NEW MODA-COMPLETE ENDPOINTS: biphase, bispectrum4, coupling functions
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/analyze_syncmap', methods=['POST'])
def analyze_syncmap():
    """Synchronisation map analysis from coupling function coefficients."""
    # Accepts the same 2-signal input as /analyze_coupling
    files = request.files.getlist('files')
    if len(files) < 2:
        return jsonify({'error': '2 phase time-series required'}), 400
    fp1 = os.path.join(app.config['UPLOAD_FOLDER'], files[0].filename)
    fp2 = os.path.join(app.config['UPLOAD_FOLDER'], files[1].filename)
    files[0].save(fp1); files[1].save(fp2)
    try:
        fs      = float(request.form.get('fs', 1.0))
        bn      = int(request.form.get('bn', 3))
        win_s   = float(request.form.get('win_s', 40.0))
        f_b1    = (float(request.form.get('band1_low', 0.5)),
                   float(request.form.get('band1_high', 2.0)))
        f_b2    = (float(request.form.get('band2_low', 0.5)),
                   float(request.form.get('band2_high', 2.0)))
        x1, afs = load_signal(fp1)
        if afs and afs != 1.0: fs = afs
        x2, _   = load_signal(fp2)
        task_id = str(uuid.uuid4())
        processing_status[task_id] = {'status': 'processing', 'progress': 0,
                                       'stage': 'Queued'}
        t = Thread(target=_syncmap_worker,
                   args=(task_id, x1, x2, fs, bn, win_s, f_b1, f_b2))
        t.daemon = True; t.start()
        return jsonify({'task_id': task_id, 'signal_length': len(x1)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _syncmap_worker(task_id, x1, x2, fs, bn, win_s, band1, band2):
    import plotly.graph_objects as go
    try:
        processing_status[task_id].update({'progress': 15, 'stage': 'Estimating coupling…'})
        from scipy.signal import butter, filtfilt, hilbert
        from fastmoda.coupling_gpu import estimate_coupling_functions, sync_map

        def phase_of(x, lo, hi):
            nyq = fs / 2
            b, a = butter(4, [max(0.001, lo/nyq), min(0.999, hi/nyq)], 'band')
            return np.angle(hilbert(filtfilt(b, a, x)))

        ph1 = phase_of(x1, *band1)
        ph2 = phase_of(x2, *band2)

        processing_status[task_id].update({'progress': 40, 'stage': 'Coupling functions…'})
        cf = estimate_coupling_functions(ph1, ph2, fs, bn=bn, win_s=win_s,
                                          device=DEVICE if USE_GPU else None)

        processing_status[task_id].update({'progress': 70, 'stage': 'Sync map…'})
        sm = sync_map(cf['c1_mean'], cf['c2_mean'], bn)

        fig = go.Figure()
        phi_d = sm['phi_diff_grid']
        prof  = sm['coupling_profile']
        fig.add_trace(go.Scatter(x=phi_d, y=prof,
            mode='lines', line={'color': 'cyan', 'width': 1.5}, name='dψ/dt'))
        fig.add_hline(y=0, line={'color': 'white', 'dash': 'dot'})
        for fp in sm['fixed_points_rad']:
            fig.add_vline(x=fp, line={'color': 'red', 'dash': 'dash'},
                          annotation_text='sync')
        fig.update_layout(title='Synchronisation Map', xaxis_title='Δφ (rad)',
                          yaxis_title='dΔφ/dt')

        processing_status[task_id].update({
            'status': 'complete', 'progress': 100, 'stage': 'Complete!',
            'results': {
                'syncmap_plot':          json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder),
                'sync_index':            sm['sync_index'],
                'is_synchronised':       sm['is_synchronised'],
                'n_stable_fixed_points': sm['n_stable_fixed_points'],
                'mean_fixed_point_deg':  sm.get('mean_fixed_point_deg'),
                'mean_direction':        round(float(np.mean(cf['direction'])), 4),
                'mean_cpl1':             round(float(np.mean(cf['cpl1'])), 4),
                'mean_cpl2':             round(float(np.mean(cf['cpl2'])), 4),
                'gpu_used':              cf['gpu_used'],
            }
        })
    except Exception as e:
        processing_status[task_id].update({'status': 'error', 'error': str(e), 'stage': 'Error'})
        import traceback; traceback.print_exc()


@app.route('/analyze_group', methods=['POST'])
def analyze_group():
    """
    Statistical comparison of two groups of signals at each frequency.
    Files named 'g1' belong to group 1, files named 'g2' to group 2.
    """
    files_g1 = request.files.getlist('g1')
    files_g2 = request.files.getlist('g2')
    if len(files_g1) < 2 or len(files_g2) < 2:
        return jsonify({'error': 'At least 2 signals required per group'}), 400
    fps1, fps2 = [], []
    for f in files_g1:
        fp = os.path.join(app.config['UPLOAD_FOLDER'], f.filename); f.save(fp); fps1.append(fp)
    for f in files_g2:
        fp = os.path.join(app.config['UPLOAD_FOLDER'], f.filename); f.save(fp); fps2.append(fp)
    try:
        fs      = float(request.form.get('fs', 1.0))
        fmin    = float(request.form.get('freq_min', 0.5))
        fmax    = float(request.form.get('freq_max', 0))
        n_freqs = int(request.form.get('n_freqs', 50))
        wavelet = request.form.get('wavelet', 'lognorm')
        task_id = str(uuid.uuid4())
        processing_status[task_id] = {'status': 'processing', 'progress': 0,
                                       'stage': 'Queued'}
        t = Thread(target=_group_worker,
                   args=(task_id, fps1, fps2, fs, fmin, fmax or fs/2,
                         n_freqs, wavelet))
        t.daemon = True; t.start()
        return jsonify({'task_id': task_id,
                        'n_g1': len(fps1), 'n_g2': len(fps2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _group_worker(task_id, fps1, fps2, fs, fmin, fmax, n_freqs, wavelet):
    import plotly.graph_objects as go
    try:
        from fastmoda.ridge_gpu import cwt_complex
        from scipy.stats import ranksums

        freqs = np.logspace(np.log10(fmin), np.log10(fmax), n_freqs)
        NF = len(freqs)

        def mean_power(fps):
            pows = []
            for i, fp in enumerate(fps):
                processing_status[task_id]['progress'] = 10 + 40 * i // len(fps)
                x, afs = load_signal(fp)
                if afs and afs != 1.0: nonlocal_fs = afs
                else: nonlocal_fs = fs
                cwt_c = cwt_complex(x, freqs, nonlocal_fs, wavelet=wavelet)
                pows.append(np.nanmean(np.abs(cwt_c) ** 2, axis=1))  # (NF,)
            return np.array(pows)  # (n_signals, NF)

        processing_status[task_id].update({'progress': 5, 'stage': 'Computing group 1 power…'})
        pow_g1 = mean_power(fps1)  # (N1, NF)
        processing_status[task_id].update({'progress': 50, 'stage': 'Computing group 2 power…'})
        pow_g2 = mean_power(fps2)  # (N2, NF)

        processing_status[task_id].update({'progress': 80, 'stage': 'Wilcoxon rank-sum tests…'})
        # Test at each frequency — O(NF) loop but NF is small
        p_vals = np.array([ranksums(pow_g1[:, f], pow_g2[:, f]).pvalue
                           for f in range(NF)])

        sig_05 = (p_vals < 0.05).astype(int).tolist()
        sig_01 = (p_vals < 0.01).astype(int).tolist()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=freqs.tolist(), y=pow_g1.mean(axis=0).tolist(),
            mode='lines', name='Group 1', line={'color': 'cyan'}))
        fig.add_trace(go.Scatter(x=freqs.tolist(), y=pow_g2.mean(axis=0).tolist(),
            mode='lines', name='Group 2', line={'color': 'orange'}))
        # Shade significant regions
        for f_idx, sig in enumerate(sig_05):
            if sig:
                fig.add_vrect(x0=freqs[f_idx]*0.98, x1=freqs[f_idx]*1.02,
                              fillcolor='rgba(255,255,0,0.15)', line_width=0)
        fig.update_layout(title='Group Comparison (yellow = p<0.05)',
                          xaxis_title='Frequency (Hz)', yaxis_title='Mean Power',
                          xaxis_type='log')

        processing_status[task_id].update({
            'status': 'complete', 'progress': 100, 'stage': 'Complete!',
            'results': {
                'group_plot':       json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder),
                'n_significant_05': int(np.sum(p_vals < 0.05)),
                'n_significant_01': int(np.sum(p_vals < 0.01)),
                'min_pvalue':       round(float(np.min(p_vals)), 4),
                'peak_diff_freq':   round(float(freqs[np.argmax(
                    np.abs(pow_g1.mean(0) - pow_g2.mean(0)))]), 2),
                'n_g1': len(fps1), 'n_g2': len(fps2),
                'wavelet': wavelet,
            }
        })
    except Exception as e:
        processing_status[task_id].update({'status': 'error', 'error': str(e), 'stage': 'Error'})
        import traceback; traceback.print_exc()


@app.route('/analyze_biphase', methods=['POST'])
def analyze_biphase():
    """Biphase & biamplitude time series at a specific frequency pair (MODA biphaseWavNew)."""
    files = request.files.getlist('files')
    if len(files) < 1:
        return jsonify({'error': 'At least 1 file required (2 for cross-biphase)'}), 400
    fp1 = os.path.join(app.config['UPLOAD_FOLDER'], files[0].filename)
    files[0].save(fp1)
    fp2 = fp1
    if len(files) >= 2:
        fp2 = os.path.join(app.config['UPLOAD_FOLDER'], files[1].filename)
        files[1].save(fp2)
    try:
        fs      = float(request.form.get('fs', 1.0))
        f1      = float(request.form.get('f1', 6.0))
        f2      = float(request.form.get('f2', 10.0))
        wavelet = request.form.get('wavelet', 'lognorm')
        n_cyc   = float(request.form.get('n_cycles', 6.0))
        x1, afs = load_signal(fp1)
        if afs and afs != 1.0: fs = afs
        x2, _   = load_signal(fp2)
        task_id = str(uuid.uuid4())
        processing_status[task_id] = {'status': 'processing', 'progress': 0,
                                       'stage': 'Queued', 'fs': fs}
        t = Thread(target=_biphase_worker,
                   args=(task_id, x1, x2, fs, f1, f2, wavelet, n_cyc))
        t.daemon = True; t.start()
        return jsonify({'task_id': task_id, 'f1': f1, 'f2': f2, 'f3': f1+f2,
                        'signal_length': len(x1), 'sampling_rate': fs})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _biphase_worker(task_id, x1, x2, fs, f1, f2, wavelet, n_cycles):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    try:
        processing_status[task_id].update({'progress': 20, 'stage': 'Computing biphase…'})
        from fastmoda.biphase_gpu import biphase_timeseries
        result = biphase_timeseries(x1, x2, fs, f1, f2,
                                     wavelet=wavelet, n_cycles=n_cycles,
                                     device=DEVICE if USE_GPU else None)

        processing_status[task_id].update({'progress': 70, 'stage': 'Building plots…'})
        t_ax  = result['time']
        step  = max(1, len(t_ax) // 800)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
            subplot_titles=(f'Biamplitude — ({f1} Hz, {f2} Hz)',
                            f'Biphase — ({f1} Hz, {f2} Hz)'),
            vertical_spacing=0.1)
        fig.add_trace(go.Scatter(x=t_ax[::step].tolist(),
                                  y=result['biamp'][::step].tolist(),
                                  mode='lines', line={'color': 'cyan', 'width': 1}),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=t_ax[::step].tolist(),
                                  y=np.degrees(result['biphase'][::step]).tolist(),
                                  mode='lines', line={'color': 'orange', 'width': 1}),
                      row=2, col=1)
        fig.update_yaxes(title_text='Biamplitude', row=1, col=1)
        fig.update_yaxes(title_text='Biphase (°)',  row=2, col=1)
        fig.update_xaxes(title_text='Time (s)', row=2, col=1)
        fig.update_layout(height=500, showlegend=False)

        processing_status[task_id].update({
            'status': 'complete', 'progress': 100, 'stage': 'Complete!',
            'results': {
                'biphase_plot':      json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder),
                'mean_biamp':        round(float(np.mean(result['biamp'])), 4),
                'mean_biphase_deg':  round(float(np.degrees(np.mean(result['biphase']))), 2),
                'std_biphase_deg':   round(float(np.degrees(np.std(result['biphase']))), 2),
                'f1': f1, 'f2': f2, 'f3': f1 + f2,
                'wavelet': wavelet, 'gpu_used': result.get('gpu_used', False),
            }
        })
    except Exception as e:
        processing_status[task_id].update({'status': 'error', 'error': str(e), 'stage': 'Error'})
        import traceback; traceback.print_exc()


@app.route('/analyze_bispectrum4', methods=['POST'])
def analyze_bispectrum4():
    """Four-way cross-bispectrum: b111, b222, b122, b211 (MODA bispecWavNew)."""
    files = request.files.getlist('files')
    if len(files) < 2:
        return jsonify({'error': 'Exactly 2 signals required for 4-way bispectrum'}), 400
    fp1 = os.path.join(app.config['UPLOAD_FOLDER'], files[0].filename)
    fp2 = os.path.join(app.config['UPLOAD_FOLDER'], files[1].filename)
    files[0].save(fp1); files[1].save(fp2)
    try:
        fs   = float(request.form.get('fs', 1.0))
        nfft = int(request.form.get('nfft', 256))
        x1, afs = load_signal(fp1)
        if afs and afs != 1.0: fs = afs
        x2, _ = load_signal(fp2)
        task_id = str(uuid.uuid4())
        processing_status[task_id] = {'status': 'processing', 'progress': 0,
                                       'stage': 'Queued', 'fs': fs}
        t = Thread(target=_bispec4_worker, args=(task_id, x1, x2, fs, nfft))
        t.daemon = True; t.start()
        return jsonify({'task_id': task_id, 'signal_length': len(x1), 'sampling_rate': fs})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _bispec4_worker(task_id, x1, x2, fs, nfft):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    try:
        processing_status[task_id].update({'progress': 20, 'stage': 'Computing 4-way bispectrum…'})
        from fastmoda.biphase_gpu import bispectrum4
        res = bispectrum4(x1, x2, fs, nfft=nfft,
                          device=DEVICE if USE_GPU else None)

        processing_status[task_id].update({'progress': 65, 'stage': 'Building plots…'})
        freqs = res['frequencies']
        # Limit each heatmap to ≤50×50 to keep JSON transportable
        fstep = max(1, len(freqs) // 50)
        fr    = freqs[::fstep].tolist()

        fig = make_subplots(rows=2, cols=2,
            subplot_titles=('b111 (auto x1)', 'b222 (auto x2)',
                            'b122 (cross 1→2,2)', 'b211 (cross 2→1,1)'),
            vertical_spacing=0.12, horizontal_spacing=0.08)

        for row, col, key in (
            (1,1,'biamp111'), (1,2,'biamp222'),
            (2,1,'biamp122'), (2,2,'biamp211'),
        ):
            mat = 10 * np.log10(res[key][::fstep, ::fstep] + 1e-12)
            fig.add_trace(go.Heatmap(x=fr, y=fr, z=mat.tolist(),
                colorscale='Hot', showscale=False,
                hovertemplate='f1:%{x:.1f}Hz f2:%{y:.1f}Hz<extra></extra>'),
                row=row, col=col)

        fig.update_layout(height=700, title='4-Way Cross-Bispectrum')

        processing_status[task_id].update({
            'status': 'complete', 'progress': 100, 'stage': 'Complete!',
            'results': {
                    'coupling_b111':      round(float(np.nan_to_num(np.mean(res['biamp111']))), 4),
                'coupling_b222':      round(float(np.nan_to_num(np.mean(res['biamp222']))), 4),
                'coupling_b122':      round(float(np.nan_to_num(np.mean(res['biamp122']))), 4),
                'coupling_b211':      round(float(np.nan_to_num(np.mean(res['biamp211']))), 4),
                'peak_b122':          round(float(np.nan_to_num(np.max(res['biamp122']))), 4),
                'peak_b211':          round(float(np.nan_to_num(np.max(res['biamp211']))), 4),
                'dominant_coupling':  'b122' if (np.nan_to_num(np.mean(res['biamp122']))
                                                 > np.nan_to_num(np.mean(res['biamp211']))) else 'b211',
                'gpu_used':           res['gpu_used'],
            }
        })
    except Exception as e:
        processing_status[task_id].update({'status': 'error', 'error': str(e), 'stage': 'Error'})
        import traceback; traceback.print_exc()


@app.route('/analyze_coupling', methods=['POST'])
def analyze_coupling():
    """Coupling function estimation via sliding-window OLS (MODA bayes_main + CFprint)."""
    files = request.files.getlist('files')
    if len(files) < 2:
        return jsonify({'error': 'Exactly 2 phase time-series required'}), 400
    fp1 = os.path.join(app.config['UPLOAD_FOLDER'], files[0].filename)
    fp2 = os.path.join(app.config['UPLOAD_FOLDER'], files[1].filename)
    files[0].save(fp1); files[1].save(fp2)
    try:
        fs      = float(request.form.get('fs', 1.0))
        bn      = int(request.form.get('bn', 3))
        win_s   = float(request.form.get('win_s', 40.0))
        overlap = float(request.form.get('overlap', 0.5))
        f_band1 = (float(request.form.get('band1_low', 0.5)),
                   float(request.form.get('band1_high', 2.0)))
        f_band2 = (float(request.form.get('band2_low', 0.5)),
                   float(request.form.get('band2_high', 2.0)))
        x1, afs = load_signal(fp1)
        if afs and afs != 1.0: fs = afs
        x2, _   = load_signal(fp2)
        task_id = str(uuid.uuid4())
        processing_status[task_id] = {'status': 'processing', 'progress': 0,
                                       'stage': 'Queued', 'fs': fs}
        t = Thread(target=_coupling_worker,
                   args=(task_id, x1, x2, fs, bn, win_s, overlap, f_band1, f_band2))
        t.daemon = True; t.start()
        return jsonify({'task_id': task_id, 'signal_length': len(x1),
                        'sampling_rate': fs, 'bn': bn, 'win_s': win_s})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _coupling_worker(task_id, x1, x2, fs, bn, win_s, overlap, band1, band2):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    try:
        processing_status[task_id].update({'progress': 10, 'stage': 'Extracting phases…'})
        from scipy.signal import butter, filtfilt, hilbert
        from fastmoda.coupling_gpu import estimate_coupling_functions

        def phase_of(x, lo, hi):
            nyq = fs / 2
            b, a = butter(4, [max(0.001, lo/nyq), min(0.999, hi/nyq)], 'band')
            return np.angle(hilbert(filtfilt(b, a, x)))

        ph1 = phase_of(x1, *band1)
        ph2 = phase_of(x2, *band2)

        processing_status[task_id].update({'progress': 30, 'stage': 'Estimating coupling functions…'})
        result = estimate_coupling_functions(ph1, ph2, fs, bn=bn,
                                              win_s=win_s, overlap=overlap,
                                              device=DEVICE if USE_GPU else None)

        processing_status[task_id].update({'progress': 75, 'stage': 'Building plots…'})
        phi_g = result['phi_grid'].tolist()

        fig = make_subplots(rows=1, cols=2,
            subplot_titles=('q21: coupling 2→1', 'q12: coupling 1→2'),
            horizontal_spacing=0.1)
        for col, key in enumerate(('q21', 'q12'), 1):
            fig.add_trace(go.Heatmap(x=phi_g, y=phi_g, z=result[key].tolist(),
                colorscale='RdBu', zmid=0,
                colorbar={'title': 'q', 'x': 0.48 if col==1 else 1.0,
                          'len': 0.9},
                hovertemplate='φ1:%{x:.2f} φ2:%{y:.2f}<extra></extra>'),
                row=1, col=col)
        fig.update_xaxes(title_text='φ1 (rad)')
        fig.update_yaxes(title_text='φ2 (rad)', row=1, col=1)
        fig.update_layout(height=450)

        # Direction time series
        fig_dir = go.Figure()
        fig_dir.add_trace(go.Scatter(x=result['times'].tolist(),
                                      y=result['direction'].tolist(),
                                      mode='lines', line={'color': 'gold'}))
        fig_dir.add_hline(y=0, line={'color': 'gray', 'dash': 'dot'})
        fig_dir.update_layout(title='Coupling Direction (1=2→1, -1=1→2)',
                               xaxis_title='Time (s)', yaxis_title='Direction',
                               yaxis_range=[-1.1, 1.1])

        processing_status[task_id].update({
            'status': 'complete', 'progress': 100, 'stage': 'Complete!',
            'results': {
                'coupling_plot':     json.dumps(fig,     cls=plotly.utils.PlotlyJSONEncoder),
                'direction_plot':    json.dumps(fig_dir, cls=plotly.utils.PlotlyJSONEncoder),
                'mean_cpl1':         round(float(np.mean(result['cpl1'])), 4),
                'mean_cpl2':         round(float(np.mean(result['cpl2'])), 4),
                'mean_direction':    round(float(np.mean(result['direction'])), 4),
                'cpl1_se':           round(float(result.get('cpl1_se', 0.0)), 4),
                'cpl2_se':           round(float(result.get('cpl2_se', 0.0)), 4),
                'direction_se':      round(float(result.get('dir_se', 0.0)), 4),
                'n_windows':         len(result['times']),
                'bn':                bn,
                'gpu_used':          result['gpu_used'],
            }
        })
    except Exception as e:
        processing_status[task_id].update({'status': 'error', 'error': str(e), 'stage': 'Error'})
        import traceback; traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════════════════
# RIDGE EXTRACTION  +  FILTER / WFT  endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/analyze_ridge', methods=['POST'])
def analyze_ridge():
    """Ridge extraction: instantaneous frequency, amplitude, phase, reconstruction."""
    if 'file' not in request.files or not request.files['file'].filename:
        return jsonify({'error': 'No file uploaded'}), 400
    f = request.files['file']
    fp = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
    f.save(fp)
    try:
        fs         = float(request.form.get('fs', 1.0))
        fmin       = float(request.form.get('freq_min', 0.5))
        fmax       = float(request.form.get('freq_max', 0))
        n_freqs    = int(request.form.get('n_freqs', 64))
        smooth     = int(request.form.get('smooth_len', 5))
        n_cyc      = float(request.form.get('n_cycles', 6.0))
        wavelet    = request.form.get('wavelet', 'lognorm')
        cut_edges  = request.form.get('cut_edges', 'true').lower() == 'true'
        x, afs     = load_signal(fp)
        if afs and afs != 1.0: fs = afs
        if fmax <= 0: fmax = fs / 2.0
        task_id = str(uuid.uuid4())
        processing_status[task_id] = {'status': 'processing', 'progress': 0,
                                       'stage': 'Queued', 'fs': fs}
        t = Thread(target=_ridge_worker,
                   args=(task_id, x, fs, fmin, fmax, n_freqs, smooth, n_cyc,
                         wavelet, cut_edges))
        t.daemon = True; t.start()
        return jsonify({'task_id': task_id, 'signal_length': len(x), 'sampling_rate': fs})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _ridge_worker(task_id, x, fs, fmin, fmax, n_freqs, smooth_len, n_cycles,
                  wavelet='lognorm', cut_edges=True):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    try:
        processing_status[task_id].update({'progress': 15, 'stage': 'Computing CWT…'})
        from fastmoda.ridge_gpu import cwt_complex, extract_ridge, time_localized_coherence

        freqs = np.logspace(np.log10(fmin), np.log10(fmax), n_freqs)
        cwt   = cwt_complex(x, freqs, fs, wavelet=wavelet, n_cycles=n_cycles,
                            cut_edges=cut_edges,
                            device=DEVICE if USE_GPU else None)          # (NF, T)

        processing_status[task_id].update({'progress': 50, 'stage': 'Extracting ridge…'})
        ridge = extract_ridge(cwt, freqs, fs, smooth_len=smooth_len,
                              device=DEVICE if USE_GPU else None)

        processing_status[task_id].update({'progress': 75, 'stage': 'Building plots…'})
        t_ax  = np.arange(len(x)) / fs
        step  = max(1, len(t_ax) // 800)   # downsample plots to ≤800 pts

        # CWT amplitude heatmap + ridge overlay
        amp_db = 10 * np.log10(np.abs(cwt) + 1e-12)
        ds_step_t = max(1, amp_db.shape[1] // 500)
        fig_cwt = go.Figure()
        fig_cwt.add_trace(go.Heatmap(
            x=(t_ax[::ds_step_t]).tolist(), y=freqs.tolist(),
            z=amp_db[:, ::ds_step_t].tolist(),
            colorscale='Viridis', colorbar={'title': 'dB'},
            hovertemplate='%{x:.2f}s / %{y:.2f}Hz<extra></extra>'))
        fig_cwt.add_trace(go.Scatter(
            x=t_ax[::step].tolist(), y=ridge['ifreq'][::step].tolist(),
            mode='lines', line={'color': 'red', 'width': 1.5}, name='Ridge'))
        fig_cwt.update_layout(title='CWT + Ridge',
                              xaxis_title='Time (s)', yaxis_title='Frequency (Hz)',
                              yaxis_type='log')

        # Instantaneous frequency / amplitude / phase / recon — 4-panel
        fig_ts = make_subplots(rows=4, cols=1, shared_xaxes=True,
            subplot_titles=('Instantaneous Frequency (Hz)',
                            'Instantaneous Amplitude',
                            'Instantaneous Phase (rad)',
                            'Reconstructed Signal'),
            vertical_spacing=0.05)
        for row, (key, label) in enumerate(
            (('ifreq','Hz'), ('iamp',''), ('iphi','rad'), ('recon','')), 1):
            fig_ts.add_trace(go.Scatter(
                x=t_ax[::step].tolist(), y=ridge[key][::step].tolist(),
                mode='lines', line={'width': 1}), row=row, col=1)
        fig_ts.update_layout(height=800, showlegend=False)

        processing_status[task_id].update({
            'status': 'complete', 'progress': 100, 'stage': 'Complete!',
            'results': {
                'cwt_plot':    json.dumps(fig_cwt, cls=plotly.utils.PlotlyJSONEncoder),
                'timeseries_plot': json.dumps(fig_ts, cls=plotly.utils.PlotlyJSONEncoder),
                'mean_ifreq':  round(float(np.nanmean(ridge['ifreq'])), 2),
                'std_ifreq':   round(float(np.nanstd(ridge['ifreq'])), 2),
                'mean_iamp':   round(float(np.nanmean(ridge['iamp'])), 4),
                'freq_min':    fmin, 'freq_max': fmax,
                'wavelet':     wavelet, 'cut_edges': cut_edges,
                'n_cycles':    n_cycles, 'gpu_used': USE_GPU,
            }
        })
    except Exception as e:
        processing_status[task_id].update({'status': 'error', 'error': str(e), 'stage': 'Error'})
        import traceback; traceback.print_exc()


@app.route('/filter_butter', methods=['POST'])
def filter_butter():
    """Butterworth bandpass filter + optional polynomial detrend."""
    if 'file' not in request.files or not request.files['file'].filename:
        return jsonify({'error': 'No file uploaded'}), 400
    f = request.files['file']
    fp = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
    f.save(fp)
    try:
        fs           = float(request.form.get('fs', 1.0))
        f_low        = float(request.form.get('f_low', 0.5))
        f_high       = float(request.form.get('f_high', 0))
        order        = int(request.form.get('order', 4))
        detrend_deg  = int(request.form.get('detrend_degree', 0))
        x, afs       = load_signal(fp)
        if afs and afs != 1.0: fs = afs
        if f_high <= 0: f_high = fs / 4.0
        task_id = str(uuid.uuid4())
        processing_status[task_id] = {'status': 'processing', 'progress': 0,
                                       'stage': 'Queued', 'fs': fs}
        t = Thread(target=_butter_worker,
                   args=(task_id, x, fs, f_low, f_high, order, detrend_deg))
        t.daemon = True; t.start()
        return jsonify({'task_id': task_id, 'signal_length': len(x), 'sampling_rate': fs})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _butter_worker(task_id, x, fs, f_low, f_high, order, detrend_degree):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    try:
        from fastmoda.filtering import butterworth_bandpass, detrend_polynomial

        processing_status[task_id].update({'progress': 20, 'stage': 'Detrending…'})
        x_in = detrend_polynomial(x, detrend_degree) if detrend_degree > 0 else x.astype(np.float32)

        processing_status[task_id].update({'progress': 40, 'stage': 'Filtering…'})
        x_filt = butterworth_bandpass(x_in, fs, f_low, f_high, order)

        processing_status[task_id].update({'progress': 70, 'stage': 'Building plots…'})
        t_ax  = np.arange(len(x)) / fs
        step  = max(1, len(t_ax) // 800)

        fig_t = make_subplots(rows=2, cols=1,
            subplot_titles=('Original Signal', f'Filtered ({f_low}–{f_high} Hz)'),
            shared_xaxes=True, vertical_spacing=0.1)
        fig_t.add_trace(go.Scatter(x=t_ax[::step].tolist(), y=x_in[::step].tolist(),
            mode='lines', line={'color': 'steelblue', 'width': 1}), row=1, col=1)
        fig_t.add_trace(go.Scatter(x=t_ax[::step].tolist(), y=x_filt[::step].tolist(),
            mode='lines', line={'color': 'tomato', 'width': 1}), row=2, col=1)
        fig_t.update_layout(height=500, showlegend=False)

        # Power spectra (vectorised rfft)
        N = len(x_in)
        freqs_f = np.fft.rfftfreq(N, 1.0/fs)
        psd_orig = np.abs(np.fft.rfft(x_in)) ** 2 / N
        psd_filt = np.abs(np.fft.rfft(x_filt)) ** 2 / N
        f_step = max(1, len(freqs_f) // 500)
        fig_f = go.Figure()
        fig_f.add_trace(go.Scatter(x=freqs_f[::f_step].tolist(), y=psd_orig[::f_step].tolist(),
            mode='lines', name='Original'))
        fig_f.add_trace(go.Scatter(x=freqs_f[::f_step].tolist(), y=psd_filt[::f_step].tolist(),
            mode='lines', name='Filtered'))
        fig_f.update_layout(title='Power Spectrum', xaxis_title='Frequency (Hz)',
                            yaxis_title='Power', yaxis_type='log')

        processing_status[task_id].update({
            'status': 'complete', 'progress': 100, 'stage': 'Complete!',
            'results': {
                'signal_plot':   json.dumps(fig_t, cls=plotly.utils.PlotlyJSONEncoder),
                'spectrum_plot': json.dumps(fig_f, cls=plotly.utils.PlotlyJSONEncoder),
                'rms_original':  round(float(np.sqrt(np.mean(x_in**2))), 4),
                'rms_filtered':  round(float(np.sqrt(np.mean(x_filt**2))), 4),
                'f_low':         f_low, 'f_high': f_high,
                'order':         order, 'detrend_degree': detrend_degree,
                'gpu_used':      False,
                # Downsampled filtered signal for app waveform display (≤512 pts)
                'filtered_signal': x_filt[::max(1, len(x_filt)//512)].tolist(),
            }
        })
    except Exception as e:
        processing_status[task_id].update({'status': 'error', 'error': str(e), 'stage': 'Error'})
        import traceback; traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════════════════
# LIVE MICROPHONE MONITOR
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/live')
def live():
    from fastmoda.audio_capture import is_available
    return render_template('live.html', gpu_enabled=USE_GPU,
                           audio_available=is_available())


@app.route('/stream/live')
def stream_live():
    from fastmoda.audio_capture import is_available, subscribe, unsubscribe
    if not is_available():
        return jsonify({'error': 'sounddevice not installed'}), 503

    def generate():
        q = subscribe()
        try:
            yield 'data: {"type":"connected"}\n\n'
            while True:
                try:
                    frame = q.get(timeout=2.0)
                    yield f'data: {json.dumps(frame)}\n\n'
                except Empty:
                    yield 'data: {"type":"keepalive"}\n\n'
        finally:
            unsubscribe(q)

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no',
                 'Connection': 'keep-alive'},
    )


@app.route('/api/live/status')
def live_status():
    from fastmoda.audio_capture import status
    return jsonify(status())


if __name__ == '__main__':
    # In Kubernetes/Docker, debug=True causes the reloader to fork and parent exits
    # Disable debug mode in production environments
    import sys
    debug_mode = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    if debug_mode:
        print("Running in DEBUG mode (dev environment)")
        app.run(host='0.0.0.0', port=5000, debug=True, threaded=True, use_reloader=False)
    else:
        print("Running in PRODUCTION mode (Kubernetes/Docker environment)")
        # Use app.run without debug for production, or preferably use gunicorn
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
