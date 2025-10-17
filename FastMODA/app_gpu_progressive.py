"""GPU-enabled Flask application with progressive rendering and progress tracking"""
from flask import Flask, render_template, request, jsonify, session
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.utils
import json
import numpy as np
import os
import uuid
from threading import Thread
import time

# Try to import GPU utilities
try:
    from fastmoda.gpu_utils import (
        sliding_fft_gpu, 
        compute_band_powers_gpu, 
        is_gpu_available,
        get_gpu_info
    )
    GPU_ENABLED = True
except ImportError:
    GPU_ENABLED = False

from fastmoda import (
    load_signal, 
    sliding_fft, 
    compute_band_powers, 
    detect_changepoints,
    extract_band_frequencies,
    detect_periodicity_changes
)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max
app.secret_key = 'fastmoda-secret-key-change-in-production'

# Create uploads directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Store for processing status
processing_status = {}

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

print(f"FastMODA starting with {'GPU' if USE_GPU else 'CPU'} backend")
if GPU_ENABLED:
    gpu_info = get_gpu_info()
    print(f"GPU Info: {json.dumps(gpu_info, indent=2)}")

@app.route('/')
def index():
    return render_template('index_progressive.html', gpu_enabled=USE_GPU)

@app.route('/api/gpu-info')
def api_gpu_info():
    """API endpoint to get GPU information"""
    if GPU_ENABLED:
        return jsonify(get_gpu_info())
    return jsonify({'pytorch_available': False, 'cuda_available': False})

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
        pen = float(request.form.get('pen', 10))
        
        # Load signal
        x, _ = load_signal(filepath)
        print(f"Loaded signal: shape={x.shape}, fs={fs}")
        
        # Create task ID
        task_id = str(uuid.uuid4())
        
        # Initialize status
        processing_status[task_id] = {
            'status': 'processing',
            'progress': 10,
            'stage': 'Loading signal...',
            'signal_shape': x.shape,
            'fs': fs
        }
        
        # Generate initial signal plot immediately
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
            hovermode='x unified'
        )
        
        # Start background processing
        thread = Thread(target=background_analysis, args=(task_id, filepath, fs, win_s, pen, x))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'task_id': task_id,
            'signal_plot': json.dumps(signal_fig, cls=plotly.utils.PlotlyJSONEncoder),
            'signal_length': len(x),
            'sampling_rate': fs,
            'duration': len(x) / fs
        })
        
    except Exception as e:
        return jsonify({'error': f"Error: {str(e)}"}), 500

@app.route('/status/<task_id>')
def get_status(task_id):
    """Get processing status"""
    if task_id not in processing_status:
        return jsonify({'error': 'Task not found'}), 404
    return jsonify(processing_status[task_id])

def background_analysis(task_id, filepath, fs, win_s, pen, x):
    """Background analysis task"""
    try:
        # Update: FFT
        processing_status[task_id].update({
            'progress': 20,
            'stage': 'Computing FFT...'
        })
        
        if USE_GPU:
            print("Using GPU-accelerated FFT")
            freqs, times, Sxx = sliding_fft_gpu(x, fs, win_s)
        else:
            print("Using CPU FFT")
            freqs, times, Sxx = sliding_fft(x, fs, win_s)
        
        # Update: Band powers
        processing_status[task_id].update({
            'progress': 40,
            'stage': 'Computing band powers...'
        })
        
        bands = [
            (0.5, 4, 'delta'),
            (4, 8, 'theta'),
            (8, 13, 'alpha'),
            (13, 30, 'beta'),
            (30, 100, 'gamma')
        ]
        
        if USE_GPU:
            feats, names = compute_band_powers_gpu(Sxx, freqs, bands)
        else:
            feats, names = compute_band_powers(Sxx, freqs, bands)
        
        # Update: Changepoints
        processing_status[task_id].update({
            'progress': 60,
            'stage': 'Detecting changepoints...'
        })
        
        cps = detect_changepoints(feats, pen=pen)
        num_cps = len(cps)
        print(f"Detected {num_cps} changepoints")
        
        # OPTIMIZATION: Limit changepoints to prevent hanging
        MAX_CHANGEPOINTS = 100
        if num_cps > MAX_CHANGEPOINTS:
            print(f"Too many changepoints ({num_cps}), sampling {MAX_CHANGEPOINTS}")
            # Sample evenly across changepoints
            indices = np.linspace(0, num_cps - 1, MAX_CHANGEPOINTS, dtype=int)
            cps = cps[indices]
            processing_status[task_id]['warning'] = f'Reduced {num_cps} changepoints to {MAX_CHANGEPOINTS} for performance'
        
        # Update: Band frequencies
        processing_status[task_id].update({
            'progress': 70,
            'stage': 'Extracting band frequencies...'
        })
        
        band_freqs = extract_band_frequencies(Sxx, freqs, times, bands)
        
        # Update: Periodicity (this is the slow part with many changepoints)
        processing_status[task_id].update({
            'progress': 80,
            'stage': f'Analyzing periodicity ({len(cps)} segments)...'
        })
        
        periodicity = detect_periodicity_changes(x, fs, times, cps, tolerance=0.1)
        
        # Generate plots
        processing_status[task_id].update({
            'progress': 90,
            'stage': 'Generating visualizations...'
        })
        
        plots = generate_plots(x, fs, times, freqs, Sxx, feats, names, cps, band_freqs, periodicity)
        
        # Complete
        processing_status[task_id].update({
            'status': 'complete',
            'progress': 100,
            'stage': 'Complete!',
            'results': plots
        })
        
        print(f"Task {task_id} completed successfully")
        
    except Exception as e:
        processing_status[task_id].update({
            'status': 'error',
            'error': str(e),
            'stage': 'Error occurred'
        })
        print(f"Error in background task {task_id}: {e}")
        import traceback
        traceback.print_exc()

def generate_plots(x, fs, times, freqs, Sxx, feats, names, cps, band_freqs, periodicity):
    """Generate all plots"""
    t_signal = np.arange(len(x)) / fs
    
    # Signal with changepoints
    signal_fig = go.Figure()
    signal_fig.add_trace(go.Scatter(
        x=t_signal, y=x, 
        mode='lines', 
        name='Signal', 
        line={'color': 'blue'}
    ))
    for cp in cps:
        t_cp = times[cp] if cp < len(times) else times[-1]
        signal_fig.add_vline(x=t_cp, line_dash="dash", line_color="red", opacity=0.5)
    signal_fig.update_layout(
        title=f'Signal with {len(cps)} Changepoints',
        xaxis_title='Time (s)',
        yaxis_title='Amplitude',
        hovermode='x unified'
    )
    
    # Spectrogram
    spec_fig = go.Figure()
    spec_fig.add_trace(go.Heatmap(
        z=20*np.log10(Sxx+1e-12),
        x=times, y=freqs,
        colorscale='Viridis',
        name='Spectrogram'
    ))
    for cp in cps:
        t_cp = times[cp] if cp < len(times) else times[-1]
        spec_fig.add_vline(x=t_cp, line_dash="dash", line_color="red", opacity=0.7)
    spec_fig.update_layout(
        title='Spectrogram (dB)',
        xaxis_title='Time (s)',
        yaxis_title='Frequency (Hz)',
        hovermode='closest'
    )
    
    # Band features
    feat_fig = go.Figure()
    for i, name in enumerate(names):
        feat_fig.add_trace(go.Scatter(x=times, y=feats[:,i], mode='lines', name=name))
    for cp in cps:
        t_cp = times[cp] if cp < len(times) else times[-1]
        feat_fig.add_vline(x=t_cp, line_dash="dash", line_color="red", opacity=0.5)
    feat_fig.update_layout(
        title='Band Power Features',
        xaxis_title='Time (s)',
        yaxis_title='Log Power',
        hovermode='x unified'
    )
    
    # Dominant frequencies
    freq_fig = go.Figure()
    for band_name, (t, f, a) in band_freqs.items():
        freq_fig.add_trace(go.Scatter(x=t, y=f, mode='lines', name=f'{band_name} freq', line={'width': 2}))
    for cp in cps:
        t_cp = times[cp] if cp < len(times) else times[-1]
        freq_fig.add_vline(x=t_cp, line_dash="dash", line_color="red", opacity=0.5)
    freq_fig.update_layout(
        title='Dominant Frequency per Band',
        xaxis_title='Time (s)',
        yaxis_title='Frequency (Hz)',
        hovermode='x unified'
    )
    
    # Periodicity
    period_fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Fitted Frequencies', 'Fitted Amplitudes'),
        vertical_spacing=0.15
    )
    
    sine_fits = periodicity['sine_fits']
    if sine_fits:
        seg_times = [sf['time_range'][0] + (sf['time_range'][1] - sf['time_range'][0])/2 for sf in sine_fits]
        seg_freqs = [sf['frequency'] for sf in sine_fits]
        seg_amps = [sf['amplitude'] for sf in sine_fits]
        
        period_fig.add_trace(
            go.Scatter(x=seg_times, y=seg_freqs, mode='lines+markers', name='Frequency',
                      line={'color': 'purple', 'width': 3}, marker={'size': 8}),
            row=1, col=1
        )
        period_fig.add_trace(
            go.Scatter(x=seg_times, y=seg_amps, mode='lines+markers', name='Amplitude',
                      line={'color': 'orange', 'width': 3}, marker={'size': 8}),
            row=2, col=1
        )
    
    period_fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    period_fig.update_yaxes(title_text="Frequency (Hz)", row=1, col=1)
    period_fig.update_yaxes(title_text="Amplitude", row=2, col=1)
    period_fig.update_layout(height=600, showlegend=False, title_text="Periodicity Analysis")
    
    return {
        'signal': json.dumps(signal_fig, cls=plotly.utils.PlotlyJSONEncoder),
        'spec': json.dumps(spec_fig, cls=plotly.utils.PlotlyJSONEncoder),
        'feats': json.dumps(feat_fig, cls=plotly.utils.PlotlyJSONEncoder),
        'freq': json.dumps(freq_fig, cls=plotly.utils.PlotlyJSONEncoder),
        'period': json.dumps(period_fig, cls=plotly.utils.PlotlyJSONEncoder),
        'cps': cps.tolist(),
        'times': times.tolist(),
        'freqs': freqs.tolist(),
        'spec_data': Sxx.tolist()
    }

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
