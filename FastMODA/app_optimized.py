"""Fully optimized GPU-enabled Flask application

Key improvements over previous version:
1. Batched GPU FFT (10-50x faster)
2. Changepoint detection on FREQUENCY not power (fewer, better changepoints)
3. Adaptive penalty tuning (auto-adjusts to signal characteristics)
4. Efficient sine fitting with smart segment merging
5. Real-time progress tracking
"""
from flask import Flask, render_template, request, jsonify
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.utils
import json
import numpy as np
import os
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
    USE_GPU = GPU_ENABLED and TORCH_AVAILABLE and torch.cuda.is_available()
elif USE_GPU == 'true':
    USE_GPU = GPU_ENABLED and TORCH_AVAILABLE and torch.cuda.is_available()
    if not USE_GPU:
        print("Warning: GPU requested but not available. Falling back to CPU.")
else:
    USE_GPU = False

DEVICE = torch.device('cuda' if USE_GPU else 'cpu') if TORCH_AVAILABLE else None

print(f"\n{'='*60}")
print(f"FastMODA OPTIMIZED - Starting")
print(f"Backend: {'GPU (OPTIMIZED)' if USE_GPU else 'CPU'}")
if USE_GPU:
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"{'='*60}\n")

@app.route('/')
def index():
    return render_template('index_optimized.html', gpu_enabled=USE_GPU)

@app.route('/coherence')
def coherence():
    """Coherence analysis page"""
    if not USE_GPU:
        return render_template('index_optimized.html', 
                             gpu_enabled=False,
                             warning='Coherence analysis requires GPU acceleration')
    return render_template('coherence.html', gpu_enabled=USE_GPU)

@app.route('/bispectrum')
def bispectrum():
    """Bispectrum analysis page"""
    if not USE_GPU:
        return render_template('index_optimized.html', 
                             gpu_enabled=False,
                             warning='Bispectrum analysis requires GPU acceleration')
    return render_template('bispectrum.html', gpu_enabled=USE_GPU)

@app.route('/bayesian')
def bayesian():
    """Bayesian inference page"""
    if not USE_GPU:
        return render_template('index_optimized.html', 
                             gpu_enabled=False,
                             warning='Bayesian inference requires GPU acceleration')
    return render_template('bayesian.html', gpu_enabled=USE_GPU)

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
        
        # Load signal
        x, _ = load_signal(filepath)
        print(f"\n{'='*60}")
        print(f"NEW ANALYSIS REQUEST")
        print(f"Signal: {len(x)} samples, {len(x)/fs:.2f} seconds")
        print(f"Window: {win_s}s, Penalty: {pen}")
        print(f"{'='*60}")
        
        # Create task
        task_id = str(uuid.uuid4())
        
        processing_status[task_id] = {
            'status': 'processing',
            'progress': 10,
            'stage': 'Loading signal...',
            'signal_shape': x.shape,
            'fs': fs
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
                       args=(task_id, filepath, fs, win_s, pen, x))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'task_id': task_id,
            'signal_plot': json.dumps(signal_fig, cls=plotly.utils.PlotlyJSONEncoder),
            'signal_length': len(x),
            'sampling_rate': fs,
            'duration': len(x) / fs,
            'optimized': USE_GPU
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

def optimized_background_analysis(task_id, filepath, fs, win_s, pen, x):
    """Optimized background analysis using new pipeline"""
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
        
        # Periodicity analysis with smart segment limiting
        processing_status[task_id].update({
            'progress': 80,
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
        processing_status[task_id].update({
            'progress': 90,
            'stage': 'Generating visualizations...'
        })
        
        plots = generate_optimized_plots(x, fs, times, freqs, Sxx, feats, names, 
                                        cps, band_freqs, periodicity, inst_freq)
        
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

def generate_optimized_plots(x, fs, times, freqs, Sxx, feats, names, cps, band_freqs, periodicity, inst_freq):
    """Generate all plots with optimization info"""
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
    
    # Add changepoints
    for cp_t in cp_times:
        spec_fig.add_vline(x=cp_t, line_dash="dash", line_color="red", opacity=0.7)
    
    spec_fig.update_layout(
        title='Time-Frequency Spectrogram',
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
    
    return {
        'signal': json.dumps(signal_fig, cls=plotly.utils.PlotlyJSONEncoder),
        'spectrogram': json.dumps(spec_fig, cls=plotly.utils.PlotlyJSONEncoder),
        'timeline': json.dumps(timeline_fig, cls=plotly.utils.PlotlyJSONEncoder),
        'instantaneous_freq': json.dumps(inst_fig, cls=plotly.utils.PlotlyJSONEncoder),
        'band_powers': json.dumps(band_fig, cls=plotly.utils.PlotlyJSONEncoder),
        'periodicity': json.dumps(period_fig, cls=plotly.utils.PlotlyJSONEncoder),
        'component_plots': component_plots,
        'frequency_summary': frequency_summary
    }

@app.route('/analyze_coherence', methods=['POST'])
def analyze_coherence():
    """Multi-signal coherence analysis endpoint"""
    if not USE_GPU:
        return jsonify({'error': 'Coherence analysis requires GPU acceleration'}), 400
    
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
        
        # Load all signals
        signals = []
        signal_names = []
        for file in files:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            signal, actual_fs = load_signal(filepath)
            if actual_fs is not None:
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
            args=(task_id, signals, signal_names, fs, win_s, overlap, numcycles)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({'task_id': task_id})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def process_coherence_background(task_id, signals, signal_names, fs, win_s, overlap, numcycles):
    """Background processing for coherence analysis"""
    from fastmoda.coherence_gpu import compute_multi_pair_coherence_gpu
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    try:
        # Update progress
        processing_status[task_id]['stage'] = 'Computing wavelet transforms'
        processing_status[task_id]['progress'] = 20
        
        # Compute coherence for all pairs
        results = compute_multi_pair_coherence_gpu(
            signals, signal_names, fs,
            win_s=win_s, overlap=overlap, numcycles=numcycles,
            device=DEVICE
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
            fig.add_trace(
                go.Scatter(
                    x=freqs, y=phcoh,
                    mode='lines',
                    name='Coherence',
                    line=dict(color='blue', width=2),
                    hovertemplate='Freq: %{x:.2f} Hz<br>Coherence: %{y:.3f}<extra></extra>'
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
                showlegend=False,
                title_text=f'Wavelet Phase Coherence Analysis: {name1} ↔ {name2}',
                title_font_size=16
            )
            
            pair_plots[f'{name1}_vs_{name2}'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        processing_status[task_id]['stage'] = 'Complete'
        processing_status[task_id]['progress'] = 100
        processing_status[task_id]['result'] = {
            'pair_plots': pair_plots,
            'n_pairs': len(results),
            'signal_names': signal_names
        }
    
    except Exception as e:
        processing_status[task_id]['error'] = str(e)
        processing_status[task_id]['stage'] = 'Error'


@app.route('/analyze_bispectrum', methods=['POST'])
def analyze_bispectrum():
    """Bispectrum analysis endpoint for detecting frequency coupling"""
    if not USE_GPU:
        return jsonify({'error': 'Bispectrum analysis requires GPU acceleration'}), 400
    
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
            if actual_fs is not None:
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
    from fastmoda.bispectrum_gpu import wavelet_bispectrum_gpu, find_significant_couplings
    import plotly.graph_objects as go
    
    try:
        processing_status[task_id]['stage'] = 'Computing bispectrum'
        processing_status[task_id]['progress'] = 20
        
        # Compute bispectrum
        result = wavelet_bispectrum_gpu(
            torch.from_numpy(signals[0]).to(DEVICE),
            torch.from_numpy(signals[1]).to(DEVICE),
            fs,
            freq_range=(freq_min, freq_max),
            n_freqs=n_freqs,
            bispectrum_type=bispec_type,
            device=DEVICE
        )
        
        processing_status[task_id]['stage'] = 'Finding significant couplings'
        processing_status[task_id]['progress'] = 60
        
        # Find significant couplings
        couplings = find_significant_couplings(result, threshold_percentile=95)
        
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
        
        processing_status[task_id]['stage'] = 'Complete'
        processing_status[task_id]['progress'] = 100
        processing_status[task_id]['result'] = {
            'bispectrum_plot': json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder),
            'coupling_strength': result['coupling_strength'],
            'top_couplings': [
                {'f1': f1, 'f2': f2, 'f3': f1+f2, 'strength': float(s)}
                for f1, f2, s in top_couplings
            ],
            'bispec_type': bispec_type,
            'freq_range': result['freq_range']
        }
    
    except Exception as e:
        processing_status[task_id]['error'] = str(e)
        processing_status[task_id]['stage'] = 'Error'


@app.route('/analyze_bayesian', methods=['POST'])
def analyze_bayesian():
    """Bayesian inference endpoint for phase coupling"""
    if not USE_GPU:
        return jsonify({'error': 'Bayesian analysis requires GPU acceleration'}), 400
    
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
        
        # Load signals
        signals = []
        signal_names = []
        for file in files:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            signal, actual_fs = load_signal(filepath)
            if actual_fs is not None:
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
                  window_s, n_surrogates)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({'task_id': task_id})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def process_bayesian_background(task_id, signals, signal_names, fs, band1, band2, window_s, n_surrogates):
    """Background processing for Bayesian inference"""
    from fastmoda.bayesian_gpu import bayesian_inference_full
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    try:
        processing_status[task_id]['stage'] = 'Running Bayesian inference'
        processing_status[task_id]['progress'] = 20
        
        # Bayesian inference
        result = bayesian_inference_full(
            torch.from_numpy(signals[0]).to(DEVICE),
            torch.from_numpy(signals[1]).to(DEVICE),
            fs,
            band1=band1,
            band2=band2,
            window_s=window_s,
            n_surrogates=n_surrogates,
            device=DEVICE
        )
        
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
        
        processing_status[task_id]['stage'] = 'Complete'
        processing_status[task_id]['progress'] = 100
        processing_status[task_id]['result'] = {
            'coupling_plot': json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder),
            'mean_cpl1': float(np.mean(cpl1)),
            'mean_cpl2': float(np.mean(cpl2)),
            'mean_direction': float(np.mean(result['direction'])),
            'band1': band1,
            'band2': band2,
            'window_s': window_s,
            'n_surrogates': n_surrogates if 'surr_cpl1' in result else 0
        }
    
    except Exception as e:
        processing_status[task_id]['error'] = str(e)
        processing_status[task_id]['stage'] = 'Error'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
