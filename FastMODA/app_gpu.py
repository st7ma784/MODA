"""GPU-enabled Flask application with automatic CPU fallback

Automatically detects GPU availability and uses optimal backend.
"""
from flask import Flask, render_template, request, jsonify
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.utils
import json
import numpy as np
import os

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

print(f"FastMODA starting with {'GPU' if USE_GPU else 'CPU'} backend")
if GPU_ENABLED:
    gpu_info = get_gpu_info()
    print(f"GPU Info: {json.dumps(gpu_info, indent=2)}")

@app.route('/')
def index():
    return render_template('index.html', result=None, error=None, gpu_enabled=USE_GPU)

@app.route('/api/gpu-info')
def api_gpu_info():
    """API endpoint to get GPU information"""
    if GPU_ENABLED:
        return jsonify(get_gpu_info())
    return jsonify({'pytorch_available': False, 'cuda_available': False})

@app.route('/analyze', methods=['POST'])
def analyze():
    result = None
    error = None
    
    if 'file' in request.files and request.files['file'].filename:
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
            
            # Choose FFT function based on GPU availability
            if USE_GPU:
                print("Using GPU-accelerated FFT")
                freqs, times, Sxx = sliding_fft_gpu(x, fs, win_s)
            else:
                print("Using CPU FFT")
                freqs, times, Sxx = sliding_fft(x, fs, win_s)
            
            # Define frequency bands
            bands = [
                (0.5, 4, 'delta'),
                (4, 8, 'theta'),
                (8, 13, 'alpha'),
                (13, 30, 'beta'),
                (30, 100, 'gamma')
            ]
            
            # Compute band powers (use GPU if available)
            if USE_GPU:
                feats, names = compute_band_powers_gpu(Sxx, freqs, bands)
            else:
                feats, names = compute_band_powers(Sxx, freqs, bands)
            
            # Detect changepoints (CPU - ruptures doesn't support GPU)
            cps = detect_changepoints(feats, pen=pen)
            print(f"Detected {len(cps)} changepoints")
            
            # Extract band frequencies
            band_freqs = extract_band_frequencies(Sxx, freqs, times, bands)
            
            # Periodicity analysis
            periodicity = detect_periodicity_changes(x, fs, times, cps, tolerance=0.1)

            # === Plot 1: Original signal with changepoints ===
            t_signal = np.arange(len(x)) / fs
            signal_fig = go.Figure()
            signal_fig.add_trace(go.Scatter(x=t_signal, y=x, mode='lines', name='Signal', line={'color': 'blue'}))
            
            # Mark changepoints
            for cp in cps:
                t_cp = times[cp] if cp < len(times) else times[-1]
                signal_fig.add_vline(x=t_cp, line_dash="dash", line_color="red", opacity=0.5)
            
            signal_fig.update_layout(
                title='Original Signal with Changepoints',
                xaxis_title='Time (s)',
                yaxis_title='Amplitude',
                hovermode='x unified'
            )

            # === Plot 2: Interactive Spectrogram with slider ===
            spec_fig = go.Figure()
            spec_fig.add_trace(go.Heatmap(
                z=20*np.log10(Sxx+1e-12),
                x=times,
                y=freqs,
                colorscale='Viridis',
                name='Spectrogram'
            ))
            
            # Add changepoint markers
            for cp in cps:
                t_cp = times[cp] if cp < len(times) else times[-1]
                spec_fig.add_vline(x=t_cp, line_dash="dash", line_color="red", opacity=0.7)
            
            spec_fig.update_layout(
                title='Spectrogram (dB) with Changepoints',
                xaxis_title='Time (s)',
                yaxis_title='Frequency (Hz)',
                hovermode='closest'
            )

            # === Plot 3: Band features with changepoints ===
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

            # === Plot 4: Dominant frequencies per band ===
            freq_fig = go.Figure()
            for band_name, (t, f, a) in band_freqs.items():
                freq_fig.add_trace(go.Scatter(
                    x=t, y=f,
                    mode='lines',
                    name=f'{band_name} freq',
                    line={'width': 2}
                ))
            
            for cp in cps:
                t_cp = times[cp] if cp < len(times) else times[-1]
                freq_fig.add_vline(x=t_cp, line_dash="dash", line_color="red", opacity=0.5)
            
            freq_fig.update_layout(
                title='Dominant Frequency per Band Over Time',
                xaxis_title='Time (s)',
                yaxis_title='Frequency (Hz)',
                hovermode='x unified'
            )

            # === Plot 5: Periodicity analysis (sine fits) ===
            period_fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Fitted Frequencies per Segment', 'Fitted Amplitudes per Segment'),
                vertical_spacing=0.15
            )
            
            sine_fits = periodicity['sine_fits']
            if sine_fits:
                seg_times = [sf['time_range'][0] + (sf['time_range'][1] - sf['time_range'][0])/2 
                            for sf in sine_fits]
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
                
                # Mark frequency changes
                for fc in periodicity['frequency_changes']:
                    period_fig.add_vline(x=fc['time'], line_dash="dot", line_color="red", 
                                        opacity=0.7, row=1, col=1)
                
                # Mark amplitude changes
                for ac in periodicity['amplitude_changes']:
                    period_fig.add_vline(x=ac['time'], line_dash="dot", line_color="red",
                                        opacity=0.7, row=2, col=1)
            
            period_fig.update_xaxes(title_text="Time (s)", row=2, col=1)
            period_fig.update_yaxes(title_text="Frequency (Hz)", row=1, col=1)
            period_fig.update_yaxes(title_text="Amplitude", row=2, col=1)
            period_fig.update_layout(height=600, showlegend=False, 
                                    title_text="Periodicity Analysis: Sine Wave Fits")

            result = {
                'signal': json.dumps(signal_fig, cls=plotly.utils.PlotlyJSONEncoder),
                'spec': json.dumps(spec_fig, cls=plotly.utils.PlotlyJSONEncoder),
                'feats': json.dumps(feat_fig, cls=plotly.utils.PlotlyJSONEncoder),
                'freq': json.dumps(freq_fig, cls=plotly.utils.PlotlyJSONEncoder),
                'period': json.dumps(period_fig, cls=plotly.utils.PlotlyJSONEncoder),
                'cps': cps.tolist(),
                'times': times.tolist(),
                'freqs': freqs.tolist(),
                'spec_data': Sxx.tolist(),
                'freq_changes': periodicity['frequency_changes'],
                'amp_changes': periodicity['amplitude_changes'],
                'gpu_used': USE_GPU
            }
        except Exception as e:
            error = f"Error processing signal: {str(e)}"
            print(f"ERROR: {error}")
            import traceback
            traceback.print_exc()
    
    return render_template('index.html', result=result, error=error, gpu_enabled=USE_GPU)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
