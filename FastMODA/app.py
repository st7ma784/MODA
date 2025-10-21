"""Enhanced Flask web UI for FastMODA with periodicity analysis"""
from flask import Flask, request, render_template, redirect, url_for
import os
from fastmoda.fastmoda import (
    load_signal, sliding_fft, compute_band_powers, detect_changepoints,
    extract_band_frequencies, detect_periodicity_changes
)
import numpy as np
import json
import plotly
import plotly.graph_objs as go
from plotly.subplots import make_subplots

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    error = None
    if request.method == 'POST':
        f = request.files.get('signal')
        fs_input = request.form.get('fs', '10.0')
        try:
            fs = float(fs_input)
        except:
            fs = 10.0
            
        if f:
            try:
                path = os.path.join(UPLOAD_FOLDER, f.filename)
                f.save(path)
                x, _ = load_signal(path)
                
                # Debug info
                print(f"Loaded signal: shape={x.shape}, ndim={x.ndim}, dtype={x.dtype}")
                
                # Compute spectrogram
                freqs, times, Sxx = sliding_fft(x, fs=fs, win_s=1.0, hop_s=0.25)
                
                # Define frequency bands
                bands = [(0, fs*0.25, 'low'), (fs*0.25, fs*0.5, 'mid'), (fs*0.5, fs*0.99, 'high')]
                feats, names = compute_band_powers(Sxx, freqs, bands=bands)
                cps = detect_changepoints(feats, pen=5)
                
                # Extract band frequencies over time
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
                    'spec_data': Sxx.tolist(),  # For interactive frequency slider
                    'freq_changes': periodicity['frequency_changes'],
                    'amp_changes': periodicity['amplitude_changes']
                }
            except Exception as e:
                error = f"Error processing signal: {str(e)}"
                print(f"ERROR: {error}")
                import traceback
                traceback.print_exc()
    
    return render_template('index.html', result=result, error=error)

if __name__ == '__main__':
    # Bind to 0.0.0.0 to make it accessible from outside the container
    # Get port from environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(host='0.0.0.0', port=port, debug=debug)
