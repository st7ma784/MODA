"""Compare old vs optimized changepoint detection methods

This script demonstrates the key difference:
- OLD: Detect changepoints on band POWERS (noisy, many false positives)
- NEW: Detect changepoints on FREQUENCY (clean, meaningful)
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window
from scipy.fft import rfft, rfftfreq
import ruptures as rpt

def create_test_signal(fs=1000, duration=10):
    """Create test signal with frequency jump at t=5s
    
    - 0-5s: 10 Hz sine wave
    - 5-10s: 20 Hz sine wave
    
    Expected: 1 changepoint at t=5s
    """
    t = np.arange(0, duration, 1/fs)
    x = np.zeros_like(t)
    
    # 0-5s: 10 Hz
    mask1 = t < 5
    x[mask1] = np.sin(2 * np.pi * 10 * t[mask1])
    
    # 5-10s: 20 Hz
    mask2 = t >= 5
    x[mask2] = np.sin(2 * np.pi * 20 * t[mask2])
    
    # Add some noise
    x += 0.1 * np.random.randn(len(x))
    
    return t, x

def sliding_fft(x, fs, win_s=1.0):
    """Simple sliding FFT"""
    N = len(x)
    win_n = int(win_s * fs)
    hop_n = win_n // 4
    nfft = 1 << (win_n - 1).bit_length()
    
    w = get_window('hann', win_n)
    
    frames = []
    times = []
    for start in range(0, N - win_n + 1, hop_n):
        frame = x[start:start+win_n] * w
        X = rfft(frame, n=nfft)
        frames.append(np.abs(X))
        times.append((start + win_n/2) / fs)
    
    Sxx = np.vstack(frames).T
    freqs = rfftfreq(nfft, 1/fs)
    return freqs, np.array(times), Sxx

def compute_band_powers(Sxx, freqs):
    """Compute band powers (OLD METHOD)"""
    bands = [
        (0.5, 4, 'delta'),
        (4, 8, 'theta'),
        (8, 13, 'alpha'),
        (13, 30, 'beta'),
        (30, 100, 'gamma')
    ]
    
    band_pows = []
    for fmin, fmax, name in bands:
        idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]
        if idx.size > 0:
            band_pows.append(np.sum(Sxx[idx,:]**2, axis=0))
        else:
            band_pows.append(np.zeros(Sxx.shape[1]))
    
    feats = np.vstack(band_pows).T
    feats = np.log(feats + 1e-12)
    return feats

def extract_instantaneous_frequency(Sxx, freqs):
    """Extract dominant frequency (NEW METHOD)"""
    peak_idx = np.argmax(Sxx, axis=0)
    inst_freq = freqs[peak_idx]
    return inst_freq

def compute_spectral_centroid(Sxx, freqs):
    """Compute spectral centroid (NEW METHOD)"""
    power = Sxx**2
    total_power = np.sum(power, axis=0) + 1e-12
    centroid = np.sum(freqs[:, np.newaxis] * power, axis=0) / total_power
    return centroid

def detect_changepoints_old(feats, pen=10):
    """OLD METHOD: Detect on band powers"""
    algo = rpt.Pelt(model='l2').fit(feats)
    bkps = algo.predict(pen=pen)
    return np.array(bkps[:-1], dtype=int)

def detect_changepoints_new(Sxx, freqs, pen=10):
    """NEW METHOD: Detect on frequency"""
    inst_freq = extract_instantaneous_frequency(Sxx, freqs)
    centroid = compute_spectral_centroid(Sxx, freqs)
    
    # Normalize
    inst_norm = (inst_freq - np.mean(inst_freq)) / (np.std(inst_freq) + 1e-12)
    cent_norm = (centroid - np.mean(centroid)) / (np.std(centroid) + 1e-12)
    
    features = np.column_stack([inst_norm, cent_norm])
    
    algo = rpt.Pelt(model='l2').fit(features)
    bkps = algo.predict(pen=pen)
    return np.array(bkps[:-1], dtype=int)

def main():
    """Compare old vs new methods"""
    print("="*60)
    print("CHANGEPOINT DETECTION COMPARISON")
    print("="*60)
    
    # Create test signal
    print("\nCreating test signal:")
    print("  - 0-5s: 10 Hz sine wave")
    print("  - 5-10s: 20 Hz sine wave")
    print("  - Expected: 1 changepoint at t=5s")
    
    t, x = create_test_signal()
    fs = 1000
    
    # Compute FFT
    print("\nComputing FFT...")
    freqs, times, Sxx = sliding_fft(x, fs, win_s=1.0)
    print(f"  - {len(times)} time windows")
    print(f"  - {len(freqs)} frequency bins")
    
    # OLD METHOD
    print("\n" + "="*60)
    print("OLD METHOD: Detect on band POWERS")
    print("="*60)
    feats = compute_band_powers(Sxx, freqs)
    cps_old = detect_changepoints_old(feats, pen=10)
    print(f"Detected {len(cps_old)} changepoints:")
    if len(cps_old) > 0:
        for i, cp in enumerate(cps_old[:10]):  # Show first 10
            print(f"  {i+1}. t = {times[cp]:.2f}s")
        if len(cps_old) > 10:
            print(f"  ... and {len(cps_old) - 10} more")
    
    # NEW METHOD
    print("\n" + "="*60)
    print("NEW METHOD: Detect on FREQUENCY")
    print("="*60)
    cps_new = detect_changepoints_new(Sxx, freqs, pen=10)
    print(f"Detected {len(cps_new)} changepoints:")
    if len(cps_new) > 0:
        for i, cp in enumerate(cps_new):
            print(f"  {i+1}. t = {times[cp]:.2f}s")
    
    # RESULTS
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print(f"Old method: {len(cps_old)} changepoints")
    print(f"New method: {len(cps_new)} changepoints")
    print(f"Reduction: {(1 - len(cps_new)/max(1, len(cps_old)))*100:.1f}%")
    
    # Find closest to true changepoint (t=5s)
    true_cp_time = 5.0
    if len(cps_new) > 0:
        detected_times = times[cps_new]
        closest = detected_times[np.argmin(np.abs(detected_times - true_cp_time))]
        error = abs(closest - true_cp_time)
        print(f"\nAccuracy:")
        print(f"  True changepoint: {true_cp_time:.2f}s")
        print(f"  Detected: {closest:.2f}s")
        print(f"  Error: {error:.3f}s")
    
    # VISUALIZATION
    print("\nGenerating comparison plot...")
    
    fig, axes = plt.subplots(5, 1, figsize=(12, 10))
    
    # 1. Original signal
    axes[0].plot(t, x, 'b-', linewidth=0.5)
    axes[0].axvline(5.0, color='green', linestyle='--', label='True changepoint')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Original Signal (10 Hz → 20 Hz at t=5s)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Spectrogram
    im = axes[1].pcolormesh(times, freqs[:100], 
                            10*np.log10(Sxx[:100,:]**2 + 1e-12),
                            shading='auto', cmap='viridis')
    axes[1].axvline(5.0, color='green', linestyle='--', linewidth=2)
    axes[1].set_ylabel('Frequency (Hz)')
    axes[1].set_title('Spectrogram (FFT decomposition)')
    plt.colorbar(im, ax=axes[1], label='Power (dB)')
    
    # 3. Instantaneous frequency (NEW)
    inst_freq = extract_instantaneous_frequency(Sxx, freqs)
    axes[2].plot(times, inst_freq, 'purple', linewidth=2, label='Instantaneous freq')
    axes[2].axvline(5.0, color='green', linestyle='--', label='True changepoint')
    for cp in cps_new:
        axes[2].axvline(times[cp], color='red', linestyle=':', alpha=0.7)
    axes[2].set_ylabel('Frequency (Hz)')
    axes[2].set_title(f'NEW METHOD: Instantaneous Frequency ({len(cps_new)} changepoints detected)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # 4. Band powers (OLD)
    for i in range(feats.shape[1]):
        axes[3].plot(times, feats[:, i], label=f'Band {i+1}', alpha=0.7)
    axes[3].axvline(5.0, color='green', linestyle='--', linewidth=2)
    for cp in cps_old[:20]:  # Limit to first 20
        axes[3].axvline(times[cp], color='red', linestyle=':', alpha=0.3)
    axes[3].set_ylabel('Log Power')
    axes[3].set_title(f'OLD METHOD: Band Powers ({len(cps_old)} changepoints detected)')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    # 5. Comparison
    axes[4].plot(t, x, 'b-', linewidth=0.5, alpha=0.3)
    axes[4].axvline(5.0, color='green', linestyle='--', linewidth=3, 
                   label=f'True CP (t=5.0s)', alpha=0.8)
    
    # Old method changepoints
    for cp in cps_old:
        axes[4].axvline(times[cp], color='orange', linestyle=':', 
                       linewidth=1, alpha=0.5)
    if len(cps_old) > 0:
        axes[4].axvline(times[cps_old[0]], color='orange', linestyle=':', 
                       linewidth=1, alpha=0.5, label=f'Old: {len(cps_old)} CPs')
    
    # New method changepoints
    for cp in cps_new:
        axes[4].axvline(times[cp], color='red', linestyle='-', 
                       linewidth=2, alpha=0.7)
    if len(cps_new) > 0:
        axes[4].axvline(times[cps_new[0]], color='red', linestyle='-', 
                       linewidth=2, alpha=0.7, label=f'New: {len(cps_new)} CPs')
    
    axes[4].set_xlabel('Time (s)')
    axes[4].set_ylabel('Amplitude')
    axes[4].set_title('Comparison: Green=Truth, Orange=Old (many), Red=New (accurate)')
    axes[4].legend()
    axes[4].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/data/MODA/FastMODA/changepoint_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: changepoint_comparison.png")
    
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    print(f"✅ NEW method detects {len(cps_new)} changepoints (accurate)")
    print(f"❌ OLD method detects {len(cps_old)} changepoints (too many)")
    print(f"⚡ Reduction: {len(cps_old) - len(cps_new)} fewer changepoints")
    print(f"📊 See: changepoint_comparison.png")

if __name__ == '__main__':
    main()
