"""Test script to validate all FastMODA features"""
import sys
sys.path.insert(0, '.')
from fastmoda.fastmoda import (
    load_signal, sliding_fft, compute_band_powers, detect_changepoints,
    extract_band_frequencies, detect_periodicity_changes
)
import numpy as np

def test_basic_processing():
    """Test basic signal processing pipeline"""
    print("=" * 60)
    print("TEST 1: Basic Signal Processing")
    print("=" * 60)
    
    x, _ = load_signal('../example_sigs/1signal_10Hz.mat')
    fs = 10.0
    print(f"✅ Loaded signal: shape={x.shape}, fs={fs} Hz")
    
    freqs, times, Sxx = sliding_fft(x, fs=fs, win_s=1.0, hop_s=0.25)
    print(f"✅ Computed spectrogram: {Sxx.shape[0]} frequencies × {Sxx.shape[1]} time points")
    print(f"   Frequency range: {freqs[0]:.2f} - {freqs[-1]:.2f} Hz")
    print(f"   Time range: {times[0]:.2f} - {times[-1]:.2f} s")
    
    return x, fs, freqs, times, Sxx

def test_changepoint_detection(freqs, times, Sxx):
    """Test changepoint detection on features"""
    print("\n" + "=" * 60)
    print("TEST 2: Changepoint Detection")
    print("=" * 60)
    
    bands = [(0, 2.5, 'low'), (2.5, 5, 'mid'), (5, 10, 'high')]
    feats, names = compute_band_powers(Sxx, freqs, bands=bands)
    print(f"✅ Extracted band features: {feats.shape}")
    print(f"   Bands: {names}")
    
    cps = detect_changepoints(feats, pen=5)
    print(f"✅ Detected {len(cps)} changepoints")
    if len(cps) > 0:
        cp_times = [times[cp] for cp in cps[:5]]
        print(f"   First 5 changepoint times: {[f'{t:.1f}s' for t in cp_times]}")
    
    return feats, names, cps, bands

def test_band_frequencies(Sxx, freqs, times, bands):
    """Test band frequency extraction"""
    print("\n" + "=" * 60)
    print("TEST 3: Band Frequency Extraction")
    print("=" * 60)
    
    band_freqs = extract_band_frequencies(Sxx, freqs, times, bands)
    for name, (t, f, a) in band_freqs.items():
        print(f"✅ {name} band:")
        print(f"   Frequency range: {f.min():.2f} - {f.max():.2f} Hz")
        print(f"   Amplitude range: {a.min():.3f} - {a.max():.3f}")
    
    return band_freqs

def test_periodicity_analysis(x, fs, times, cps):
    """Test periodicity analysis with sine fitting"""
    print("\n" + "=" * 60)
    print("TEST 4: Periodicity Analysis")
    print("=" * 60)
    
    # Use subset of changepoints for faster testing
    cps_subset = cps[:min(20, len(cps))]
    periodicity = detect_periodicity_changes(x, fs, times, cps_subset, tolerance=0.1)
    
    print(f"✅ Fitted sine waves to {len(periodicity['sine_fits'])} segments")
    
    if periodicity['sine_fits']:
        first_fit = periodicity['sine_fits'][0]
        print(f"   First segment:")
        print(f"     Time range: {first_fit['time_range'][0]:.1f} - {first_fit['time_range'][1]:.1f} s")
        print(f"     Frequency: {first_fit['frequency']:.2f} Hz")
        print(f"     Amplitude: {first_fit['amplitude']:.3f}")
        print(f"     Fit error: {first_fit['fit_error']:.4f}")
    
    print(f"✅ Detected {len(periodicity['frequency_changes'])} frequency changes")
    if periodicity['frequency_changes']:
        fc = periodicity['frequency_changes'][0]
        print(f"   First frequency change at t={fc['time']:.1f}s:")
        print(f"     {fc['from_freq']:.2f} Hz → {fc['to_freq']:.2f} Hz ({fc['rel_change']*100:.1f}% change)")
    
    print(f"✅ Detected {len(periodicity['amplitude_changes'])} amplitude changes")
    if periodicity['amplitude_changes']:
        ac = periodicity['amplitude_changes'][0]
        print(f"   First amplitude change at t={ac['time']:.1f}s:")
        print(f"     {ac['from_amp']:.3f} → {ac['to_amp']:.3f} ({ac['rel_change']*100:.1f}% change)")
    
    return periodicity

def test_frequency_component(Sxx, freqs, times):
    """Test frequency component extraction for slider"""
    print("\n" + "=" * 60)
    print("TEST 5: Frequency Component Extraction")
    print("=" * 60)
    
    # Select a frequency in the middle
    idx = len(freqs) // 2
    selected_freq = freqs[idx]
    component = Sxx[idx, :]
    
    print(f"✅ Selected frequency: {selected_freq:.2f} Hz (index {idx})")
    print(f"✅ Component amplitude over time:")
    print(f"   Shape: {component.shape}")
    print(f"   Range: {component.min():.3f} - {component.max():.3f}")
    print(f"   Mean: {component.mean():.3f}")
    
    # Check data is suitable for plotting
    assert len(component) == len(times), "Component length must match times"
    print(f"✅ Data ready for interactive slider (slider range: 0 - {len(freqs)-1})")

def main():
    """Run all tests"""
    print("\n" + "🔬" * 30)
    print("FastMODA Feature Test Suite")
    print("🔬" * 30 + "\n")
    
    try:
        # Test 1: Basic processing
        x, fs, freqs, times, Sxx = test_basic_processing()
        
        # Test 2: Changepoint detection
        feats, names, cps, bands = test_changepoint_detection(freqs, times, Sxx)
        
        # Test 3: Band frequencies
        band_freqs = test_band_frequencies(Sxx, freqs, times, bands)
        
        # Test 4: Periodicity analysis
        periodicity = test_periodicity_analysis(x, fs, times, cps)
        
        # Test 5: Frequency component
        test_frequency_component(Sxx, freqs, times)
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nFastMODA is ready to use. Run the web UI with:")
        print("  conda run -n open-ce python FastMODA/app.py")
        print("\nThen open: http://127.0.0.1:5000")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
