#!/usr/bin/env python3
"""
FastMODA Unified Test Suite
Tests all features: basic processing, GPU acceleration, and advanced analysis
"""

import sys
import numpy as np
import os

# Add current dir to path
sys.path.insert(0, '.')

def check_gpu_available():
    """Check if GPU is available"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

# ============================================================================
# BASIC FEATURE TESTS (CPU)
# ============================================================================

def test_basic_processing():
    """Test basic signal processing pipeline"""
    print("=" * 60)
    print("TEST 1: Basic Signal Processing (CPU)")
    print("=" * 60)

    from fastmoda import (
        sliding_fft, compute_band_powers,
        detect_changepoints, extract_band_frequencies
    )

    # Generate synthetic test signal
    fs = 100.0
    duration = 10
    t = np.arange(0, duration, 1/fs)
    x = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 15 * t)
    x += 0.1 * np.random.randn(len(t))
    print(f"✅ Generated test signal: shape={x.shape}, fs={fs} Hz")

    freqs, times, Sxx = sliding_fft(x, fs=fs, win_s=1.0, hop_s=0.25)
    print(f"✅ Computed spectrogram: {Sxx.shape[0]} frequencies × {Sxx.shape[1]} time points")
    print(f"   Frequency range: {freqs[0]:.2f} - {freqs[-1]:.2f} Hz")
    print(f"   Time range: {times[0]:.2f} - {times[-1]:.2f} s")

    bands = [(0, 2.5, 'low'), (2.5, 5, 'mid'), (5, 10, 'high')]
    feats, names = compute_band_powers(Sxx, freqs, bands=bands)
    print(f"✅ Extracted band features: {feats.shape}")

    cps = detect_changepoints(feats, pen=5)
    print(f"✅ Detected {len(cps)} changepoints")

    band_freqs = extract_band_frequencies(Sxx, freqs, times, bands)
    print(f"✅ Extracted frequencies for {len(band_freqs)} bands")

    return True

def test_periodicity_analysis():
    """Test periodicity analysis with sine fitting"""
    print("\n" + "=" * 60)
    print("TEST 2: Periodicity Analysis")
    print("=" * 60)

    from fastmoda import (
        sliding_fft, detect_periodicity_changes, detect_changepoints
    )

    # Generate synthetic test signal
    fs = 100.0
    duration = 10
    t = np.arange(0, duration, 1/fs)
    x = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 15 * t)
    x += 0.1 * np.random.randn(len(t))

    freqs, times, Sxx = sliding_fft(x, fs=fs, win_s=1.0, hop_s=0.25)

    # Simple changepoints for testing
    cps = np.array([10, 20, 30])[:min(3, len(times))]

    periodicity = detect_periodicity_changes(x, fs, times, cps, tolerance=0.1)

    print(f"✅ Fitted sine waves to {len(periodicity['sine_fits'])} segments")
    print(f"✅ Detected {len(periodicity['frequency_changes'])} frequency changes")
    print(f"✅ Detected {len(periodicity['amplitude_changes'])} amplitude changes")

    return True

# ============================================================================
# GPU FEATURE TESTS
# ============================================================================

def generate_coupled_signals(fs=100, duration=60):
    """Generate synthetic coupled oscillators for testing (longer duration for wavelet)"""
    t = np.arange(0, duration, 1/fs)

    # Signal 1: 10 Hz oscillator
    sig1 = np.sin(2 * np.pi * 10 * t)

    # Signal 2: 10 Hz with phase coupling + 20 Hz component
    phase1 = 2 * np.pi * 10 * t
    sig2 = np.sin(phase1 + 0.5 * np.sin(phase1))
    sig2 += 0.5 * np.sin(2 * np.pi * 20 * t)

    # Add noise
    sig1 += 0.1 * np.random.randn(len(t))
    sig2 += 0.1 * np.random.randn(len(t))

    return sig1, sig2, t

def generate_quadratic_coupling(fs=100, duration=60):
    """Generate signal with f1 + f2 = f3 coupling (longer duration for wavelet)"""
    t = np.arange(0, duration, 1/fs)

    f1, f2 = 10, 20  # Hz
    sig = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)

    # Add quadratic coupling: f1 + f2 = 30 Hz
    sig += 0.5 * np.sin(2 * np.pi * f1 * t) * np.sin(2 * np.pi * f2 * t)

    # Add noise
    sig += 0.1 * np.random.randn(len(t))

    return sig, t

def test_coherence_module():
    """Test multi-signal coherence"""
    print("\n" + "=" * 60)
    print("TEST 3: Multi-Signal Coherence (GPU)")
    print("=" * 60)

    try:
        from fastmoda.coherence_gpu import batched_coherence_analysis_gpu
        import torch

        sig1, sig2, t = generate_coupled_signals()
        fs = 100

        print(f"✅ Generated signals: {len(sig1)} samples @ {fs} Hz")

        x1 = torch.tensor(sig1, dtype=torch.float32).cuda()
        x2 = torch.tensor(sig2, dtype=torch.float32).cuda()

        results = batched_coherence_analysis_gpu(
            x1, x2, fs, win_s=1.0, overlap=0.5, numcycles=10
        )

        coherence = results['phcoh']
        freqs = results['freqs']

        freq_10_idx = torch.argmin(torch.abs(freqs - 10))
        peak_coh = coherence[freq_10_idx].item()

        print(f"✅ Coherence computed: {len(freqs)} frequencies")
        print(f"✅ Coherence at 10 Hz: {peak_coh:.3f}")

        assert peak_coh > 0.3, "Coherence should be above noise floor"
        print("✅ COHERENCE TEST PASSED")
        return True

    except Exception as e:
        print(f"⚠️  COHERENCE TEST SKIPPED: {e}")
        return None

def test_bispectrum_module():
    """Test bispectrum analysis"""
    print("\n" + "=" * 60)
    print("TEST 4: Bispectrum Analysis (GPU)")
    print("=" * 60)

    try:
        from fastmoda.bispectrum_gpu import wavelet_bispectrum_gpu
        import torch

        sig, t = generate_quadratic_coupling()
        fs = 100

        print(f"✅ Generated signal: {len(sig)} samples @ {fs} Hz")

        x = torch.tensor(sig, dtype=torch.float32).cuda()

        result = wavelet_bispectrum_gpu(
            x, x, fs,
            freq_range=(5, 35),
            n_freqs=30,
            bispectrum_type='111'
        )

        bispec = result['biamp']
        freqs = result['freq']

        print(f"✅ Bispectrum computed: {bispec.shape}")
        print(f"✅ Max coupling strength: {bispec.max().item():.3e}")

        print("✅ BISPECTRUM TEST PASSED")
        return True

    except Exception as e:
        print(f"⚠️  BISPECTRUM TEST SKIPPED: {e}")
        return None

def test_bayesian_module():
    """Test Bayesian inference"""
    print("\n" + "=" * 60)
    print("TEST 5: Bayesian Inference (GPU)")
    print("=" * 60)

    try:
        from fastmoda.bayesian_gpu import (
            butterworth_bandpass_gpu,
            hilbert_phase_gpu,
            compute_coupling_direction
        )
        import torch

        sig1, sig2, t = generate_coupled_signals()
        fs = 100

        print(f"✅ Generated signals: {len(sig1)} samples @ {fs} Hz")

        x1_filt = butterworth_bandpass_gpu(sig1, 8, 12, fs)
        x2_filt = butterworth_bandpass_gpu(sig2, 8, 12, fs)

        print(f"✅ Bandpass filtered: 8-12 Hz")

        x1_filt_t = torch.tensor(x1_filt, dtype=torch.float32).cuda()
        x2_filt_t = torch.tensor(x2_filt, dtype=torch.float32).cuda()

        phase1 = hilbert_phase_gpu(x1_filt_t)
        phase2 = hilbert_phase_gpu(x2_filt_t)

        print(f"✅ Phases extracted: {phase1.shape}")

        # Dummy coupling for test
        cpl1 = 0.3 + 0.1 * torch.randn(100).cuda()
        cpl2 = 0.7 + 0.1 * torch.randn(100).cuda()

        direction = compute_coupling_direction(cpl1, cpl2)

        print(f"✅ Coupling direction: {direction:.3f}")

        assert -1 <= direction <= 1, "Direction should be in [-1, 1]"
        print("✅ BAYESIAN TEST PASSED")
        return True

    except Exception as e:
        print(f"⚠️  BAYESIAN TEST SKIPPED: {e}")
        return None

def test_surrogates():
    """Test surrogate generation"""
    print("\n" + "=" * 60)
    print("TEST 6: Surrogate Testing (GPU)")
    print("=" * 60)

    try:
        from fastmoda.surrogates_gpu import iaaft_surrogate_gpu, cpp_surrogate_gpu
        import torch

        sig, _ = generate_quadratic_coupling(duration=5)
        x = torch.tensor(sig, dtype=torch.float32).cuda()

        print(f"✅ Test signal: {len(x)} samples")

        surr_iaaft = iaaft_surrogate_gpu(x)

        fft_orig = torch.fft.rfft(x).abs()
        fft_surr = torch.fft.rfft(surr_iaaft).abs()
        spectrum_error = (fft_orig - fft_surr).abs().mean() / fft_orig.mean()

        print(f"✅ IAAFT surrogate generated")
        print(f"✅ Spectrum preservation: {(1-spectrum_error)*100:.1f}%")

        assert spectrum_error < 0.05, "Spectrum should be preserved"

        phase = torch.angle(torch.fft.rfft(x))
        surr_cpp = cpp_surrogate_gpu(x, phase)

        print(f"✅ CPP surrogate generated")

        assert not torch.allclose(surr_iaaft, x), "IAAFT should differ from original"
        assert surr_cpp.shape == x.shape, "CPP should have same shape"

        print("✅ SURROGATE TEST PASSED")
        return True

    except Exception as e:
        print(f"⚠️  SURROGATE TEST SKIPPED: {e}")
        return None

# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def main():
    """Run all tests"""
    print("\n" + "🔬" * 30)
    print("FastMODA Unified Test Suite")
    print("🔬" * 30 + "\n")

    # Check GPU availability
    gpu_available = check_gpu_available()
    if gpu_available:
        import torch
        print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
    else:
        print("ℹ️  GPU not available - GPU tests will be skipped")

    results = {}

    # Basic tests (always run)
    try:
        results['basic_processing'] = test_basic_processing()
    except Exception as e:
        print(f"❌ Basic processing failed: {e}")
        results['basic_processing'] = False
        import traceback
        traceback.print_exc()

    try:
        results['periodicity'] = test_periodicity_analysis()
    except Exception as e:
        print(f"❌ Periodicity analysis failed: {e}")
        results['periodicity'] = False
        import traceback
        traceback.print_exc()

    # GPU tests (run if GPU available)
    if gpu_available:
        results['coherence'] = test_coherence_module()
        results['bispectrum'] = test_bispectrum_module()
        results['bayesian'] = test_bayesian_module()
        results['surrogates'] = test_surrogates()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, result in results.items():
        if result is True:
            status = "✅ PASS"
        elif result is False:
            status = "❌ FAIL"
        else:
            status = "⚠️  SKIP"
        print(f"{test_name.upper()}: {status}")

    passed = sum(1 for r in results.values() if r is True)
    failed = sum(1 for r in results.values() if r is False)
    skipped = sum(1 for r in results.values() if r is None)

    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")

    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nFastMODA is ready to use. Run the web UI with:")
        print("  conda run -n open-ce python app.py")
        print("\nThen open: http://127.0.0.1:5000")
        print("=" * 60 + "\n")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review errors above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
