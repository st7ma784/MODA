#!/usr/bin/env python
"""
Example: Complete Multi-Modal Feature Extraction & Neural Network Workflow

This script demonstrates:
1. Running all analyses on a signal
2. Extracting features from all modalities
3. Creating and using the neural network for diagnosis
"""

import numpy as np
from fastmoda import load_signal, sliding_fft, detect_changepoints, compute_band_powers
from fastmoda.analysis_gpu import (
    compute_instantaneous_phase_gpu,
    stft_gpu,
    cwt_gpu,
    wavelet_coherence_gpu,
    bispectrum_gpu
)
from fastmoda.feature_extraction import extract_all_features, normalize_features
from fastmoda.diagnosis_network import create_diagnosis_model, DiagnosisTrainer
import torch
from torch.utils.data import DataLoader, TensorDataset


def run_all_analyses(x, fs):
    """Run all analyses and collect results"""
    print("Running all analyses...")

    # 1. Spectral Analysis
    freqs, times, Sxx = sliding_fft(x, fs, win_s=1.0)
    feats, names = compute_band_powers(Sxx, freqs, [
        (0.5, 4, 'delta'), (4, 8, 'theta'), (8, 13, 'alpha'),
        (13, 30, 'beta'), (30, 100, 'gamma')
    ])
    cps = detect_changepoints(feats, pen=10)

    # 2. Phase Analysis
    from scipy.signal import hilbert
    analytic = hilbert(x)
    phase_data = {
        'amplitude': np.abs(analytic),
        'phase': np.angle(analytic),
        'frequency': np.diff(np.unwrap(np.angle(analytic))) * fs / (2 * np.pi)
    }
    phase_data['frequency'] = np.concatenate([[phase_data['frequency'][0]],
                                               phase_data['frequency']])

    # 3. STFT
    from scipy.signal import stft as scipy_stft
    stft_freqs, stft_times, stft_Zxx = scipy_stft(x, fs, window='hann',
                                                   nperseg=256, noverlap=128)
    stft_Sxx = np.abs(stft_Zxx)

    # 4. Wavelet
    from scipy import signal
    wav_freqs = np.logspace(np.log10(0.5), np.log10(50), 50)
    wav_times = np.arange(len(x)) / fs
    widths = fs / wav_freqs
    cwt_matrix = signal.cwt(x, signal.morlet2, widths)
    cwt_mag = np.abs(cwt_matrix)

    # 5. Coherence
    from scipy import signal as sp_signal
    delay = int(0.1 * fs)
    x1, x2 = x[:-delay], x[delay:]
    coh_freqs, coherence_vals = sp_signal.coherence(x1, x2, fs, nperseg=256)
    coh_times = np.arange(min(len(x1), len(x2))) / fs
    coherence = np.outer(coherence_vals, np.ones(len(coh_times)))

    # 6. Bispectrum (simplified CPU version)
    nfft = 256
    overlap = 0.5
    hop = int(nfft * (1 - overlap))
    n_segments = (len(x) - nfft) // hop + 1
    n_freq = nfft // 2 + 1
    bispectrum = np.zeros((n_freq, n_freq), dtype=np.complex64)

    for i in range(n_segments):
        start = i * hop
        segment = x[start:start + nfft] * np.hanning(nfft)
        X = np.fft.rfft(segment)
        for f1 in range(n_freq):
            for f2 in range(n_freq):
                f3 = f1 + f2
                if f3 < n_freq:
                    bispectrum[f1, f2] += X[f1] * X[f2] * np.conj(X[f3])

    bispectrum /= n_segments
    bicoherence = np.abs(bispectrum)
    bispec_freqs = np.fft.rfftfreq(nfft, 1/fs)

    # Collect all results
    analysis_results = {
        'spectral': {
            'freqs': freqs,
            'spec_data': Sxx,
            'times': times,
            'changepoints': cps,
            'bands': [(0.5, 4, 'delta'), (4, 8, 'theta'), (8, 13, 'alpha'),
                      (13, 30, 'beta'), (30, 100, 'gamma')]
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
            'freqs': bispec_freqs,
            'bicoherence': bicoherence,
            'bispectrum': bispectrum
        }
    }

    return analysis_results


def main():
    """Main demonstration workflow"""

    # ==================== Step 1: Load and Analyze Signal ====================
    print("="*60)
    print("STEP 1: Loading and analyzing test signal")
    print("="*60)

    # Create test signal
    fs = 100  # Hz
    t = np.linspace(0, 10, 1000)
    x = (np.sin(2 * np.pi * 2 * t) +          # 2 Hz component
         0.5 * np.sin(2 * np.pi * 10 * t))    # 10 Hz component

    # Add 30 Hz burst at t=4-6s
    burst_mask = (t >= 4) & (t <= 6)
    x[burst_mask] += 0.3 * np.sin(2 * np.pi * 30 * t[burst_mask])

    print(f"Signal: {len(x)} samples, {fs} Hz, {len(x)/fs:.1f}s duration\n")

    # Run all analyses
    analysis_results = run_all_analyses(x, fs)

    # ==================== Step 2: Extract Features ====================
    print("="*60)
    print("STEP 2: Extracting features from all modalities")
    print("="*60)

    feature_vector, feature_names = extract_all_features(analysis_results)
    normalized_features, feat_mean, feat_std = normalize_features(feature_vector)

    print(f"Total features extracted: {len(feature_names)}")
    print("\nFeature distribution:")
    modality_counts = {}
    for name in feature_names:
        modality = name.split('_')[0]
        modality_counts[modality] = modality_counts.get(modality, 0) + 1

    for modality, count in sorted(modality_counts.items()):
        print(f"  {modality}: {count} features")

    print("\nTop 10 features by value:")
    top_indices = np.argsort(np.abs(feature_vector))[-10:][::-1]
    for idx in top_indices:
        print(f"  {feature_names[idx]}: {feature_vector[idx]:.4f}")

    # ==================== Step 3: Create Neural Network ====================
    print("\n" + "="*60)
    print("STEP 3: Creating neural network model")
    print("="*60)

    # Create model for binary classification (e.g., normal vs abnormal)
    model = create_diagnosis_model(feature_names, n_classes=2)

    print(f"\nModel architecture:")
    print(f"  - Modality-specific encoders: {len(modality_counts)} modalities")
    print(f"  - Hidden dimension: 128")
    print(f"  - Attention heads: 4")
    print(f"  - Output classes: 2")
    print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ==================== Step 4: Prepare for Training (Example) ====================
    print("\n" + "="*60)
    print("STEP 4: Preparing data for training (example)")
    print("="*60)

    print("\nTo train the model, you need labeled data:")
    print("  1. Collect multiple signals with known diagnoses")
    print("  2. Run all analyses and extract features for each")
    print("  3. Create dataset with features and labels")
    print("  4. Train using DiagnosisTrainer")

    print("\nExample training code:")
    print("""
    # Prepare features by modality
    features_dict = {}
    for name, value in zip(feature_names, normalized_features):
        modality = name.split('_')[0]
        if modality not in features_dict:
            features_dict[modality] = []
        features_dict[modality].append(value)

    features_dict = {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0)
                     for k, v in features_dict.items()}

    # Create trainer
    trainer = DiagnosisTrainer(model, learning_rate=1e-3)

    # Train (with your labeled dataset)
    # for epoch in range(100):
    #     train_loss = trainer.train_epoch(train_loader)
    #     val_metrics = trainer.evaluate(val_loader)
    #     print(f"Epoch {epoch}: Loss={train_loss:.4f}, Acc={val_metrics['accuracy']:.4f}")
    """)

    # ==================== Step 5: Inference Example ====================
    print("\n" + "="*60)
    print("STEP 5: Inference example (with untrained model)")
    print("="*60)

    # Prepare features for model input
    features_dict = {}
    for name, value in zip(feature_names, normalized_features):
        modality = name.split('_')[0]
        if modality not in features_dict:
            features_dict[modality] = []
        features_dict[modality].append(value)

    features_dict = {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0)
                     for k, v in features_dict.items()}

    # Run inference
    model.eval()
    with torch.no_grad():
        output, attention = model(features_dict, return_attention=True)
        probs = torch.softmax(output, dim=1)
        predicted_class = output.argmax(dim=1).item()
        confidence = probs[0, predicted_class].item()

    print(f"\nPrediction (untrained model):")
    print(f"  Predicted class: {predicted_class}")
    print(f"  Confidence: {confidence:.3f}")
    print(f"  Class probabilities: {probs[0].tolist()}")

    print("\nAttention weights (modality importance):")
    attention_avg = attention[0].mean(dim=0).numpy()
    for i, modality in enumerate(sorted(modality_counts.keys())):
        print(f"  {modality}: {attention_avg[i, i]:.4f}")

    print("\n" + "="*60)
    print("✅ Complete! Neural network system ready for training.")
    print("="*60)
    print("\nNext steps:")
    print("  1. Collect labeled training data (signals with known diagnoses)")
    print("  2. Extract features using extract_all_features()")
    print("  3. Train model using DiagnosisTrainer")
    print("  4. Save trained model for deployment")
    print("\nSee NEURAL_NETWORK_DIAGNOSIS.md for detailed documentation.")


if __name__ == "__main__":
    main()
