# Cardiac Arrhythmia Modeling Plan

## Overview

This document outlines how to create synthetic cardiac signals (ECG-like) to model both **regular heartbeats** and **erratic/arrhythmic patterns** for testing and training the FastMODA neural network.

---

## 🫀 Physiological Background

### Normal Heartbeat Characteristics

**Frequency Components:**
- **Fundamental heart rate**: 0.8-2.0 Hz (48-120 BPM)
  - Resting: ~1.0-1.2 Hz (60-72 BPM)
  - Exercise: ~1.5-2.5 Hz (90-150 BPM)
- **Respiratory sinus arrhythmia**: 0.2-0.4 Hz (breathing-related variation)
- **QRS complex**: ~10-40 Hz (sharp spike of ventricular depolarization)
- **T-wave**: ~2-5 Hz (ventricular repolarization)
- **Baseline wander**: 0.05-0.5 Hz (due to respiration, motion)

**Regularity:**
- Normal: Heart rate variability (HRV) ~5-10% beat-to-beat variation
- Healthy variability follows specific patterns (controlled by autonomic nervous system)

### Arrhythmic Patterns

**1. Atrial Fibrillation (AFib)**
- Irregular, chaotic atrial activity
- Ventricular response: **irregularly irregular** (~100-180 BPM)
- No clear P-waves
- Beat-to-beat intervals vary widely (coefficient of variation >30%)

**2. Premature Ventricular Contractions (PVCs)**
- Extra beats occurring earlier than expected
- Wide QRS complex
- Followed by compensatory pause
- Can be isolated or in patterns (bigeminy, trigeminy)

**3. Ventricular Tachycardia (VT)**
- Very fast (150-250 BPM), regular rhythm
- Wide QRS complexes
- Life-threatening

**4. Heart Block**
- Delayed or absent conduction
- Missed beats
- Can be regular (2:1 block) or irregular

---

## 🎯 Modeling Strategy

### Signal Components

Our synthetic cardiac signal will combine:

```
ECG(t) = Baseline(t) + HeartBeats(t) + Noise(t)
```

Where:
- **Baseline(t)**: Low-frequency drift (0.1-0.5 Hz) from respiration/movement
- **HeartBeats(t)**: Series of QRS complexes with varying inter-beat intervals
- **Noise(t)**: Physiological noise, muscle artifacts (~20-100 Hz)

### Classes to Model

| Class | Description | Heart Rate | Regularity | Key Features |
|-------|-------------|------------|------------|--------------|
| **Normal** | Healthy sinus rhythm | 60-100 BPM | Regular (HRV ~5%) | Clear P-QRS-T, respiratory variation |
| **Mild AFib** | Early atrial fibrillation | 90-130 BPM | Irregular (HRV ~20%) | Some irregularity, mild chaos |
| **Moderate AFib** | Established AFib | 100-150 BPM | Very irregular (HRV ~40%) | Highly variable intervals |
| **Severe AFib** | Uncontrolled AFib | 120-180 BPM | Chaotic (HRV >50%) | Completely irregular, rapid |
| **PVCs** | Premature beats | 70-90 BPM | Regular with extras | Early beats every N beats |
| **Bradycardia** | Slow heart rate | 40-60 BPM | Regular | Slow but organized |

---

## 📐 Mathematical Modeling

### 1. Normal Heartbeat

```python
def generate_normal_ecg(duration, fs=250, bpm=70):
    """Generate normal ECG with respiratory sinus arrhythmia"""
    t = np.linspace(0, duration, int(duration * fs))

    # Base heart rate with respiratory variation
    hr_base = bpm / 60  # Hz
    resp_freq = 0.25  # Hz (15 breaths/min)
    hr_modulation = hr_base * (1 + 0.1 * np.sin(2*np.pi*resp_freq*t))

    # Generate beat times
    beat_times = []
    current_time = 0
    while current_time < duration:
        # Inter-beat interval with small variation
        ibi = 1/hr_modulation[int(current_time*fs)] * (1 + 0.05*np.random.randn())
        current_time += ibi
        if current_time < duration:
            beat_times.append(current_time)

    # Create QRS complexes
    signal = np.zeros_like(t)
    for beat_time in beat_times:
        signal += qrs_complex(t, beat_time, amplitude=1.0, width=0.08)

    # Add baseline wander
    baseline = 0.1 * np.sin(2*np.pi*0.2*t)

    # Add noise
    noise = 0.02 * np.random.randn(len(t))

    return t, signal + baseline + noise, beat_times
```

### 2. Atrial Fibrillation (Erratic)

```python
def generate_afib_ecg(duration, fs=250, severity='moderate'):
    """Generate ECG with atrial fibrillation"""
    t = np.linspace(0, duration, int(duration * fs))

    # Severity-dependent parameters
    params = {
        'mild': {'mean_bpm': 110, 'cv': 0.20, 'chaos': 0.3},
        'moderate': {'mean_bpm': 130, 'cv': 0.40, 'chaos': 0.6},
        'severe': {'mean_bpm': 155, 'cv': 0.55, 'chaos': 0.9}
    }

    p = params[severity]
    mean_hr = p['mean_bpm'] / 60  # Hz

    # Generate irregular beat times
    beat_times = []
    current_time = 0

    while current_time < duration:
        # Highly irregular inter-beat intervals
        # Mix of random + chaotic components
        random_component = np.random.randn() * p['cv']
        chaotic_component = p['chaos'] * np.sin(2*np.pi*13.7*current_time)

        ibi = (1/mean_hr) * (1 + random_component + chaotic_component)
        ibi = np.clip(ibi, 0.3, 1.5)  # Physiological limits

        current_time += ibi
        if current_time < duration:
            beat_times.append(current_time)

    # Create irregular QRS complexes (varying amplitude/width)
    signal = np.zeros_like(t)
    for beat_time in beat_times:
        amplitude = 1.0 + 0.3*np.random.randn()  # Varying amplitude
        width = 0.08 + 0.02*np.random.randn()    # Varying width
        signal += qrs_complex(t, beat_time, amplitude, width)

    # Add fibrillatory waves (chaotic atrial activity 4-6 Hz)
    fib_waves = 0.05 * np.random.randn(len(t))
    for freq in np.linspace(4, 6, 10):
        phase = np.random.rand() * 2 * np.pi
        fib_waves += 0.01 * np.sin(2*np.pi*freq*t + phase)

    # Enhanced baseline wander (more erratic)
    baseline = 0.15 * np.sin(2*np.pi*0.2*t + np.random.randn(len(t))*0.1)

    # Noise
    noise = 0.03 * np.random.randn(len(t))

    return t, signal + fib_waves + baseline + noise, beat_times


def qrs_complex(t, center, amplitude=1.0, width=0.08):
    """Generate a single QRS complex (simplified)"""
    # Gaussian-based QRS
    qrs = amplitude * np.exp(-((t - center)**2) / (2*(width/6)**2))

    # Add T-wave (wider, later, smaller)
    t_wave = 0.3*amplitude * np.exp(-((t - center - 0.2)**2) / (2*0.05**2))

    return qrs + t_wave
```

### 3. Premature Ventricular Contractions

```python
def generate_pvc_ecg(duration, fs=250, pvc_frequency='bigeminy'):
    """Generate ECG with PVCs"""
    t = np.linspace(0, duration, int(duration * fs))

    # Normal beats at 70 BPM
    normal_hr = 70 / 60
    normal_ibi = 1 / normal_hr

    # Generate beat times with PVCs
    beat_times = []
    beat_types = []  # 'N' for normal, 'P' for PVC
    current_time = 0
    beat_count = 0

    pvc_patterns = {
        'bigeminy': 2,    # Every other beat
        'trigeminy': 3,   # Every third beat
        'isolated': 10    # Occasional PVCs
    }

    pvc_interval = pvc_patterns[pvc_frequency]

    while current_time < duration:
        beat_count += 1

        if beat_count % pvc_interval == 0:
            # PVC - comes early
            ibi = normal_ibi * 0.7  # 30% earlier
            beat_type = 'P'
        else:
            # Normal beat
            ibi = normal_ibi * (1 + 0.05*np.random.randn())
            beat_type = 'N'

            # Compensatory pause after PVC
            if beat_types and beat_types[-1] == 'P':
                ibi *= 1.3

        current_time += ibi
        if current_time < duration:
            beat_times.append(current_time)
            beat_types.append(beat_type)

    # Create signal with different QRS for normal vs PVC
    signal = np.zeros_like(t)
    for beat_time, beat_type in zip(beat_times, beat_types):
        if beat_type == 'N':
            signal += qrs_complex(t, beat_time, amplitude=1.0, width=0.08)
        else:  # PVC
            # Wider, different morphology
            signal += qrs_complex(t, beat_time, amplitude=1.5, width=0.15)

    # Baseline and noise
    baseline = 0.1 * np.sin(2*np.pi*0.2*t)
    noise = 0.02 * np.random.randn(len(t))

    return t, signal + baseline + noise, beat_times, beat_types
```

---

## 🗂️ Dataset Structure

### Dataset Organization

```
cardiac_dataset/
├── normal/
│   ├── patient_001_rest.npy
│   ├── patient_002_rest.npy
│   └── ...
├── mild_afib/
│   ├── patient_101_afib_mild.npy
│   └── ...
├── moderate_afib/
│   ├── patient_201_afib_mod.npy
│   └── ...
├── severe_afib/
│   ├── patient_301_afib_severe.npy
│   └── ...
├── pvcs/
│   ├── patient_401_bigeminy.npy
│   ├── patient_402_trigeminy.npy
│   └── ...
├── bradycardia/
│   ├── patient_501_brady.npy
│   └── ...
└── metadata.csv
```

### Metadata Format

```csv
filename,class,bpm,duration,severity,notes
patient_001_rest.npy,normal,68,30,0,healthy_resting
patient_002_rest.npy,normal,72,30,0,healthy_resting
patient_101_afib_mild.npy,afib,110,30,1,early_afib
patient_201_afib_mod.npy,afib,135,30,2,established_afib
patient_301_afib_severe.npy,afib,162,30,3,uncontrolled_afib
patient_401_bigeminy.npy,pvc,75,30,1,bigeminy_pattern
```

### Dataset Size Recommendations

For robust training:
- **Minimum**: 100 samples per class (600 total)
- **Good**: 500 samples per class (3000 total)
- **Excellent**: 1000+ samples per class (6000+ total)

**Variation within each class:**
- Different patients (varying baseline characteristics)
- Different durations (10s, 30s, 60s segments)
- Different severity levels (mild → severe)
- Different noise levels
- Different sampling rates (simulate different devices)

---

## 🧪 Implementation Plan

### Phase 1: Signal Generation Script

Create `generate_cardiac_dataset.py`:

```python
import numpy as np
import os
from tqdm import tqdm

class CardiacSignalGenerator:
    def __init__(self, fs=250):
        self.fs = fs

    def generate_dataset(self, n_samples_per_class=500, duration=30):
        """Generate complete synthetic cardiac dataset"""

        classes = {
            'normal': self.generate_normal,
            'mild_afib': lambda d: self.generate_afib(d, 'mild'),
            'moderate_afib': lambda d: self.generate_afib(d, 'moderate'),
            'severe_afib': lambda d: self.generate_afib(d, 'severe'),
            'pvcs_bigeminy': lambda d: self.generate_pvcs(d, 'bigeminy'),
            'pvcs_trigeminy': lambda d: self.generate_pvcs(d, 'trigeminy'),
        }

        metadata = []

        for class_name, generator in classes.items():
            os.makedirs(f'cardiac_dataset/{class_name}', exist_ok=True)

            print(f"\nGenerating {n_samples_per_class} samples for {class_name}...")
            for i in tqdm(range(n_samples_per_class)):
                # Add variation
                dur = duration + np.random.randint(-5, 5)

                # Generate signal
                t, signal, beat_times = generator(dur)

                # Calculate actual BPM
                if len(beat_times) > 1:
                    avg_ibi = np.mean(np.diff(beat_times))
                    bpm = 60 / avg_ibi
                else:
                    bpm = 0

                # Save
                filename = f'patient_{i:04d}_{class_name}.npy'
                filepath = f'cardiac_dataset/{class_name}/{filename}'
                np.save(filepath, signal)

                # Metadata
                severity = self._get_severity(class_name)
                metadata.append({
                    'filename': filepath,
                    'class': class_name.split('_')[0],
                    'bpm': f'{bpm:.1f}',
                    'duration': dur,
                    'severity': severity,
                    'notes': class_name
                })

        # Save metadata
        import pandas as pd
        df = pd.DataFrame(metadata)
        df.to_csv('cardiac_dataset/metadata.csv', index=False)
        print(f"\n✅ Dataset generated: {len(metadata)} samples")
        return df

    def generate_normal(self, duration):
        # Implementation here
        pass

    def generate_afib(self, duration, severity):
        # Implementation here
        pass

    def generate_pvcs(self, duration, pattern):
        # Implementation here
        pass

    def _get_severity(self, class_name):
        severity_map = {
            'normal': 0,
            'mild_afib': 1,
            'moderate_afib': 2,
            'severe_afib': 3,
            'pvcs_bigeminy': 1,
            'pvcs_trigeminy': 1,
        }
        return severity_map.get(class_name, 0)
```

### Phase 2: Feature Extraction & Visualization

Verify that features capture the differences:

```python
from fastmoda.feature_extraction import extract_all_features
import matplotlib.pyplot as plt

def analyze_sample(signal_path, fs=250):
    """Analyze a single sample"""
    signal = np.load(signal_path)

    # Run all analyses
    analysis_results = run_all_analyses(signal, fs)

    # Extract features
    features, names = extract_all_features(analysis_results)

    # Which features differ between normal and arrhythmic?
    return features, names

# Compare feature distributions
normal_features = []
afib_features = []

for file in normal_files:
    f, n = analyze_sample(file)
    normal_features.append(f)

for file in afib_files:
    f, n = analyze_sample(file)
    afib_features.append(f)

# Plot key discriminative features
plot_feature_comparison(normal_features, afib_features, names)
```

### Phase 3: Neural Network Training

```python
from fastmoda.diagnosis_network import create_diagnosis_model, DiagnosisTrainer

# Load dataset
X_train, y_train, X_val, y_val = load_cardiac_dataset()

# Extract features for all samples
train_features = [extract_all_features(run_all_analyses(x, fs))
                  for x in X_train]

# Create model (6 classes)
model = create_diagnosis_model(feature_names, n_classes=6)

# Train
trainer = DiagnosisTrainer(model, learning_rate=1e-3)

for epoch in range(200):
    train_loss = trainer.train_epoch(train_loader)
    val_metrics = trainer.evaluate(val_loader)

    print(f"Epoch {epoch}: Loss={train_loss:.4f}, "
          f"Acc={val_metrics['accuracy']:.4f}")

    # Early stopping
    if val_metrics['accuracy'] > 0.95:
        break

# Save model
torch.save(model.state_dict(), 'cardiac_arrhythmia_model.pt')
```

### Phase 4: Evaluation & Interpretation

```python
# Confusion matrix
from sklearn.metrics import confusion_matrix, classification_report

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred,
                          target_names=['Normal', 'Mild AFib', 'Mod AFib',
                                       'Severe AFib', 'PVC-Bi', 'PVC-Tri']))

# Feature importance
importance = model.get_feature_importance(test_sample)
print("\nTop 10 discriminative features:")
for name, score in sorted(importance.items(), key=lambda x: -x[1])[:10]:
    print(f"  {name}: {score:.4f}")
```

---

## 🎯 Expected Feature Responses

### Features that Should Discriminate

| Feature | Normal | Mild AFib | Severe AFib | PVCs |
|---------|--------|-----------|-------------|------|
| **phase_coherence** | High (>0.9) | Medium (~0.7) | Low (<0.5) | Medium-High |
| **spectral_entropy** | Low | Medium | High | Low |
| **changepoint_density** | Low | Medium | High | Medium |
| **phase_freq_range** | Narrow | Medium | Wide | Medium |
| **coherence_temporal_std** | Low | Medium | High | Medium |
| **n_spectral_peaks** | 1-2 | 2-3 | 3-5 | 2 |

**Why these features?**

- **Phase coherence**: Measures regularity of oscillations
  - Normal: Very regular heartbeat → high coherence
  - AFib: Chaotic → low coherence

- **Spectral entropy**: Measures frequency complexity
  - Normal: Narrow frequency band → low entropy
  - AFib: Spread across many frequencies → high entropy

- **Changepoint density**: Detects rhythm changes
  - Normal: Few changepoints (just respiratory variation)
  - AFib: Many changepoints (constantly changing)

---

## 📊 Validation Strategy

### 1. Synthetic Data Validation

Before training on real data:
- Generate small test set (10 samples per class)
- Manually verify signals look realistic
- Check feature extraction catches expected patterns
- Ensure classes are separable in feature space

### 2. Cross-Validation

- 5-fold stratified cross-validation
- Ensure each fold has balanced classes
- Report mean ± std performance across folds

### 3. Real Data Testing (Future)

Once trained on synthetic data:
- Test on real ECG databases (PhysioNet, MIT-BIH)
- Compare to clinical annotations
- Fine-tune on real data if needed

---

## 🚀 Quick Start Commands

```bash
# 1. Generate dataset
python generate_cardiac_dataset.py --n-samples 500 --duration 30

# 2. Visualize samples
python visualize_cardiac_samples.py --class-name mild_afib --n-samples 5

# 3. Extract features and check separability
python check_feature_separability.py

# 4. Train model
python train_cardiac_model.py --epochs 200 --batch-size 32

# 5. Evaluate
python evaluate_cardiac_model.py --model cardiac_model.pt --test-set test/

# 6. Test on web interface
python app_tabbed.py
# Upload generated signal → see classification
```

---

## 💡 Advanced Enhancements

### 1. More Realistic ECG Morphology

Use dynamical models:
- McSharry ECG model (differential equations)
- Realistic P-QRS-T waves
- Proper electrical axis

### 2. Patient Variability

Add inter-patient variation:
- Different QRS amplitudes
- Different T-wave morphologies
- Age-related changes
- Medication effects

### 3. Noise Models

Realistic artifacts:
- Baseline wander (patient movement)
- Muscle artifacts (EMG)
- Powerline interference (50/60 Hz)
- Electrode motion

### 4. Multi-Lead ECG

Simulate 12-lead ECG:
- Different leads show different morphologies
- Spatial information
- Better arrhythmia classification

---

## 📚 References

### Physiological Models
- McSharry et al. (2003): "A dynamical model for generating synthetic electrocardiogram signals"
- Goldberger et al. (2000): "PhysioBank, PhysioToolkit, and PhysioNet"

### Arrhythmia Detection
- Acharya et al. (2017): "A deep convolutional neural network model to classify heartbeats"
- Hannun et al. (2019): "Cardiologist-level arrhythmia detection with CNNs"

### Heart Rate Variability
- Task Force (1996): "Heart rate variability: standards of measurement"

---

## ✅ Summary

This plan provides:
1. ✅ Physiologically-motivated signal models
2. ✅ Normal and arrhythmic patterns
3. ✅ Dataset generation strategy
4. ✅ Feature extraction validation
5. ✅ Training and evaluation pipeline
6. ✅ Integration with FastMODA system

**Next step**: Implement `generate_cardiac_dataset.py` with the signal generation functions!
