# Neural Network for Automated Diagnosis

## Overview

The FastMODA diagnosis system extracts features from multiple analysis modalities and uses a graph-aware neural network to provide automated diagnosis. The network understands relationships between physiological parameters for improved accuracy.

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │   Multi-Modal Signal Analyses      │
                    └─────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │   Feature Extraction          │
                    │   (~100-200 features total)   │
                    └───────────────┬───────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            │                       │                       │
     ┌──────▼───────┐      ┌───────▼────────┐     ┌───────▼──────┐
     │  Spectral    │      │  Phase         │     │  Wavelet     │
     │  Features    │      │  Features      │     │  Features    │
     │  (~25)       │      │  (~15)         │     │  (~20)       │
     └──────┬───────┘      └───────┬────────┘     └───────┬──────┘
            │                      │                       │
            │       Modality-Specific Encoders           │
            │              (Dense + LayerNorm)            │
            └──────────────────────┼───────────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  Cross-Modal Attention      │
                    │  (learns inter-modality     │
                    │   relationships)            │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  Parameter Relationship     │
                    │  Graph Neural Network       │
                    │  (encodes freq-power,       │
                    │   phase-coherence links)    │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  Classification Head        │
                    │  (Dense layers + dropout)   │
                    └──────────────┬──────────────┘
                                   │
                              ┌────▼─────┐
                              │  Output  │
                              │  (Class  │
                              │   or     │
                              │  Score)  │
                              └──────────┘
```

## Feature Extraction

### Features Per Modality

#### 1. Spectral Analysis (~25 features)
- **Frequency domain**:
  - Dominant frequency
  - Spectral centroid (center of mass)
  - Spectral spread (bandwidth)
  - Spectral entropy (complexity)
  - Spectral flatness (tonality vs noise)

- **Peak characteristics**:
  - Number of spectral peaks
  - Max/mean peak prominence

- **Band powers**:
  - Delta (0.5-4 Hz)
  - Theta (4-8 Hz)
  - Alpha (8-13 Hz)
  - Beta (13-30 Hz)
  - Gamma (30-100 Hz)

- **Temporal**:
  - Number of changepoints
  - Changepoint density
  - Spectral variability over time

- **Ratios**:
  - High/low frequency ratio

#### 2. Phase Analysis (~15 features)
- **Instantaneous frequency**:
  - Mean, std, median, IQR
  - Frequency range
  - Frequency modulation index

- **Instantaneous amplitude**:
  - Mean, std, coefficient of variation
  - Amplitude range
  - Amplitude modulation index

- **Phase properties**:
  - Phase coherence (Kuramoto order parameter)
  - Phase concentration
  - Phase entropy
  - Phase-amplitude coupling strength

#### 3. STFT (~10 features)
- Temporal spectral centroid (mean, std, trend)
- Temporal spectral spread (mean, std)
- Spectral flux (mean, std)
- Temporal modulation
- Time-frequency concentration

#### 4. Wavelet (~20 features)
- Scale-averaged power (mean, std)
- Frequency-averaged power (mean, std)
- Multi-scale entropy (at 5 scales)
- Ridge strength
- Dominant scale (mean, std)
- Scale energy entropy
- Low/Mid/High scale energy ratios

#### 5. Coherence (~10 features)
- Mean, std, max coherence
- High coherence fraction
- Peak coherence frequency
- Band-specific coherence (5 bands)
- Temporal coherence variability

#### 6. Bispectrum (~8 features)
- Max, mean, std bicoherence
- Strong coupling fraction
- Peak coupling frequencies (f1, f2, f_sum)
- Diagonal/off-diagonal bicoherence
- Coupling entropy

**Total: ~100-200 features** (depending on which analyses are run)

## Neural Network Components

### 1. Modality-Specific Encoders

Each modality (spectral, phase, etc.) has its own encoder:

```python
nn.Sequential(
    nn.Linear(input_dim, 128),
    nn.LayerNorm(128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 128)
)
```

**Purpose**: Learn modality-specific representations before fusion

### 2. Cross-Modal Attention

Multi-head attention mechanism that learns relationships between modalities:

```python
Attention(Q, K, V) = softmax(QK^T / √d) V
```

**Purpose**:
- Learn which modalities are most relevant for diagnosis
- Capture inter-modal dependencies (e.g., spectral peaks correlate with phase characteristics)

**Output**: Attention weights showing modality importance

### 3. Parameter Relationship Graph

Graph Neural Network that encodes known physiological/physical relationships:

**Known relationships**:
```
frequency ←→ power (spectral relationship)
phase ←→ frequency (phase-frequency coupling)
coherence ←→ phase_difference (synchronization)
low_freq_power ←→ high_freq_power (cross-frequency coupling)
amplitude ←→ frequency (amplitude modulation)
```

**Architecture**:
- Learnable adjacency matrix
- Message passing between parameter groups
- Aggregation and update functions

**Purpose**: Incorporate domain knowledge about parameter relationships

### 4. Classification/Regression Head

Final layers for prediction:

```python
nn.Sequential(
    nn.Linear(n_modalities * 128, 256),
    nn.LayerNorm(256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, n_classes)
)
```

## Usage

### 1. Feature Extraction

```python
from fastmoda.feature_extraction import extract_all_features, normalize_features

# Collect analysis results
analysis_results = {
    'spectral': {
        'freqs': freqs,
        'spec_data': Sxx,
        'times': times,
        'changepoints': cps,
        'bands': bands
    },
    'phase': {
        'phase': phase,
        'amplitude': amplitude,
        'inst_freq': inst_freq,
        'fs': fs
    },
    # ... other modalities
}

# Extract features
feature_vector, feature_names = extract_all_features(analysis_results)

# Normalize
normalized_features, mean, std = normalize_features(feature_vector)
```

### 2. Create Model

```python
from fastmoda.diagnosis_network import create_diagnosis_model

# For classification (e.g., normal vs abnormal)
model = create_diagnosis_model(feature_names, n_classes=2)

# For regression (e.g., severity score)
model = create_diagnosis_model(feature_names, n_classes=1)
```

### 3. Training (when you have labeled data)

```python
from fastmoda.diagnosis_network import DiagnosisTrainer
import torch
from torch.utils.data import DataLoader, TensorDataset

# Prepare data
# features_dict: Dict[str, torch.Tensor] for each sample
# labels: torch.Tensor

train_dataset = TensorDataset(features_dict, labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Create trainer
trainer = DiagnosisTrainer(model, learning_rate=1e-3)

# Train
for epoch in range(100):
    train_loss = trainer.train_epoch(train_loader)
    val_metrics = trainer.evaluate(val_loader)

    print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Acc={val_metrics['accuracy']:.4f}")
```

### 4. Inference

```python
# Extract features from new signal
features, names = extract_all_features(new_analysis_results)
normalized, _, _ = normalize_features(features, mean, std)  # Use training mean/std

# Convert to dict by modality
features_dict = {}
for name, value in zip(names, normalized):
    modality = name.split('_')[0]
    if modality not in features_dict:
        features_dict[modality] = []
    features_dict[modality].append(value)

features_dict = {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0)
                 for k, v in features_dict.items()}

# Predict
model.eval()
with torch.no_grad():
    output, attention = model(features_dict, return_attention=True)

    if n_classes == 1:
        prediction = output.item()
        print(f"Severity score: {prediction:.3f}")
    else:
        probs = torch.softmax(output, dim=1)
        predicted_class = output.argmax(dim=1).item()
        confidence = probs[0, predicted_class].item()
        print(f"Predicted class: {predicted_class}, Confidence: {confidence:.3f}")
```

### 5. Feature Importance

```python
# Get feature importance scores
importance = model.get_feature_importance(features_dict)

print("Modality importance:")
for modality, score in sorted(importance.items(), key=lambda x: -x[1]):
    print(f"  {modality}: {score:.4f}")
```

## Training Data Requirements

### Minimum Dataset Size

- **Classification**: 100+ samples per class (preferably 500+)
- **Regression**: 500+ samples (preferably 1000+)

### Data Collection

1. **Labeled signals**: Time-series data with ground truth labels
2. **Quality control**: Remove artifacts, noise, bad recordings
3. **Balanced classes**: Equal or weighted representation
4. **Cross-validation**: 5-10 fold for robust evaluation

### Label Examples

**Classification tasks**:
- Normal vs Abnormal
- Disease subtypes (Type A, B, C)
- Severity levels (Mild, Moderate, Severe)

**Regression tasks**:
- Severity score (0-100)
- Recovery time (days)
- Biomarker levels

## Model Performance Metrics

### Classification
- Accuracy
- Precision, Recall, F1-score (per class)
- ROC-AUC
- Confusion matrix
- Cross-validated performance

### Regression
- MSE, RMSE
- MAE (Mean Absolute Error)
- R² score
- Prediction intervals

### Interpretability
- Feature importance (integrated gradients)
- Attention weights (which modalities matter)
- SHAP values (feature contributions)

## Example: EEG Sleep Stage Classification

```python
# 1. Collect labeled EEG data
#    Labels: Wake, N1, N2, N3, REM (5 classes)

# 2. Run all analyses on each 30-second epoch
for epoch in eeg_epochs:
    results = {
        'spectral': run_spectral_analysis(epoch),
        'phase': run_phase_analysis(epoch),
        'wavelet': run_wavelet_analysis(epoch),
        # ...
    }

    features, names = extract_all_features(results)
    all_features.append(features)
    all_labels.append(epoch.label)  # Sleep stage

# 3. Train model
model = create_diagnosis_model(names, n_classes=5)
trainer = DiagnosisTrainer(model)

# 4. Evaluate
#    Expected performance: 80-90% accuracy for sleep staging
#    (comparable to expert human scorers)
```

## Advanced Features

### 1. Multi-Task Learning

Train on multiple related tasks simultaneously:

```python
model = MultiTaskDiagnosisNetwork(
    feature_dims=modality_dims,
    tasks={
        'classification': 3,  # 3 classes
        'severity': 1,        # regression
        'subtype': 4          # 4 subtypes
    }
)
```

### 2. Temporal Modeling

For sequential data (multiple time windows):

```python
model = TemporalDiagnosisNetwork(
    feature_dims=modality_dims,
    sequence_length=10,  # 10 time windows
    lstm_hidden=128
)
```

### 3. Uncertainty Quantification

Get prediction confidence intervals:

```python
# Monte Carlo dropout
predictions = []
model.train()  # Keep dropout active
for _ in range(100):
    pred = model(features)
    predictions.append(pred)

mean_pred = np.mean(predictions)
std_pred = np.std(predictions)
confidence_interval = (mean_pred - 2*std_pred, mean_pred + 2*std_pred)
```

## Parameter Relationship Examples

### Physiological Relationships

1. **Frequency-Power coupling**:
   - Higher frequency → typically lower power (1/f noise)
   - Pathological states may violate this

2. **Phase-Frequency coupling**:
   - Phase of slow oscillation modulates amplitude of fast oscillation
   - Important in neural communication

3. **Cross-Frequency coupling**:
   - Delta power ←→ Gamma power (in sleep)
   - Theta power ←→ Gamma power (in cognition)

4. **Coherence-Phase**:
   - High coherence → consistent phase relationship
   - Low coherence → variable phase

### Graph Encoding

The graph neural network learns these relationships:

```python
# Adjacency matrix (learned during training)
#              freq  power  phase  coher  coupling
# frequency  [[1.0,  0.8,   0.6,   0.3,   0.4    ],
#  power      [0.8,  1.0,   0.4,   0.2,   0.5    ],
#  phase      [0.6,  0.4,   1.0,   0.7,   0.3    ],
#  coherence  [0.3,  0.2,   0.7,   1.0,   0.6    ],
#  coupling   [0.4,  0.5,   0.3,   0.6,   1.0    ]]
```

Higher values = stronger learned relationship

## Best Practices

### 1. Data Preprocessing
- Remove artifacts before analysis
- Ensure consistent sampling rates
- Normalize signal amplitudes
- Use surrogate testing to validate features

### 2. Feature Selection
- Start with all modalities
- Use feature importance to identify key features
- Remove highly correlated features (>0.95 correlation)
- Consider domain expertise

### 3. Model Training
- Use cross-validation
- Monitor for overfitting (train vs val loss)
- Apply early stopping
- Use learning rate scheduling
- Try different architectures

### 4. Validation
- Test on completely held-out data
- Compare to baseline methods
- Validate on different datasets/populations
- Check for dataset shift

### 5. Deployment
- Save model with feature normalization parameters
- Implement input validation
- Provide uncertainty estimates
- Monitor performance in production

## Limitations and Future Work

### Current Limitations
- Requires labeled training data
- May overfit on small datasets
- Computational cost for training
- Black-box nature (partially addressed by attention/importance)

### Future Enhancements
- **Transfer learning**: Pre-train on large datasets
- **Few-shot learning**: Learn from few examples
- **Active learning**: Intelligently select samples to label
- **Federated learning**: Train across institutions without sharing data
- **Explainable AI**: Better interpretability methods

## References

### Neural Network Architectures
- Vaswani et al. (2017): "Attention Is All You Need"
- Kipf & Welling (2017): "Semi-Supervised Classification with Graph Convolutional Networks"

### Medical Signal Analysis
- Schirrmeister et al. (2017): "Deep learning with convolutional neural networks for EEG decoding"
- Roy et al. (2019): "Deep learning-based electroencephalography analysis"

### Multi-Modal Learning
- Baltrusaitis et al. (2019): "Multimodal Machine Learning: A Survey and Taxonomy"
- Ngiam et al. (2011): "Multimodal Deep Learning"

## Summary

The FastMODA diagnosis system:
- ✅ Extracts 100-200 interpretable features from multi-modal analyses
- ✅ Uses graph neural networks to encode parameter relationships
- ✅ Employs cross-modal attention to learn modality importance
- ✅ Provides feature importance and attention visualizations
- ✅ Supports both classification and regression tasks
- ✅ Designed for clinical/research applications

**Ready for training when you have labeled data!**
