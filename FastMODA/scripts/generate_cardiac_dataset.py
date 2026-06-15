#!/usr/bin/env python
"""Generate a synthetic ECG dataset for training per-condition classifiers.

Implements the signal models described in
docs/fastmoda/CARDIAC_ARRHYTHMIA_MODELING_PLAN.md: normal sinus rhythm,
bradycardia, atrial fibrillation (mild/moderate/severe), and premature
ventricular contractions (bigeminy/trigeminy).

Writes one .npy file per sample under <output-dir>/<condition>/ plus a
metadata.csv describing each sample (filepath, condition, bpm, duration, fs).

Usage:
    python scripts/generate_cardiac_dataset.py --n-samples 100
"""

import argparse
import csv
import os

import numpy as np


def qrs_complex(t, center, amplitude=1.0, width=0.08):
    """A QRS-like spike plus a small trailing T-wave bump, centered at `center`."""
    qrs = amplitude * np.exp(-((t - center) ** 2) / (2 * (width / 6) ** 2))
    t_wave = 0.3 * amplitude * np.exp(-((t - center - 0.2) ** 2) / (2 * 0.05 ** 2))
    return qrs + t_wave


class CardiacSignalGenerator:
    """Generates synthetic single-lead ECG-like signals for several conditions."""

    def __init__(self, fs=250.0):
        self.fs = float(fs)

    def _beat_signal(self, t, beat_times, beat_amplitudes=None, beat_widths=None):
        signal = np.zeros_like(t)
        for i, bt in enumerate(beat_times):
            amplitude = beat_amplitudes[i] if beat_amplitudes is not None else 1.0
            width = beat_widths[i] if beat_widths is not None else 0.08
            signal += qrs_complex(t, bt, amplitude=amplitude, width=max(width, 0.01))
        return signal

    def generate_normal(self, duration, bpm=70.0, hrv=0.05):
        """Regular sinus rhythm with mild respiratory heart-rate modulation."""
        fs = self.fs
        n = int(duration * fs)
        t = np.linspace(0, duration, n, endpoint=False)

        hr_base = bpm / 60.0
        resp_freq = 0.25
        hr_modulation = hr_base * (1 + 0.1 * np.sin(2 * np.pi * resp_freq * t))

        beat_times = []
        current_time = 0.0
        while current_time < duration:
            idx = min(int(current_time * fs), n - 1)
            ibi = (1.0 / hr_modulation[idx]) * (1 + hrv * np.random.randn())
            ibi = max(ibi, 0.25)
            current_time += ibi
            if current_time < duration:
                beat_times.append(current_time)

        signal = self._beat_signal(t, beat_times)
        baseline = 0.1 * np.sin(2 * np.pi * 0.2 * t)
        noise = 0.02 * np.random.randn(n)
        return t, signal + baseline + noise, beat_times

    def generate_bradycardia(self, duration):
        """Slow, otherwise-regular sinus rhythm (40-58 bpm)."""
        return self.generate_normal(duration, bpm=np.random.uniform(40, 58), hrv=0.04)

    def generate_afib(self, duration, severity='moderate'):
        """Atrial fibrillation: irregularly-irregular RR intervals + fibrillatory waves."""
        fs = self.fs
        n = int(duration * fs)
        t = np.linspace(0, duration, n, endpoint=False)

        severity_params = {
            'mild':     {'mean_bpm': 110, 'cv': 0.20, 'chaos': 0.3},
            'moderate': {'mean_bpm': 130, 'cv': 0.40, 'chaos': 0.6},
            'severe':   {'mean_bpm': 155, 'cv': 0.55, 'chaos': 0.9},
        }
        params = severity_params[severity]
        mean_hr = params['mean_bpm'] / 60.0

        beat_times = []
        amplitudes = []
        widths = []
        current_time = 0.0
        while current_time < duration:
            random_component = params['cv'] * np.random.randn()
            chaotic_component = params['chaos'] * 0.3 * np.sin(2 * np.pi * 13.7 * current_time)
            ibi = (1.0 / mean_hr) * (1 + random_component + chaotic_component)
            ibi = float(np.clip(ibi, 0.25, 1.5))
            current_time += ibi
            if current_time < duration:
                beat_times.append(current_time)
                amplitudes.append(max(0.3, 1.0 + 0.3 * np.random.randn()))
                widths.append(max(0.02, 0.08 + 0.02 * np.random.randn()))

        signal = self._beat_signal(t, beat_times, amplitudes, widths)

        fibrillatory_waves = np.zeros(n)
        for freq in np.linspace(4, 8, 6):
            phase = np.random.uniform(0, 2 * np.pi)
            fibrillatory_waves += (0.03 * params['chaos']) * np.sin(2 * np.pi * freq * t + phase)

        baseline = 0.1 * np.sin(2 * np.pi * 0.2 * t)
        noise = 0.03 * np.random.randn(n)
        return t, signal + fibrillatory_waves + baseline + noise, beat_times

    def generate_pvcs(self, duration, pattern='bigeminy', base_bpm=72.0):
        """Normal sinus rhythm interspersed with premature ventricular contractions."""
        fs = self.fs
        n = int(duration * fs)
        t = np.linspace(0, duration, n, endpoint=False)

        pvc_period = {'bigeminy': 2, 'trigeminy': 3, 'isolated': 8}[pattern]
        base_ibi = 60.0 / base_bpm

        beat_times = []
        amplitudes = []
        widths = []
        current_time = 0.0
        beat_index = 0
        while current_time < duration:
            beat_index += 1
            is_pvc = (beat_index % pvc_period == 0)
            if is_pvc:
                ibi = base_ibi * 0.6
                amplitude = 1.4 + 0.2 * np.random.randn()
                width = 0.14 + 0.02 * np.random.randn()
            else:
                ibi = base_ibi * (1 + 0.05 * np.random.randn())
                if beat_times and amplitudes and amplitudes[-1] > 1.3:
                    ibi *= 1.25  # compensatory pause after a PVC
                amplitude = 1.0 + 0.05 * np.random.randn()
                width = 0.08
            ibi = max(ibi, 0.2)
            current_time += ibi
            if current_time < duration:
                beat_times.append(current_time)
                amplitudes.append(amplitude)
                widths.append(width)

        signal = self._beat_signal(t, beat_times, amplitudes, widths)
        baseline = 0.1 * np.sin(2 * np.pi * 0.2 * t)
        noise = 0.02 * np.random.randn(n)
        return t, signal + baseline + noise, beat_times


# Maps dataset sub-folder name -> generator callable(generator, duration) -> (t, x, beat_times)
CONDITIONS = {
    'normal': lambda gen, d: gen.generate_normal(d, bpm=np.random.uniform(60, 90)),
    'bradycardia': lambda gen, d: gen.generate_bradycardia(d),
    'mild_afib': lambda gen, d: gen.generate_afib(d, 'mild'),
    'moderate_afib': lambda gen, d: gen.generate_afib(d, 'moderate'),
    'severe_afib': lambda gen, d: gen.generate_afib(d, 'severe'),
    'pvc_bigeminy': lambda gen, d: gen.generate_pvcs(d, 'bigeminy'),
    'pvc_trigeminy': lambda gen, d: gen.generate_pvcs(d, 'trigeminy'),
}


def _mean_bpm(beat_times):
    if len(beat_times) < 2:
        return 0.0
    return 60.0 / float(np.mean(np.diff(beat_times)))


def generate_dataset(output_dir, n_samples=100, duration=30.0, fs=250.0, seed=None):
    if seed is not None:
        np.random.seed(seed)

    output_dir = os.path.abspath(output_dir)
    generator = CardiacSignalGenerator(fs=fs)
    rows = []

    for condition, make_signal in CONDITIONS.items():
        condition_dir = os.path.join(output_dir, condition)
        os.makedirs(condition_dir, exist_ok=True)
        for i in range(n_samples):
            sample_duration = max(5.0, duration + np.random.uniform(-2.0, 2.0))
            _, x, beat_times = make_signal(generator, sample_duration)

            filename = f'patient_{i:04d}_{condition}.npy'
            filepath = os.path.join(condition_dir, filename)
            np.save(filepath, x.astype(np.float64))

            rows.append({
                'filepath': os.path.relpath(filepath, output_dir),
                'condition': condition,
                'bpm': round(_mean_bpm(beat_times), 1),
                'duration': round(sample_duration, 2),
                'fs': fs,
            })
        print(f'Generated {n_samples} "{condition}" samples in {condition_dir}')

    metadata_path = os.path.join(output_dir, 'metadata.csv')
    with open(metadata_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filepath', 'condition', 'bpm', 'duration', 'fs'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote metadata for {len(rows)} samples to {metadata_path}')
    return metadata_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_output = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'cardiac_dataset')
    parser.add_argument('--output-dir', default=default_output,
                         help='Directory to write <condition>/*.npy + metadata.csv (default: %(default)s)')
    parser.add_argument('--n-samples', type=int, default=100, help='Samples per condition')
    parser.add_argument('--duration', type=float, default=30.0, help='Approx. signal duration in seconds')
    parser.add_argument('--fs', type=float, default=250.0, help='Sampling rate in Hz')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    args = parser.parse_args()

    generate_dataset(args.output_dir, n_samples=args.n_samples, duration=args.duration,
                      fs=args.fs, seed=args.seed)


if __name__ == '__main__':
    main()
