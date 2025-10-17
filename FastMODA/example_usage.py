"""Example: run FastMODA on a repo example signal and save plots"""
from fastmoda.fastmoda import load_signal, sliding_fft, compute_band_powers, detect_changepoints
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    # try to find a .mat in repository example_sigs
    candidates = [
        'example_sigs/1signal_10Hz.mat',
        'example_sigs/2signals_10Hz.mat',
        'example_sigs/6signals_10Hz.mat'
    ]
    path = None
    for c in candidates:
        if os.path.exists(c):
            path = c
            break
    if path is None:
        print('No example .mat found in repository; please provide a signal file')
        return
    x, fs = load_signal(path)
    freqs, times, Sxx = sliding_fft(x, fs=fs, win_s=0.5, hop_s=0.125)
    bands = [(0, fs*0.25, 'low'), (fs*0.25, fs*0.5, 'mid'), (fs*0.5, fs*0.99, 'high')]
    feats, names = compute_band_powers(Sxx, freqs, bands=bands)
    cps = detect_changepoints(feats, pen=5)

    plt.figure(figsize=(10,4))
    plt.pcolormesh(times, freqs, 20*np.log10(Sxx+1e-12))
    plt.title('Spectrogram')
    plt.xlabel('time (s)')
    plt.ylabel('freq (Hz)')
    plt.tight_layout()
    plt.savefig('fastmoda_spec.png')

    plt.figure()
    for i,name in enumerate(names):
        plt.plot(times, feats[:,i], label=name)
    for cp in cps:
        t = times[cp] if cp < len(times) else times[-1]
        plt.axvline(t, color='k', linestyle='--')
    plt.legend()
    plt.title('Band features')
    plt.savefig('fastmoda_feats.png')
    print('Saved fastmoda_spec.png and fastmoda_feats.png, cps=', cps)

if __name__ == '__main__':
    main()
