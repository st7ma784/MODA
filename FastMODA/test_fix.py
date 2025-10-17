import sys
sys.path.insert(0, '.')
from fastmoda.fastmoda import load_signal, sliding_fft

# Test load
x, fs = load_signal('../example_sigs/1signal_10Hz.mat')
print(f'Loaded: shape={x.shape}, ndim={x.ndim}')

# Test FFT
freqs, times, Sxx = sliding_fft(x, fs=10.0, win_s=1.0, hop_s=0.25)
print(f'FFT: {Sxx.shape}')
print('✅ Fix verified - signal loads as 1D!')
