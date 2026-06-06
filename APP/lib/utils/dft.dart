import 'dart:math' as math;

/// Hann-windowed FFT spectrum for up to 256 samples.
///
/// Uses a Cooley-Tukey radix-2 in-place FFT for power-of-2 sizes (O(N log N)).
/// Falls back to DFT for non-power-of-2 sizes (rare in practice at N≤256).
///
/// Input  map keys: 'data' (List<double>), 'fs' (double, sample rate in Hz).
/// Output map keys: 'mags', 'delta', 'theta', 'alpha', 'beta', 'gamma',
///                  'dominant' (Hz), 'quality' (0–100), 'entropy', 'flatness'.
///
/// Top-level function so Flutter's compute() can spawn it in an isolate.
Map<String, dynamic> dftWorker(Map<String, dynamic> args) {
  final data = List<double>.from(args['data'] as List);
  final fs = (args['fs'] as num).toDouble();
  final N = data.length.clamp(2, 256);
  final hN = N ~/ 2;

  // Hann window — reduces spectral leakage
  final windowed = List.generate(
      N, (i) => data[i] * 0.5 * (1 - math.cos(2 * math.pi * i / (N - 1))));

  // Compute magnitude spectrum via FFT (or fallback DFT)
  final mags = _spectrum(windowed, N, hN);

  final freqRes = fs / N;

  // Band ranges: caller can override via args['bands'] (List of [low, high] pairs).
  // Falls back to EEG defaults so existing call-sites without 'bands' still work.
  final bandsArg = args['bands'] as List?;
  final bandRanges = bandsArg != null
      ? [for (final b in bandsArg) ((b[0] as num).toDouble(), (b[1] as num).toDouble())]
      : [(0.5, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 30.0), (30.0, 100.0)];

  double bandPower(double fLow, double fHigh) {
    final lo = (fLow / freqRes).floor().clamp(0, hN - 1);
    final hi = (fHigh / freqRes).ceil().clamp(0, hN - 1);
    var s = 0.0;
    for (int k = lo; k <= hi; k++) s += mags[k] * mags[k];
    return s;
  }

  // Skip DC (k=0) when finding the dominant frequency
  int maxK = 1;
  for (int k = 2; k < hN; k++) {
    if (mags[k] > mags[maxK]) maxK = k;
  }

  final totalPow = mags.fold(0.0, (a, b) => a + b * b);
  final mean = totalPow / hN;
  final quality = mean > 0
      ? ((mags[maxK] * mags[maxK]) / mean / hN * 100).clamp(0.0, 100.0)
      : 0.0;

  // Spectral entropy & flatness — skip DC bin (k=0) to avoid bias.
  const eps = 1e-12;
  final specBins = hN - 1;
  final totalSpecPow = mags.skip(1).fold(0.0, (a, b) => a + b * b);

  double entropy = 0.0;
  if (totalSpecPow > eps) {
    for (int k = 1; k < hN; k++) {
      final p = (mags[k] * mags[k]) / totalSpecPow;
      if (p > eps) entropy -= p * math.log(p);
    }
    entropy /= math.log(specBins);
  }

  // Wiener spectral flatness
  double logSum = 0.0;
  double linSum = 0.0;
  for (int k = 1; k < hN; k++) {
    logSum += math.log(mags[k] + eps);
    linSum += mags[k];
  }
  final geoMean = math.exp(logSum / specBins);
  final arithMean = linSum / specBins;
  final flatness =
      arithMean > eps ? (geoMean / arithMean).clamp(0.0, 1.0) : 0.0;

  return {
    'mags': mags,
    'delta': bandPower(bandRanges[0].$1, bandRanges[0].$2),
    'theta': bandPower(bandRanges[1].$1, bandRanges[1].$2),
    'alpha': bandPower(bandRanges[2].$1, bandRanges[2].$2),
    'beta':  bandPower(bandRanges[3].$1, bandRanges[3].$2),
    'gamma': bandPower(bandRanges[4].$1, bandRanges[4].$2),
    'dominant': maxK * freqRes,
    'quality': quality,
    'entropy': entropy,
    'flatness': flatness,
    'rhythmicity': 1.0 - flatness,
  };
}

/// Returns the one-sided magnitude spectrum (bins 0..hN-1), normalised by N.
List<double> _spectrum(List<double> windowed, int N, int hN) {
  if (_isPow2(N)) {
    return _fftMags(windowed, N, hN);
  }
  return _dftMags(windowed, N, hN);
}

bool _isPow2(int n) => n > 0 && (n & (n - 1)) == 0;

/// Cooley-Tukey radix-2 in-place FFT — O(N log N).
List<double> _fftMags(List<double> x, int N, int hN) {
  final re = List<double>.from(x);
  final im = List.filled(N, 0.0);

  // Bit-reversal permutation
  int j = 0;
  for (int i = 1; i < N; i++) {
    int bit = N >> 1;
    for (; (j & bit) != 0; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      final tr = re[i]; re[i] = re[j]; re[j] = tr;
      final ti = im[i]; im[i] = im[j]; im[j] = ti;
    }
  }

  // Butterfly stages
  for (int len = 2; len <= N; len <<= 1) {
    final half = len >> 1;
    final angle = -2 * math.pi / len;
    final wRe = math.cos(angle);
    final wIm = math.sin(angle);
    for (int i = 0; i < N; i += len) {
      double curRe = 1.0, curIm = 0.0;
      for (int k = 0; k < half; k++) {
        final uRe = re[i + k],       uIm = im[i + k];
        final vRe = re[i + k + half], vIm = im[i + k + half];
        final tRe = curRe * vRe - curIm * vIm;
        final tIm = curRe * vIm + curIm * vRe;
        re[i + k]        = uRe + tRe;
        im[i + k]        = uIm + tIm;
        re[i + k + half] = uRe - tRe;
        im[i + k + half] = uIm - tIm;
        final nextRe = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = nextRe;
      }
    }
  }

  return List.generate(hN, (k) {
    final mag = math.sqrt(re[k] * re[k] + im[k] * im[k]);
    return mag / N;
  });
}

/// Fallback naive DFT for non-power-of-2 sizes — O(N·hN).
List<double> _dftMags(List<double> x, int N, int hN) {
  final mags = List<double>.filled(hN, 0.0);
  for (int k = 0; k < hN; k++) {
    double re = 0, im = 0;
    final factor = 2 * math.pi * k / N;
    for (int n = 0; n < N; n++) {
      re += x[n] * math.cos(factor * n);
      im -= x[n] * math.sin(factor * n);
    }
    mags[k] = math.sqrt(re * re + im * im) / N;
  }
  return mags;
}
