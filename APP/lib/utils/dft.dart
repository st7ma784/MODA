import 'dart:math' as math;

/// Hann-windowed DFT for up to 256 samples.
///
/// Input  map keys: 'data' (List<double>), 'fs' (double, sample rate in Hz).
/// Output map keys: 'mags', 'delta', 'theta', 'alpha', 'beta', 'gamma',
///                  'dominant' (Hz), 'quality' (0–100).
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

  // Naive DFT — O(N·hN), fine for N≤256 inside an isolate
  final mags = List<double>.filled(hN, 0.0);
  for (int k = 0; k < hN; k++) {
    double re = 0, im = 0;
    final factor = 2 * math.pi * k / N;
    for (int n = 0; n < N; n++) {
      re += windowed[n] * math.cos(factor * n);
      im -= windowed[n] * math.sin(factor * n);
    }
    mags[k] = math.sqrt(re * re + im * im) / N;
  }

  final freqRes = fs / N;

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

  return {
    'mags': mags,
    'delta': bandPower(0.5, 4.0),
    'theta': bandPower(4.0, 8.0),
    'alpha': bandPower(8.0, 12.0),
    'beta': bandPower(12.0, 30.0),
    'gamma': bandPower(30.0, 100.0),
    'dominant': maxK * freqRes,
    'quality': quality,
  };
}
