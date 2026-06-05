import 'dart:math' as math;

/// CUSUM-based changepoint detector with AIC false-positive penalty.
///
/// Detects mean-shift changepoints using the cumulative sum (CUSUM) statistic,
/// which is more sensitive and specific than a variance-ratio test, particularly
/// at low SNR.  An AIC-derived penalty suppresses spurious detections.
///
/// Algorithm:
///   1. Standardise the signal (zero mean, unit variance).
///   2. Compute cumulative sums S[k] = Σ_{i=0}^{k} (x[i] - mean).
///   3. A changepoint at position t maximises |S[t] - S[t-1]|, i.e. the
///      normalised CUSUM magnitude.
///   4. Apply AIC penalty: accept a changepoint only when the improvement
///      in log-likelihood exceeds log(N)/2 (BIC-like threshold, more
///      conservative than AIC but less than MDL in practice).
///   5. Suppress detections closer than `minSep` samples (= half windowSize).
///
/// Input  keys: 'data' (List<double>), optionally 'windowSize' (int, default 32),
///              'threshold' (double, default 3.0 — ignored; kept for API compat).
/// Output key:  'changepoints' (List<int> of sample indices).
///
/// Top-level so Flutter's compute() can spawn it in an isolate.
Map<String, dynamic> changepointWorker(Map<String, dynamic> args) {
  final data = List<double>.from(args['data'] as List);
  final windowSize = (args['windowSize'] as int?) ?? 32;
  final minSep = windowSize ~/ 2;
  final threshold = (args['threshold'] as double?) ?? 1.0;
  final n = data.length;

  if (n < 4) return {'changepoints': <int>[]};

  // Step 1: compute mean and std of the whole signal
  double sum = 0.0, sum2 = 0.0;
  for (final v in data) {
    sum += v;
    sum2 += v * v;
  }
  final mean = sum / n;
  final variance = (sum2 / n) - mean * mean;
  final sigma = math.sqrt(variance.clamp(1e-12, double.infinity));

  // Step 2: cumulative sum of standardised signal
  // S[0] = 0, S[k] = Σ_{i=0}^{k-1} (x[i] - mean) / sigma
  final S = List<double>.filled(n + 1, 0.0);
  for (int i = 0; i < n; i++) {
    S[i + 1] = S[i] + (data[i] - mean) / sigma;
  }

  // AIC/BIC-like penalty: accept changepoint when improvement in
  // negative log-likelihood exceeds penalty.
  // For a Gaussian model split at t, the log-likelihood gain is:
  //   ΔLL(t) = n/2 * log(σ²_full) - t/2*log(σ²_left) - (n-t)/2*log(σ²_right)
  // Approximated cheaply via the CUSUM magnitude:
  //   cusum(t) = |S[t] - S[0]| / sqrt(t)  or  the Vost statistic.
  //
  // We use the simpler normalised CUSUM score and threshold against
  // log(n)/2 (half the BIC penalty for adding one parameter).
  final penalty = math.log(n) / 2.0;
  final effectivePenalty = penalty * threshold;

  // Score every candidate location using the normalised CUSUM statistic:
  //   Q(t) = max(S[t] - min_{0≤s≤t} S[s],  max_{0≤s≤t} S[s] - S[t])
  // This is the one-sided CUSUM for detecting upward or downward shifts.
  final changepoints = <int>[];

  // Slide a detection window to find local CUSUM maxima
  // Use the full-signal CUSUM and find positions where Q(t) > penalty
  double runMin = 0.0, runMax = 0.0;
  final scores = List<double>.filled(n, 0.0);
  for (int t = 1; t <= n; t++) {
    if (S[t] < runMin) runMin = S[t];
    if (S[t] > runMax) runMax = S[t];
    scores[t - 1] = math.max(S[t] - runMin, runMax - S[t]);
  }

  // Find peaks in scores above penalty, separated by at least minSep
  // Use a greedy peak-picker: walk left-to-right, take the first score
  // above threshold, then skip ahead by minSep.
  int lastCP = -minSep - 1;
  for (int t = 1; t < n - 1; t++) {
    if (scores[t] > effectivePenalty &&
        scores[t] >= scores[t - 1] &&
        scores[t] >= scores[t + 1] &&
        t - lastCP > minSep) {
      // Refine: find the exact point of maximum gradient in ±minSep window
      int best = t;
      double bestScore = scores[t];
      final lo = math.max(1, t - minSep ~/ 2);
      final hi = math.min(n - 1, t + minSep ~/ 2);
      for (int s = lo; s <= hi; s++) {
        if (scores[s] > bestScore) {
          bestScore = scores[s];
          best = s;
        }
      }
      changepoints.add(best);
      lastCP = best;
    }
  }

  return {'changepoints': changepoints};
}
