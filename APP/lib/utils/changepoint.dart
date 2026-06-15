import 'dart:math' as math;

/// Changepoint detector with three modes, selectable via args['mode']:
///
///   'raw'       — CUSUM on the standardised signal (detects mean shifts).
///   'envelope'  — CUSUM on short-time RMS (detects changes in signal amplitude
///                 / rhythm strength without reacting to the oscillation itself).
///   'frequency' — CUSUM on per-window dominant frequency (detects when the
///                 rhythm speeds up or slows down).
///
/// Input keys:  'data' (List<double>), 'windowSize' (int), 'threshold' (double),
///              'mode' (String, default 'raw'), 'fs' (double, needed for 'frequency').
/// Output key:  'changepoints' (List<int> of sample indices).
Map<String, dynamic> changepointWorker(Map<String, dynamic> args) {
  final mode = args['mode'] as String? ?? 'raw';
  return switch (mode) {
    'envelope'  => _envelopeMode(args),
    'frequency' => _frequencyMode(args),
    _           => _rawMode(args),
  };
}

// ── Raw mode — CUSUM on standardised signal (original algorithm) ─────────────

Map<String, dynamic> _rawMode(Map<String, dynamic> args) {
  final data = List<double>.from(args['data'] as List);
  final windowSize = (args['windowSize'] as int?) ?? 32;
  final minSep = windowSize ~/ 2;
  final threshold = (args['threshold'] as double?) ?? 1.0;
  final n = data.length;

  if (n < 4) return {'changepoints': <int>[]};

  double sum = 0.0, sum2 = 0.0;
  for (final v in data) {
    sum += v;
    sum2 += v * v;
  }
  final mean = sum / n;
  final variance = (sum2 / n) - mean * mean;
  final sigma = math.sqrt(variance.clamp(1e-12, double.infinity));

  final S = List<double>.filled(n + 1, 0.0);
  for (int i = 0; i < n; i++) {
    S[i + 1] = S[i] + (data[i] - mean) / sigma;
  }

  final penalty = math.log(n) / 2.0;
  final effectivePenalty = penalty * threshold;

  double runMin = 0.0, runMax = 0.0;
  final scores = List<double>.filled(n, 0.0);
  for (int t = 1; t <= n; t++) {
    if (S[t] < runMin) runMin = S[t];
    if (S[t] > runMax) runMax = S[t];
    scores[t - 1] = math.max(S[t] - runMin, runMax - S[t]);
  }

  final changepoints = <int>[];
  int lastCP = -minSep - 1;
  for (int t = 1; t < n - 1; t++) {
    if (scores[t] > effectivePenalty &&
        scores[t] >= scores[t - 1] &&
        scores[t] >= scores[t + 1] &&
        t - lastCP > minSep) {
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

// ── Envelope mode — CUSUM on short-time RMS ──────────────────────────────────
//
// Computes RMS per hop window, then runs CUSUM on that series.
// Fires when the signal's amplitude changes, not when its waveform oscillates —
// ideal for rhythmic signals where you want to ignore the rhythm and catch
// changes in how strong or weak it is.

Map<String, dynamic> _envelopeMode(Map<String, dynamic> args) {
  final data   = List<double>.from(args['data'] as List);
  final winSz  = (args['windowSize'] as int?)    ?? 32;
  final thresh = (args['threshold']  as double?) ?? 1.0;
  final hop    = math.max(1, winSz ~/ 2);
  final n      = data.length;

  if (n < winSz) return {'changepoints': <int>[]};

  final rms = <double>[];
  for (int s = 0; s + winSz <= n; s += hop) {
    double ss = 0;
    for (int i = s; i < s + winSz; i++) {
      ss += data[i] * data[i];
    }
    rms.add(math.sqrt(ss / winSz));
  }
  if (rms.length < 4) return {'changepoints': <int>[]};

  final peaks = _cusumPeaks(rms, thresh);
  return {
    'changepoints': peaks
        .map((fi) => (fi * hop + winSz ~/ 2).clamp(0, n - 1))
        .toList(),
  };
}

// ── Frequency mode — CUSUM on per-window dominant frequency ──────────────────
//
// Computes the dominant frequency via DFT for each hop window, then runs CUSUM
// on that frequency time series.  Fires when the rhythm speeds up or slows
// down — not when its amplitude changes.

Map<String, dynamic> _frequencyMode(Map<String, dynamic> args) {
  final data   = List<double>.from(args['data'] as List);
  final winSz  = (args['windowSize'] as int?)    ?? 32;
  final thresh = (args['threshold']  as double?) ?? 1.0;
  final fs     = (args['fs']         as num?)?.toDouble() ?? 256.0;
  final hop    = math.max(1, winSz ~/ 2);
  final n      = data.length;

  if (n < winSz) return {'changepoints': <int>[]};

  final freqs = <double>[];
  for (int s = 0; s + winSz <= n; s += hop) {
    freqs.add(_windowDominantFreq(data, s, winSz, fs));
  }
  if (freqs.length < 4) return {'changepoints': <int>[]};

  final peaks = _cusumPeaks(freqs, thresh);
  return {
    'changepoints': peaks
        .map((fi) => (fi * hop + winSz ~/ 2).clamp(0, n - 1))
        .toList(),
  };
}

// ── Shared helpers ────────────────────────────────────────────────────────────

/// Standardise-then-CUSUM peak-picker on an arbitrary series.
/// Returns indices into [series] where changepoints occur.
List<int> _cusumPeaks(List<double> series, double threshold) {
  final n = series.length;
  double sum = 0, sum2 = 0;
  for (final v in series) {
    sum += v;
    sum2 += v * v;
  }
  final mean  = sum / n;
  final sigma = math.sqrt((sum2 / n - mean * mean).clamp(1e-12, double.infinity));
  final minSep = math.max(1, n ~/ 8);

  final S = List<double>.filled(n + 1, 0.0);
  for (int i = 0; i < n; i++) {
    S[i + 1] = S[i] + (series[i] - mean) / sigma;
  }

  final penalty = math.log(n) / 2.0 * threshold;
  double runMin = 0, runMax = 0;
  final scores = List<double>.filled(n, 0.0);
  for (int t = 1; t <= n; t++) {
    if (S[t] < runMin) runMin = S[t];
    if (S[t] > runMax) runMax = S[t];
    scores[t - 1] = math.max(S[t] - runMin, runMax - S[t]);
  }

  final cps    = <int>[];
  int lastCP   = -minSep - 1;
  for (int t = 1; t < n - 1; t++) {
    if (scores[t] > penalty &&
        scores[t] >= scores[t - 1] &&
        scores[t] >= scores[t + 1] &&
        t - lastCP > minSep) {
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
      cps.add(best);
      lastCP = best;
    }
  }
  return cps;
}

/// Dominant frequency (Hz) for a sub-slice of [data] starting at [start]
/// for [winSz] samples, using a naive DFT (O(N²) — fine for small windows).
double _windowDominantFreq(List<double> data, int start, int winSz, double fs) {
  final hN = winSz ~/ 2;
  double maxMagSq = -1;
  int maxK = 1; // skip DC
  for (int k = 1; k < hN; k++) {
    double re = 0, im = 0;
    final factor = 2 * math.pi * k / winSz;
    for (int i = 0; i < winSz; i++) {
      re += data[start + i] * math.cos(factor * i);
      im -= data[start + i] * math.sin(factor * i);
    }
    final magSq = re * re + im * im;
    if (magSq > maxMagSq) {
      maxMagSq = magSq;
      maxK = k;
    }
  }
  return maxK * fs / winSz;
}
