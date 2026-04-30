import 'dart:math' as math;
import 'package:flutter_test/flutter_test.dart';
import 'package:moda_mobile/utils/dft.dart';

void main() {
  group('dftWorker — band identification', () {
    List<double> sine(double freqHz, double fsHz, int n) =>
        List.generate(n, (i) => math.sin(2 * math.pi * freqHz * i / fsHz));

    test('10 Hz sine is classified as alpha (8–12 Hz)', () {
      final result = dftWorker({'data': sine(10, 256, 256), 'fs': 256.0});
      final alpha = result['alpha'] as double;
      final theta = result['theta'] as double;
      final beta = result['beta'] as double;
      expect(alpha, greaterThan(theta));
      expect(alpha, greaterThan(beta));
    });

    test('6 Hz sine is classified as theta (4–8 Hz)', () {
      final result = dftWorker({'data': sine(6, 256, 256), 'fs': 256.0});
      final theta = result['theta'] as double;
      final alpha = result['alpha'] as double;
      expect(theta, greaterThan(alpha));
    });

    test('2 Hz sine is classified as delta (0.5–4 Hz)', () {
      final result = dftWorker({'data': sine(2, 256, 256), 'fs': 256.0});
      final delta = result['delta'] as double;
      final theta = result['theta'] as double;
      expect(delta, greaterThan(theta));
    });

    test('20 Hz sine is classified as beta (12–30 Hz)', () {
      final result = dftWorker({'data': sine(20, 256, 256), 'fs': 256.0});
      final beta = result['beta'] as double;
      final alpha = result['alpha'] as double;
      expect(beta, greaterThan(alpha));
    });
  });

  group('dftWorker — dominant frequency', () {
    test('reports dominant frequency close to input sine frequency', () {
      const freq = 10.0;
      final data = List.generate(
          256, (i) => math.sin(2 * math.pi * freq * i / 256.0));
      final result = dftWorker({'data': data, 'fs': 256.0});
      expect(result['dominant'] as double, closeTo(freq, 2.0));
    });

    test('dominant frequency respects non-default sample rate', () {
      const freq = 5.0;
      const fs = 100.0;
      final data = List.generate(
          200, (i) => math.sin(2 * math.pi * freq * i / fs));
      final result = dftWorker({'data': data, 'fs': fs});
      expect(result['dominant'] as double, closeTo(freq, 2.0));
    });
  });

  group('dftWorker — output shape & invariants', () {
    test('returns N/2 spectrum bins for 256-sample input', () {
      final result =
          dftWorker({'data': List<double>.filled(256, 0.0), 'fs': 256.0});
      expect((result['mags'] as List).length, 128);
    });

    test('returns fewer bins for shorter input', () {
      final result =
          dftWorker({'data': List<double>.filled(100, 0.0), 'fs': 100.0});
      expect((result['mags'] as List).length, 50);
    });

    test('all band powers are zero for a zero signal', () {
      final result =
          dftWorker({'data': List<double>.filled(256, 0.0), 'fs': 256.0});
      expect(result['delta'] as double, 0.0);
      expect(result['theta'] as double, 0.0);
      expect(result['alpha'] as double, 0.0);
      expect(result['beta'] as double, 0.0);
      expect(result['gamma'] as double, 0.0);
    });

    test('all band powers are non-negative', () {
      final data = List.generate(
          256, (i) => math.sin(2 * math.pi * 10 * i / 256.0));
      final result = dftWorker({'data': data, 'fs': 256.0});
      for (final key in ['delta', 'theta', 'alpha', 'beta', 'gamma']) {
        expect(result[key] as double, greaterThanOrEqualTo(0.0),
            reason: '$key should be ≥ 0');
      }
    });

    test('quality is in range 0–100', () {
      final data = List.generate(
          256, (i) => math.sin(2 * math.pi * 10 * i / 256.0));
      final result = dftWorker({'data': data, 'fs': 256.0});
      final q = result['quality'] as double;
      expect(q, greaterThanOrEqualTo(0.0));
      expect(q, lessThanOrEqualTo(100.0));
    });

    test('handles minimum-length input (2 samples) without throwing', () {
      expect(
          () => dftWorker({'data': [0.0, 1.0], 'fs': 2.0}), returnsNormally);
    });
  });
}
