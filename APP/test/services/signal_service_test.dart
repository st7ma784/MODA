import 'package:flutter_test/flutter_test.dart';
import 'package:moda_mobile/services/signal_service.dart';

void main() {
  group('SignalService — ring buffer', () {
    test('recentSamples returns exactly what was added (below capacity)', () {
      final svc = SignalService();
      svc.addSamples([1.0, 2.0, 3.0]);
      expect(svc.recentSamples, [1.0, 2.0, 3.0]);
    });

    test('buffer length is capped at 512 (bufferSize)', () {
      final svc = SignalService();
      svc.addSamples(List.generate(600, (i) => i.toDouble()));
      expect(svc.recentSamples.length, 512);
    });

    test('oldest samples are overwritten when buffer wraps', () {
      final svc = SignalService();
      svc.addSamples(List.generate(512, (i) => i.toDouble()));
      svc.addSamples([999.0]);
      final recent = svc.recentSamples;
      expect(recent.contains(0.0), isFalse); // sample 0 evicted by wrap
      expect(recent.last, 999.0);
    });

    test('insertion order is preserved after wrap', () {
      final svc = SignalService();
      // Add 520 values: 0..519. After wrap pos=8, result is [8..519].
      svc.addSamples(List.generate(520, (i) => i.toDouble()));
      final recent = svc.recentSamples;
      expect(recent.length, 512);
      for (int i = 1; i < recent.length; i++) {
        expect(recent[i], recent[i - 1] + 1.0,
            reason: 'order broken at index $i');
      }
    });

    test('single-element add works', () {
      final svc = SignalService();
      svc.addSamples([42.0]);
      expect(svc.recentSamples, [42.0]);
    });
  });

  group('SignalService — hasData', () {
    test('is false with 0 samples', () {
      expect(SignalService().hasData, isFalse);
    });

    test('is false with 63 samples', () {
      final svc = SignalService();
      svc.addSamples(List.filled(63, 0.0));
      expect(svc.hasData, isFalse);
    });

    test('becomes true at exactly 64 samples', () {
      final svc = SignalService();
      svc.addSamples(List.filled(64, 0.0));
      expect(svc.hasData, isTrue);
    });

    test('remains true with more than 64 samples', () {
      final svc = SignalService();
      svc.addSamples(List.filled(512, 0.0));
      expect(svc.hasData, isTrue);
    });
  });

  group('SignalService — sampleRate', () {
    test('default is 256.0 Hz', () {
      expect(SignalService().sampleRate, 256.0);
    });

    test('setter updates the value', () {
      final svc = SignalService();
      svc.sampleRate = 512.0;
      expect(svc.sampleRate, 512.0);
    });

    test('setter calls notifyListeners', () {
      final svc = SignalService();
      var fired = false;
      svc.addListener(() => fired = true);
      svc.sampleRate = 100.0;
      expect(fired, isTrue);
    });
  });

  group('SignalService — addSamples', () {
    test('calls notifyListeners synchronously', () {
      final svc = SignalService();
      var count = 0;
      svc.addListener(() => count++);
      svc.addSamples([1.0, 2.0]);
      expect(count, greaterThan(0));
    });

    test('accumulates across multiple calls', () {
      final svc = SignalService();
      svc.addSamples([1.0, 2.0]);
      svc.addSamples([3.0, 4.0]);
      expect(svc.recentSamples, [1.0, 2.0, 3.0, 4.0]);
    });
  });

  group('SignalService — bandPowers', () {
    test('has exactly five bands', () {
      expect(SignalService().bandPowers.keys.toSet(),
          {'delta', 'theta', 'alpha', 'beta', 'gamma'});
    });

    test('all values are 0.0 before any data is added', () {
      for (final v in SignalService().bandPowers.values) {
        expect(v, 0.0);
      }
    });
  });

  group('SignalService — spectrum', () {
    test('has 128 bins after construction (dftSize/2)', () {
      expect(SignalService().spectrum.length, 128);
    });

    test('all values are 0.0 before data arrives', () {
      expect(SignalService().spectrum.every((v) => v == 0.0), isTrue);
    });
  });

  group('SignalService — submitAnalysis guards', () {
    test('is a no-op when no client is bound (no throw)', () async {
      final svc = SignalService();
      svc.addSamples(List.filled(64, 1.0));
      await expectLater(svc.submitAnalysis(), completes);
      expect(svc.isSubmitting, isFalse);
    });

    test('is a no-op when hasData is false', () async {
      final svc = SignalService();
      svc.addSamples([1.0, 2.0]); // only 2 samples
      await expectLater(svc.submitAnalysis(), completes);
      expect(svc.isSubmitting, isFalse);
    });
  });

  group('SignalService — dispose', () {
    test('dispose does not throw', () {
      final svc = SignalService();
      svc.addSamples([1.0]);
      expect(() => svc.dispose(), returnsNormally);
    });

    test('errors stream closes cleanly on dispose', () async {
      final svc = SignalService();
      final errors = <String>[];
      final sub = svc.errors.listen(errors.add);
      svc.dispose();
      await sub.cancel();
      // No assertion needed — absence of throw is the test
    });
  });
}
