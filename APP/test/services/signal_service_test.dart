import 'package:flutter_test/flutter_test.dart';
import 'package:moda_mobile/services/signal_service.dart';

void main() {
  group('SignalService — ring buffer', () {
    test('recentSamples returns exactly what was added (below capacity)', () {
      final svc = SignalService();
      svc.addSamples([1.0, 2.0, 3.0]);
      expect(svc.recentSamples, [1.0, 2.0, 3.0]);
    });

    test('buffer length is capped at _bufferSize (512)', () {
      final svc = SignalService();
      svc.addSamples(List.generate(600, (i) => i.toDouble()));
      expect(svc.recentSamples.length, 512);
    });

    test('oldest samples are overwritten when buffer wraps', () {
      final svc = SignalService();
      // Add 512 then 1 more — the first sample should be gone
      svc.addSamples(List.generate(512, (i) => i.toDouble()));
      svc.addSamples([999.0]);
      final recent = svc.recentSamples;
      expect(recent.contains(0.0), isFalse); // sample 0 evicted
      expect(recent.last, 999.0);
    });

    test('preserves insertion order after wrap', () {
      final svc = SignalService();
      svc.addSamples(List.generate(520, (i) => i.toDouble()));
      final recent = svc.recentSamples;
      // Values should be monotonically increasing (8..519)
      for (int i = 1; i < recent.length; i++) {
        expect(recent[i], recent[i - 1] + 1.0,
            reason: 'index $i out of order');
      }
    });
  });

  group('SignalService — hasData', () {
    test('is false with 63 samples', () {
      final svc = SignalService();
      svc.addSamples(List.filled(63, 0.0));
      expect(svc.hasData, isFalse);
    });

    test('is true with exactly 64 samples', () {
      final svc = SignalService();
      svc.addSamples(List.filled(64, 0.0));
      expect(svc.hasData, isTrue);
    });

    test('is true with more than 64 samples', () {
      final svc = SignalService();
      svc.addSamples(List.filled(128, 0.0));
      expect(svc.hasData, isTrue);
    });
  });

  group('SignalService — sampleRate', () {
    test('default is 256 Hz', () {
      expect(SignalService().sampleRate, 256.0);
    });

    test('setter updates value', () {
      final svc = SignalService();
      svc.sampleRate = 512.0;
      expect(svc.sampleRate, 512.0);
    });

    test('setter notifies listeners', () {
      final svc = SignalService();
      var notified = false;
      svc.addListener(() => notified = true);
      svc.sampleRate = 100.0;
      expect(notified, isTrue);
    });
  });

  group('SignalService — addSamples notifies listeners', () {
    test('notifyListeners is called after addSamples', () {
      final svc = SignalService();
      var count = 0;
      svc.addListener(() => count++);
      svc.addSamples([1.0, 2.0]);
      expect(count, greaterThan(0));
    });
  });

  group('SignalService — bandPowers initial state', () {
    test('all five bands present with value 0 initially', () {
      final svc = SignalService();
      final powers = svc.bandPowers;
      expect(powers.keys.toSet(),
          {'delta', 'theta', 'alpha', 'beta', 'gamma'});
      for (final v in powers.values) {
        expect(v, 0.0);
      }
    });
  });

  group('SignalService — spectrum initial state', () {
    test('spectrum is non-empty after construction', () {
      expect(SignalService().spectrum.length, greaterThan(0));
    });

    test('spectrum values are zero before any data added', () {
      final svc = SignalService();
      expect(svc.spectrum.every((v) => v == 0.0), isTrue);
    });
  });

  group('SignalService — submitAnalysis guards', () {
    test('does nothing when no client is bound', () async {
      final svc = SignalService();
      svc.addSamples(List.filled(64, 0.0));
      // Should complete without throwing even though no client
      await expectLater(svc.submitAnalysis(), completes);
      expect(svc.isSubmitting, isFalse);
    });

    test('does nothing when hasData is false', () async {
      final svc = SignalService();
      svc.addSamples(List.filled(10, 0.0)); // not enough
      await expectLater(svc.submitAnalysis(), completes);
      expect(svc.isSubmitting, isFalse);
    });
  });
}
