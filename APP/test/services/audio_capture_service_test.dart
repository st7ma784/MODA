import 'package:flutter_test/flutter_test.dart';
import 'package:moda_mobile/services/audio_capture_service.dart';

void main() {
  group('AudioCaptureService', () {
    test('is not capturing on construction', () {
      expect(AudioCaptureService().isCapturing, isFalse);
    });

    test('targetSampleRate defaults to 256 and is updatable', () {
      final svc = AudioCaptureService();
      expect(svc.targetSampleRate, 256.0);
      svc.targetSampleRate = 100.0;
      expect(svc.targetSampleRate, 100.0);
    });

    test('exposes a broadcast sampleStream', () {
      final svc = AudioCaptureService();
      expect(svc.sampleStream.isBroadcast, isTrue);
    });

    test('dispose does not throw', () {
      expect(() => AudioCaptureService().dispose(), returnsNormally);
    });
  });
}
