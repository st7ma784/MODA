import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:moda_mobile/utils/audio_resampler.dart';

void main() {
  group('pcm16ToDoubles', () {
    test('decodes little-endian signed PCM16 into normalized doubles', () {
      // 0x0000 = 0, 0x7FFF = +32767, 0x8000 = -32768
      final bytes = Uint8List.fromList([0x00, 0x00, 0xFF, 0x7F, 0x00, 0x80]);
      final out = pcm16ToDoubles(bytes);
      expect(out.length, 3);
      expect(out[0], 0.0);
      expect(out[1], closeTo(32767 / 32768, 1e-9));
      expect(out[2], -1.0);
    });

    test('ignores a trailing odd byte', () {
      final bytes = Uint8List.fromList([0x00, 0x00, 0x11]);
      expect(pcm16ToDoubles(bytes).length, 1);
    });
  });

  group('AudioResampler', () {
    test('16 kHz -> 256 Hz yields 256 averaged samples for 16000 inputs', () {
      final r = AudioResampler(inputRate: 16000, outputRate: 256);
      final out = r.process(List<double>.filled(16000, 1.0));
      expect(out.length, 256);
      expect(out.every((v) => (v - 1.0).abs() < 1e-9), isTrue);
    });

    test('preserves output count across chunk boundaries', () {
      final r = AudioResampler(inputRate: 16000, outputRate: 256);
      final a = r.process(List<double>.filled(8000, 1.0));
      final b = r.process(List<double>.filled(8000, 1.0));
      expect(a.length + b.length, 256);
    });

    test('passes samples through when output rate >= input rate', () {
      final r = AudioResampler(inputRate: 16000, outputRate: 16000);
      final out = r.process([1.0, 2.0, 3.0]);
      expect(out, [1.0, 2.0, 3.0]);
    });
  });
}
