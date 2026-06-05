import 'dart:typed_data';

/// Decode little-endian signed 16-bit PCM bytes into normalized doubles
/// in the range −1.0…1.0. A trailing odd byte (incomplete sample) is ignored.
List<double> pcm16ToDoubles(Uint8List bytes) {
  final n = bytes.length ~/ 2;
  final out = List<double>.filled(n, 0.0);
  final bd = ByteData.view(bytes.buffer, bytes.offsetInBytes, n * 2);
  for (var i = 0; i < n; i++) {
    out[i] = bd.getInt16(i * 2, Endian.little) / 32768.0;
  }
  return out;
}

/// Streaming decimator that averages blocks of input samples down to a lower
/// output rate. Carries a fractional budget across [process] calls so no
/// samples are lost at chunk boundaries. When [outputRate] >= [inputRate] the
/// input is passed through unchanged (no upsampling).
class AudioResampler {
  final double _ratio; // input samples consumed per output sample
  double _budget;
  double _acc = 0.0;
  int _count = 0;

  AudioResampler({required double inputRate, required double outputRate})
      : _ratio = outputRate >= inputRate ? 1.0 : inputRate / outputRate,
        _budget = outputRate >= inputRate ? 1.0 : inputRate / outputRate;

  List<double> process(List<double> input) {
    final out = <double>[];
    for (final s in input) {
      _acc += s;
      _count++;
      _budget -= 1.0;
      if (_budget <= 0.0) {
        out.add(_acc / _count);
        _acc = 0.0;
        _count = 0;
        _budget += _ratio;
      }
    }
    return out;
  }
}
