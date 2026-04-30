import 'dart:typed_data';

/// Parsed Signal Data packet (type 0x01) from the MODA-BLE-SP protocol.
class ModaPacket {
  final int sequenceNum;
  final int numSamples;
  final int numChannels;
  final bool lossDetected; // flags bit 0
  final bool overflow; // flags bit 1

  /// Samples from channel 0 only, normalised to approximately −1..1.
  final List<double> channel0;

  const ModaPacket({
    required this.sequenceNum,
    required this.numSamples,
    required this.numChannels,
    required this.channel0,
    this.lossDetected = false,
    this.overflow = false,
  });
}

/// Parses a raw BLE notification payload into a [ModaPacket].
///
/// Returns `null` for:
/// - Payloads shorter than 10 bytes (incomplete header).
/// - Non-signal-data packet types (byte 0 ≠ 0x01).
/// - Packets with zero samples or zero channels.
///
/// Truncated data (header claims N samples but payload ends early) is handled
/// gracefully — only the samples that fit are returned.
ModaPacket? parseModaPacket(List<int> bytes, {bool isFloat32 = false}) {
  if (bytes.length < 10) return null;
  if (bytes[0] != 0x01) return null;

  final bd = ByteData.view(Uint8List.fromList(bytes).buffer);
  final seqNum = bytes[1];
  final numSamples = bd.getUint16(6, Endian.little);
  final numChannels = bytes[8];
  final flags = bytes[9];

  if (numChannels == 0 || numSamples == 0) return null;

  final bytesPerSample = isFloat32 ? 4 : 2;
  int offset = 10; // header is always 10 bytes

  // Data is interleaved: [ch0_s0, ch1_s0, ..., ch0_s1, ch1_s1, ...]
  // Only channel 0 is extracted for single-channel analysis.
  final ch0 = <double>[];
  for (int s = 0; s < numSamples; s++) {
    for (int ch = 0; ch < numChannels; ch++) {
      if (offset + bytesPerSample > bytes.length) break;
      if (ch == 0) {
        ch0.add(isFloat32
            ? bd.getFloat32(offset, Endian.little)
            : bd.getInt16(offset, Endian.little) / 32768.0);
      }
      offset += bytesPerSample;
    }
  }

  return ModaPacket(
    sequenceNum: seqNum,
    numSamples: numSamples,
    numChannels: numChannels,
    lossDetected: (flags & 0x01) != 0,
    overflow: (flags & 0x02) != 0,
    channel0: ch0,
  );
}
