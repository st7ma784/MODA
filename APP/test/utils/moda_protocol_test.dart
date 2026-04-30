import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:moda_mobile/utils/moda_protocol.dart';

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Builds a well-formed MODA Signal Data packet (type 0x01).
List<int> makePacket({
  int packetType = 0x01,
  int seqNum = 0,
  int timestamp = 0,
  required int numSamples,
  required int numChannels,
  int flags = 0,
  required List<int> payload,
}) {
  final hdr = ByteData(10);
  hdr.setUint8(0, packetType);
  hdr.setUint8(1, seqNum);
  hdr.setUint32(2, timestamp, Endian.little);
  hdr.setUint16(6, numSamples, Endian.little);
  hdr.setUint8(8, numChannels);
  hdr.setUint8(9, flags);
  return [...hdr.buffer.asUint8List(), ...payload];
}

/// Encodes a list of int16 LE samples as raw bytes.
List<int> int16Bytes(List<int> values) {
  final bd = ByteData(values.length * 2);
  for (int i = 0; i < values.length; i++) {
    bd.setInt16(i * 2, values[i], Endian.little);
  }
  return bd.buffer.asUint8List().toList();
}

/// Encodes a list of float32 LE samples as raw bytes.
List<int> float32Bytes(List<double> values) {
  final bd = ByteData(values.length * 4);
  for (int i = 0; i < values.length; i++) {
    bd.setFloat32(i * 4, values[i], Endian.little);
  }
  return bd.buffer.asUint8List().toList();
}

// ── Tests ─────────────────────────────────────────────────────────────────────

void main() {
  group('parseModaPacket — invalid input', () {
    test('returns null for empty bytes', () {
      expect(parseModaPacket([]), isNull);
    });

    test('returns null for payload shorter than 10 bytes', () {
      expect(parseModaPacket(List.filled(9, 0)), isNull);
    });

    test('returns null for packet type ≠ 0x01', () {
      final bytes = makePacket(
          packetType: 0x02,
          numSamples: 1,
          numChannels: 1,
          payload: int16Bytes([100]));
      expect(parseModaPacket(bytes), isNull);
    });

    test('returns null when numSamples is zero', () {
      final bytes = makePacket(
          numSamples: 0, numChannels: 1, payload: []);
      expect(parseModaPacket(bytes), isNull);
    });

    test('returns null when numChannels is zero', () {
      final bytes = makePacket(
          numSamples: 1, numChannels: 0, payload: int16Bytes([100]));
      expect(parseModaPacket(bytes), isNull);
    });
  });

  group('parseModaPacket — sequence number', () {
    test('returns the correct sequence number', () {
      final bytes = makePacket(
          seqNum: 42,
          numSamples: 1,
          numChannels: 1,
          payload: int16Bytes([0]));
      expect(parseModaPacket(bytes)!.sequenceNum, 42);
    });

    test('sequence number wraps at 255', () {
      final bytes = makePacket(
          seqNum: 255,
          numSamples: 1,
          numChannels: 1,
          payload: int16Bytes([0]));
      expect(parseModaPacket(bytes)!.sequenceNum, 255);
    });
  });

  group('parseModaPacket — int16 LE decoding', () {
    test('max positive int16 (32767) normalises to ≈+1.0', () {
      final bytes = makePacket(
          numSamples: 1,
          numChannels: 1,
          payload: int16Bytes([32767]));
      final pkt = parseModaPacket(bytes, isFloat32: false)!;
      expect(pkt.channel0[0], closeTo(1.0, 1e-4));
    });

    test('min int16 (−32768) normalises to −1.0', () {
      final bytes = makePacket(
          numSamples: 1,
          numChannels: 1,
          payload: int16Bytes([-32768]));
      final pkt = parseModaPacket(bytes, isFloat32: false)!;
      expect(pkt.channel0[0], closeTo(-1.0, 1e-4));
    });

    test('zero encodes to 0.0', () {
      final bytes = makePacket(
          numSamples: 1,
          numChannels: 1,
          payload: int16Bytes([0]));
      expect(parseModaPacket(bytes)!.channel0[0], 0.0);
    });

    test('multiple samples are all decoded', () {
      final bytes = makePacket(
          numSamples: 4,
          numChannels: 1,
          payload: int16Bytes([1000, 2000, -1000, 0]));
      final ch = parseModaPacket(bytes)!.channel0;
      expect(ch.length, 4);
      expect(ch[0], closeTo(1000 / 32768.0, 1e-6));
      expect(ch[2], closeTo(-1000 / 32768.0, 1e-6));
    });
  });

  group('parseModaPacket — float32 LE decoding', () {
    test('decodes a float32 value correctly', () {
      final bytes = makePacket(
          numSamples: 1,
          numChannels: 1,
          payload: float32Bytes([0.75]));
      final pkt = parseModaPacket(bytes, isFloat32: true)!;
      expect(pkt.channel0[0], closeTo(0.75, 1e-6));
    });

    test('decodes negative float32 value', () {
      final bytes = makePacket(
          numSamples: 1,
          numChannels: 1,
          payload: float32Bytes([-0.5]));
      expect(parseModaPacket(bytes, isFloat32: true)!.channel0[0],
          closeTo(-0.5, 1e-6));
    });
  });

  group('parseModaPacket — multi-channel deinterleaving', () {
    // Layout: [ch0_s0, ch1_s0, ch0_s1, ch1_s1]
    test('extracts only channel 0 from 2-channel stream', () {
      final payload =
          int16Bytes([100, 200, 300, 400]); // ch0: 100, 300 | ch1: 200, 400
      final bytes = makePacket(
          numSamples: 2, numChannels: 2, payload: payload);
      final pkt = parseModaPacket(bytes)!;
      expect(pkt.channel0.length, 2);
      expect(pkt.channel0[0], closeTo(100 / 32768.0, 1e-6));
      expect(pkt.channel0[1], closeTo(300 / 32768.0, 1e-6));
    });

    test('reports correct numChannels in result', () {
      final bytes = makePacket(
          numSamples: 1,
          numChannels: 3,
          payload: int16Bytes([1, 2, 3]));
      expect(parseModaPacket(bytes)!.numChannels, 3);
    });
  });

  group('parseModaPacket — flags', () {
    test('lossDetected is true when flags bit 0 is set', () {
      final bytes = makePacket(
          numSamples: 1, numChannels: 1, flags: 0x01,
          payload: int16Bytes([0]));
      expect(parseModaPacket(bytes)!.lossDetected, isTrue);
    });

    test('overflow is true when flags bit 1 is set', () {
      final bytes = makePacket(
          numSamples: 1, numChannels: 1, flags: 0x02,
          payload: int16Bytes([0]));
      expect(parseModaPacket(bytes)!.overflow, isTrue);
    });

    test('both flags false when flags byte is 0x00', () {
      final bytes = makePacket(
          numSamples: 1, numChannels: 1, flags: 0x00,
          payload: int16Bytes([0]));
      final pkt = parseModaPacket(bytes)!;
      expect(pkt.lossDetected, isFalse);
      expect(pkt.overflow, isFalse);
    });
  });

  group('parseModaPacket — truncated data', () {
    test('does not throw when payload is shorter than header claims', () {
      // Header says 10 samples but payload has only 2 bytes (= 1 sample)
      final bytes = makePacket(
          numSamples: 10, numChannels: 1, payload: int16Bytes([999]));
      expect(() => parseModaPacket(bytes), returnsNormally);
    });

    test('returns only the samples that fit', () {
      final bytes = makePacket(
          numSamples: 10, numChannels: 1, payload: int16Bytes([1, 2]));
      final pkt = parseModaPacket(bytes)!;
      expect(pkt.channel0.length, lessThanOrEqualTo(10));
    });
  });
}
