import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:moda_mobile/utils/npy.dart';

void main() {
  group('packNpy — magic & version', () {
    test('first 6 bytes are the NumPy magic string', () {
      final bytes = packNpy([1.0]);
      expect(bytes.sublist(0, 6),
          equals([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59])); // \x93NUMPY
    });

    test('version is 1.0 (bytes 6–7)', () {
      final bytes = packNpy([1.0]);
      expect(bytes[6], 0x01);
      expect(bytes[7], 0x00);
    });
  });

  group('packNpy — alignment', () {
    // NumPy spec: (10 + header_len) must be a multiple of 64.
    for (final n in [1, 10, 99, 100, 255, 256, 511, 512]) {
      test('preamble is 64-byte aligned for n=$n', () {
        final bytes = packNpy(List.filled(n, 0.0));
        final headerLen = bytes[8] | (bytes[9] << 8);
        expect((10 + headerLen) % 64, 0,
            reason: 'n=$n gave headerLen=$headerLen');
      });
    }
  });

  group('packNpy — header content', () {
    test('header contains correct shape', () {
      final bytes = packNpy([1.0, 2.0, 3.0]);
      final headerLen = bytes[8] | (bytes[9] << 8);
      final header =
          String.fromCharCodes(bytes.sublist(10, 10 + headerLen));
      expect(header, contains("'shape': (3,)"));
    });

    test('header contains float32 descriptor', () {
      final bytes = packNpy([0.0]);
      final headerLen = bytes[8] | (bytes[9] << 8);
      final header =
          String.fromCharCodes(bytes.sublist(10, 10 + headerLen));
      expect(header, contains("'descr': '<f4'"));
    });

    test('header declares C-contiguous (not Fortran) order', () {
      final bytes = packNpy([0.0]);
      final headerLen = bytes[8] | (bytes[9] << 8);
      final header =
          String.fromCharCodes(bytes.sublist(10, 10 + headerLen));
      expect(header, contains("'fortran_order': False"));
    });

    test('header ends with newline', () {
      final bytes = packNpy([0.0]);
      final headerLen = bytes[8] | (bytes[9] << 8);
      expect(bytes[10 + headerLen - 1], 0x0a); // \n
    });
  });

  group('packNpy — data encoding', () {
    test('total length equals preamble + n*4 data bytes', () {
      const n = 100;
      final bytes = packNpy(List.filled(n, 0.0));
      final headerLen = bytes[8] | (bytes[9] << 8);
      expect(bytes.length, 10 + headerLen + n * 4);
    });

    test('encodes values as little-endian float32', () {
      final values = [1.0, -1.0, 0.5, 0.0, 3.14];
      final bytes = packNpy(values);
      final headerLen = bytes[8] | (bytes[9] << 8);
      final dataStart = 10 + headerLen;
      final bd = ByteData.view(
          Uint8List.fromList(bytes.sublist(dataStart)).buffer);
      for (int i = 0; i < values.length; i++) {
        expect(bd.getFloat32(i * 4, Endian.little),
            closeTo(values[i], 1e-6),
            reason: 'value at index $i');
      }
    });

    test('handles a single-element array', () {
      final bytes = packNpy([42.0]);
      final headerLen = bytes[8] | (bytes[9] << 8);
      final bd = ByteData.view(
          Uint8List.fromList(bytes.sublist(10 + headerLen)).buffer);
      expect(bd.getFloat32(0, Endian.little), closeTo(42.0, 1e-6));
    });

    test('handles a 512-element array (full buffer)', () {
      final values = List.generate(512, (i) => i.toDouble());
      final bytes = packNpy(values);
      final headerLen = bytes[8] | (bytes[9] << 8);
      final bd = ByteData.view(
          Uint8List.fromList(bytes.sublist(10 + headerLen)).buffer);
      expect(bd.getFloat32(511 * 4, Endian.little), closeTo(511.0, 1e-6));
    });
  });
}
