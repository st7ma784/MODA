import 'dart:typed_data';

/// Packs [values] as a NumPy v1.0 `.npy` file containing a 1-D float32 array.
///
/// The FastMODA server calls `np.load()` on the uploaded file, so raw bytes
/// without the `.npy` header cause a load error.  This produces a byte-for-byte
/// identical format to `numpy.save`.
///
/// Format: magic (6) + version (2) + header_len LE (2) + header + data.
/// Constraint: (10 + header_len) must be a multiple of 64.
List<int> packNpy(List<double> values) {
  final n = values.length;

  // Build header dict, padded with spaces so the total preamble is 64-byte aligned
  final baseStr =
      "{'descr': '<f4', 'fortran_order': False, 'shape': ($n,), }";
  final base = baseStr.codeUnits; // ASCII-safe
  final pad = (64 - (11 + base.length) % 64) % 64; // spaces needed before \n
  final header = [...base, ...List.filled(pad, 0x20), 0x0a]; // … + \n

  // Raw float32 data (little-endian)
  final dataBytes = ByteData(n * 4);
  for (int i = 0; i < n; i++) {
    dataBytes.setFloat32(i * 4, values[i], Endian.little);
  }

  return [
    0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, // \x93NUMPY magic
    0x01, 0x00, // version 1.0
    header.length & 0xff, (header.length >> 8) & 0xff, // header_len (LE uint16)
    ...header,
    ...dataBytes.buffer.asUint8List(),
  ];
}
