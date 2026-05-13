import 'dart:convert';
import 'dart:io';
import 'package:path_provider/path_provider.dart';
import 'package:share_plus/share_plus.dart';

/// Writes [samples] as a timestamped CSV and opens the OS share sheet.
Future<void> exportSignalCsv(List<double> samples, double sampleRate) async {
  final dir = await getApplicationDocumentsDirectory();
  final ts = DateTime.now().millisecondsSinceEpoch;
  final file = File('${dir.path}/moda_signal_$ts.csv');

  final buf = StringBuffer('index,time_s,amplitude\n');
  for (int i = 0; i < samples.length; i++) {
    buf.writeln('$i,${(i / sampleRate).toStringAsFixed(4)},${samples[i]}');
  }
  await file.writeAsString(buf.toString());
  await Share.shareXFiles([XFile(file.path)], subject: 'MODA Signal Data');
}

/// Writes [result] as pretty-printed JSON, stripping large Plotly blobs first.
Future<void> exportResultJson(Map<String, dynamic> result) async {
  final dir = await getApplicationDocumentsDirectory();
  final ts = DateTime.now().millisecondsSinceEpoch;
  final file = File('${dir.path}/moda_result_$ts.json');

  // Drop any value whose string representation is a large blob (Plotly JSON).
  final stripped = Map<String, dynamic>.fromEntries(
    result.entries.where((e) {
      final v = e.value;
      if (v is String && v.length > 500) return false;
      return true;
    }),
  );
  await file.writeAsString(const JsonEncoder.withIndent('  ').convert(stripped));
  await Share.shareXFiles([XFile(file.path)], subject: 'MODA Analysis Result');
}
