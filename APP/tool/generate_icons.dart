// Regenerates the web/PWA app icons with the MODA "Lancaster Heritage"
// dual-wave mark (terracotta background, white slow wave, mustard fast
// wave) — mirrors android/app/src/main/res/drawable/ic_launcher_foreground.xml.
//
// Run with: flutter test tool/generate_icons.dart
import 'dart:io';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

const _background = Color(0xFFC1502E); // moda-primary (terracotta)
const _slowWaveColor = Colors.white;
const _fastWaveColor = Color(0xFFF4C95D); // moda-highlight (mustard)

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  test('generate brand icons', () async {
    await _renderIcon('web/icons/Icon-192.png', 192);
    await _renderIcon('web/icons/Icon-512.png', 512);
    await _renderIcon('web/icons/Icon-maskable-192.png', 192);
    await _renderIcon('web/icons/Icon-maskable-512.png', 512);
    await _renderIcon('web/favicon.png', 16);
  });
}

Future<void> _renderIcon(String relativePath, int size) async {
  final recorder = ui.PictureRecorder();
  final canvas = Canvas(
      recorder, Rect.fromLTWH(0, 0, size.toDouble(), size.toDouble()));
  canvas.scale(size / 108.0);

  canvas.drawRect(
      const Rect.fromLTWH(0, 0, 108, 108), Paint()..color = _background);

  final slowWave = Path()
    ..moveTo(16, 66)
    ..cubicTo(36, 50, 72, 82, 92, 66);
  canvas.drawPath(
      slowWave,
      Paint()
        ..color = _slowWaveColor
        ..style = PaintingStyle.stroke
        ..strokeWidth = 9
        ..strokeCap = StrokeCap.round
        ..strokeJoin = StrokeJoin.round);

  final fastWave = Path()
    ..moveTo(16, 42)
    ..cubicTo(21, 34, 31, 34, 36, 42)
    ..cubicTo(41, 50, 51, 50, 56, 42)
    ..cubicTo(61, 34, 71, 34, 76, 42)
    ..cubicTo(81, 50, 91, 50, 96, 42);
  canvas.drawPath(
      fastWave,
      Paint()
        ..color = _fastWaveColor
        ..style = PaintingStyle.stroke
        ..strokeWidth = 6
        ..strokeCap = StrokeCap.round
        ..strokeJoin = StrokeJoin.round);

  final picture = recorder.endRecording();
  final image = await picture.toImage(size, size);
  final bytes = await image.toByteData(format: ui.ImageByteFormat.png);
  final file = File(relativePath);
  await file.create(recursive: true);
  await file.writeAsBytes(bytes!.buffer.asUint8List());
  // ignore: avoid_print
  print('wrote $relativePath (${size}x$size)');
}
