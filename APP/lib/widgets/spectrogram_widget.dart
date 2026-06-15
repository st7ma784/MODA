import 'dart:math' as math;
import 'package:flutter/material.dart';

class SpectrogramWidget extends StatelessWidget {
  final List<List<double>> history;
  final double sampleRate;
  final double height;

  const SpectrogramWidget({
    super.key,
    required this.history,
    required this.sampleRate,
    this.height = 120,
  });

  @override
  Widget build(BuildContext context) {
    if (history.isEmpty) {
      return SizedBox(
        height: height,
        child: const Center(
          child: Text(
            'Waiting for spectrogram…',
            style: TextStyle(fontSize: 12, color: Colors.white38),
          ),
        ),
      );
    }

    final freqBins = history.first.length;
    final nyquist = sampleRate / 2;

    return SizedBox(
      height: height,
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          _FreqAxis(nyquist: nyquist, height: height),
          const SizedBox(width: 4),
          Expanded(
            child: ClipRect(
              child: CustomPaint(
                painter: _SpectrogramPainter(history: history, freqBins: freqBins),
                child: const SizedBox.expand(),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _FreqAxis extends StatelessWidget {
  final double nyquist;
  final double height;
  const _FreqAxis({required this.nyquist, required this.height});

  String _fmt(double hz) {
    if (hz >= 1000) return '${(hz / 1000).toStringAsFixed(1)}k';
    return hz.toStringAsFixed(0);
  }

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: 28,
      height: height,
      child: Column(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        crossAxisAlignment: CrossAxisAlignment.end,
        children: [
          Text('${_fmt(nyquist)}Hz',
              style: const TextStyle(fontSize: 8, color: Colors.white38)),
          Text(_fmt(nyquist * 0.75),
              style: const TextStyle(fontSize: 8, color: Colors.white38)),
          Text(_fmt(nyquist * 0.5),
              style: const TextStyle(fontSize: 8, color: Colors.white38)),
          Text(_fmt(nyquist * 0.25),
              style: const TextStyle(fontSize: 8, color: Colors.white38)),
          const Text('0', style: TextStyle(fontSize: 8, color: Colors.white38)),
        ],
      ),
    );
  }
}

class _SpectrogramPainter extends CustomPainter {
  final List<List<double>> history;
  final int freqBins;

  const _SpectrogramPainter({required this.history, required this.freqBins});

  // Viridis-inspired heat map: dark blue → teal → yellow → white
  Color _heatColor(double v) {
    v = v.clamp(0.0, 1.0);
    if (v < 0.25) {
      return Color.lerp(const Color(0xFF0D0887), const Color(0xFF5B02A3), v / 0.25)!;
    } else if (v < 0.5) {
      return Color.lerp(const Color(0xFF5B02A3), const Color(0xFF21908C), (v - 0.25) / 0.25)!;
    } else if (v < 0.75) {
      return Color.lerp(const Color(0xFF21908C), const Color(0xFFFDE725), (v - 0.5) / 0.25)!;
    } else {
      return Color.lerp(const Color(0xFFFDE725), Colors.white, (v - 0.75) / 0.25)!;
    }
  }

  @override
  void paint(Canvas canvas, Size size) {
    if (history.isEmpty || size.width == 0 || size.height == 0) return;

    final nT = history.length;
    final nF = freqBins;

    // Global normalisation across the visible history
    double globalMax = 0;
    for (final frame in history) {
      for (final v in frame) {
        if (v > globalMax) globalMax = v;
      }
    }
    if (globalMax == 0) return;

    final cellW = size.width / nT;
    final cellH = size.height / nF;
    final paint = Paint()..style = PaintingStyle.fill;

    for (int t = 0; t < nT; t++) {
      final frame = history[t];
      final bins = math.min(frame.length, nF);
      for (int f = 0; f < bins; f++) {
        paint.color = _heatColor(frame[f] / globalMax);
        // f=0 is DC (bottom), f=nF-1 is Nyquist (top)
        canvas.drawRect(
          Rect.fromLTWH(
            t * cellW,
            size.height - (f + 1) * cellH,
            cellW + 0.5, // slight overlap avoids sub-pixel gaps
            cellH + 0.5,
          ),
          paint,
        );
      }
    }
  }

  @override
  bool shouldRepaint(_SpectrogramPainter old) =>
      !identical(old.history, history);
}
