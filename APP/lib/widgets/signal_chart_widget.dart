import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';

enum ChartType { timeDomain, spectrum }

class SignalChartWidget extends StatelessWidget {
  final double height;
  final ChartType type;

  /// Live data. When null a placeholder sinusoid is shown.
  final List<double>? data;

  const SignalChartWidget({
    super.key,
    required this.height,
    this.type = ChartType.timeDomain,
    this.data,
  });

  static const int _maxDisplayPts = 150;

  List<FlSpot> _buildSpots() {
    if (data != null && data!.isNotEmpty) {
      final raw = data!;
      if (raw.length <= _maxDisplayPts) {
        return [for (int i = 0; i < raw.length; i++) FlSpot(i.toDouble(), raw[i])];
      }
      // Uniform downsample — keeps rendering fast at high BLE data rates.
      final step = raw.length / _maxDisplayPts;
      return [
        for (int i = 0; i < _maxDisplayPts; i++)
          FlSpot(i.toDouble(), raw[(i * step).floor()])
      ];
    }
    // Placeholder sinusoid while no signal is available
    return List.generate(80, (i) {
      final x = i.toDouble();
      final y = type == ChartType.timeDomain
          ? math.sin(i * 0.3) * 0.6 + math.sin(i * 0.7) * 0.3
          : math.exp(-i * 0.04) * (1 - (i % 8) * 0.05).clamp(0, 1);
      return FlSpot(x, y);
    });
  }

  @override
  Widget build(BuildContext context) {
    final color = Theme.of(context).colorScheme.primary;
    final spots = _buildSpots();
    final isLive = data != null && data!.isNotEmpty;

    // Compute Y range from live data with 10 % padding; fall back to ±1.
    double minY = -1.0, maxY = 1.0;
    if (isLive && spots.isNotEmpty) {
      final ys = spots.map((s) => s.y);
      final lo = ys.reduce(math.min);
      final hi = ys.reduce(math.max);
      final range = (hi - lo).abs();
      final pad = range * 0.10 + 0.01;
      minY = lo - pad;
      maxY = hi + pad;
    }

    return SizedBox(
      height: height,
      child: LineChart(
        LineChartData(
          minY: minY,
          maxY: maxY,
          gridData: FlGridData(
            show: true,
            drawVerticalLine: false,
            getDrawingHorizontalLine: (_) =>
                const FlLine(color: Colors.white10, strokeWidth: 1),
          ),
          titlesData: const FlTitlesData(show: false),
          borderData: FlBorderData(show: false),
          lineTouchData: const LineTouchData(enabled: false),
          lineBarsData: [
            LineChartBarData(
              spots: spots,
              isCurved: true,
              color: color,
              barWidth: 1.5,
              dotData: const FlDotData(show: false),
              belowBarData: BarAreaData(
                show: true,
                color: color.withOpacity(0.08),
              ),
            ),
          ],
        ),
        // No animation for live data — avoids the chart always being
        // 150ms behind the signal. Re-enable for static/history views.
        duration: isLive ? Duration.zero : const Duration(milliseconds: 200),
      ),
    );
  }
}
