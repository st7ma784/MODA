import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';

enum ChartType { timeDomain, spectrum }

class SignalChartWidget extends StatefulWidget {
  final double height;
  final ChartType type;

  /// Live data. When null a "Waiting for signal…" placeholder is shown.
  final List<double>? data;

  const SignalChartWidget({
    super.key,
    required this.height,
    this.type = ChartType.timeDomain,
    this.data,
  });

  @override
  State<SignalChartWidget> createState() => _SignalChartWidgetState();
}

class _SignalChartWidgetState extends State<SignalChartWidget> {
  static const int _maxDisplayPts = 150;

  List<FlSpot>? _cachedSpots;
  List<double>? _prevData;

  List<FlSpot> _buildSpots(List<double> raw) {
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

  @override
  Widget build(BuildContext context) {
    final data = widget.data;

    if (data == null || data.isEmpty) {
      return SizedBox(
        height: widget.height,
        child: Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(
                widget.type == ChartType.spectrum
                    ? Icons.bar_chart_outlined
                    : Icons.show_chart,
                size: 28,
                color: Colors.white24,
              ),
              const SizedBox(height: 6),
              const Text(
                'Waiting for signal…',
                style: TextStyle(fontSize: 12, color: Colors.white38),
              ),
            ],
          ),
        ),
      );
    }

    // Only recompute spots when the data reference changes.
    if (!identical(data, _prevData)) {
      _cachedSpots = _buildSpots(data);
      _prevData = data;
    }
    final spots = _cachedSpots!;

    final color = Theme.of(context).colorScheme.primary;

    final ys = spots.map((s) => s.y);
    final lo = ys.reduce(math.min);
    final hi = ys.reduce(math.max);
    final range = (hi - lo).abs();
    final pad = range * 0.10 + 0.01;

    return SizedBox(
      height: widget.height,
      child: LineChart(
        LineChartData(
          minY: lo - pad,
          maxY: hi + pad,
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
                color: color.withValues(alpha: 0.08),
              ),
            ),
          ],
        ),
        // No animation for live data — avoids the chart always being
        // 150ms behind the signal.
        duration: Duration.zero,
      ),
    );
  }
}
