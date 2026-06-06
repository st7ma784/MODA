import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';

enum ChartType { timeDomain, spectrum }

class SignalChartWidget extends StatefulWidget {
  final double height;
  final ChartType type;
  final double sampleRate;

  /// Live data. When null a "Waiting for signal…" placeholder is shown.
  final List<double>? data;

  const SignalChartWidget({
    super.key,
    required this.height,
    this.type = ChartType.timeDomain,
    this.data,
    this.sampleRate = 256.0,
  });

  @override
  State<SignalChartWidget> createState() => _SignalChartWidgetState();
}

class _SignalChartWidgetState extends State<SignalChartWidget> {
  static const int _maxDisplayPts = 300;

  List<FlSpot>? _cachedSpots;
  List<double>? _prevData;

  List<FlSpot> _buildSpots(List<double> raw) {
    if (raw.length <= _maxDisplayPts) {
      return [for (int i = 0; i < raw.length; i++) FlSpot(i.toDouble(), raw[i])];
    }
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

    final totalSeconds = data.length / widget.sampleRate;
    final n = spots.length;
    final tickInterval = n > 4 ? (n - 1) / 4.0 : 1.0;

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
          titlesData: FlTitlesData(
            leftTitles:
                const AxisTitles(sideTitles: SideTitles(showTitles: false)),
            rightTitles:
                const AxisTitles(sideTitles: SideTitles(showTitles: false)),
            topTitles:
                const AxisTitles(sideTitles: SideTitles(showTitles: false)),
            bottomTitles: AxisTitles(
              sideTitles: SideTitles(
                showTitles: true,
                reservedSize: 16,
                interval: tickInterval,
                getTitlesWidget: (value, meta) {
                  final t = value / math.max(1, n - 1) * totalSeconds;
                  return Padding(
                    padding: const EdgeInsets.only(top: 2),
                    child: Text(
                      '${t.toStringAsFixed(1)}s',
                      style:
                          const TextStyle(fontSize: 8, color: Colors.white38),
                    ),
                  );
                },
              ),
            ),
          ),
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
        duration: Duration.zero,
      ),
    );
  }
}
