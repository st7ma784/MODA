import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../services/ble_service.dart';
import '../services/signal_service.dart';
import '../widgets/band_power_card.dart';
import '../widgets/signal_chart_widget.dart';

class DashboardScreen extends StatelessWidget {
  const DashboardScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final ble = context.watch<BleService>();
    final signal = context.watch<SignalService>();
    final theme = Theme.of(context);

    final samples = signal.recentSamples;
    final powers = signal.bandPowers;
    final maxPower = powers.values.fold(0.0, math.max);

    double norm(String band) =>
        maxPower > 0 ? (powers[band]! / maxPower).clamp(0.0, 1.0) : 0.0;

    return Scaffold(
      appBar: AppBar(
        title: const Text('MODA'),
        backgroundColor: Colors.transparent,
        elevation: 0,
        actions: [
          _ServerDot(status: signal.serverStatus),
          const SizedBox(width: 8),
          _ConnectionBadge(ble: ble),
          const SizedBox(width: 12),
        ],
      ),
      body: ListView(
        padding: const EdgeInsets.fromLTRB(16, 8, 16, 24),
        children: [
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Text('Live Signal',
                          style: theme.textTheme.labelLarge
                              ?.copyWith(color: theme.colorScheme.primary)),
                      const Spacer(),
                      if (ble.isStreaming)
                        _PulseDot(color: theme.colorScheme.secondary),
                    ],
                  ),
                  const SizedBox(height: 8),
                  SignalChartWidget(
                    height: 120,
                    data: samples.isEmpty ? null : samples,
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 12),
          Text('Band Powers', style: theme.textTheme.labelLarge),
          const SizedBox(height: 8),
          Row(
            children: [
              Expanded(
                  child: BandPowerCard(
                      band: 'Delta',
                      hz: '0.5–4 Hz',
                      color: Colors.purple,
                      power: norm('delta'))),
              const SizedBox(width: 8),
              Expanded(
                  child: BandPowerCard(
                      band: 'Theta',
                      hz: '4–8 Hz',
                      color: Colors.blue,
                      power: norm('theta'))),
              const SizedBox(width: 8),
              Expanded(
                  child: BandPowerCard(
                      band: 'Alpha',
                      hz: '8–12 Hz',
                      color: Colors.teal,
                      power: norm('alpha'))),
            ],
          ),
          const SizedBox(height: 8),
          Row(
            children: [
              Expanded(
                  child: BandPowerCard(
                      band: 'Beta',
                      hz: '12–30 Hz',
                      color: Colors.orange,
                      power: norm('beta'))),
              const SizedBox(width: 8),
              Expanded(
                  child: BandPowerCard(
                      band: 'Gamma',
                      hz: '30–100 Hz',
                      color: Colors.red,
                      power: norm('gamma'))),
              const SizedBox(width: 8),
              const Expanded(child: SizedBox()),
            ],
          ),
          const SizedBox(height: 16),
          Row(
            children: [
              Expanded(
                  child: _MetricCard(
                      label: 'Sample Rate',
                      value: signal.sampleRate.toStringAsFixed(0),
                      unit: 'Hz')),
              const SizedBox(width: 8),
              Expanded(
                  child: _MetricCard(
                      label: 'Dominant Freq',
                      value: signal.hasData
                          ? signal.dominantFreq.toStringAsFixed(1)
                          : '—',
                      unit: 'Hz')),
              const SizedBox(width: 8),
              Expanded(
                  child: _MetricCard(
                      label: 'Signal Quality',
                      value: signal.hasData
                          ? signal.signalQuality.toStringAsFixed(0)
                          : '—',
                      unit: '%')),
            ],
          ),
        ],
      ),
    );
  }
}

class _ServerDot extends StatelessWidget {
  final ServerStatus status;
  const _ServerDot({required this.status});

  @override
  Widget build(BuildContext context) {
    final color = switch (status) {
      ServerStatus.up => Colors.green,
      ServerStatus.down => Colors.red,
      ServerStatus.checking => Colors.orange,
      ServerStatus.unknown => Colors.white24,
    };
    return Tooltip(
      message: switch (status) {
        ServerStatus.up => 'Server online',
        ServerStatus.down => 'Server offline',
        ServerStatus.checking => 'Checking server…',
        ServerStatus.unknown => 'Server status unknown',
      },
      child: Container(
        width: 8,
        height: 8,
        decoration: BoxDecoration(shape: BoxShape.circle, color: color),
      ),
    );
  }
}

class _PulseDot extends StatefulWidget {
  final Color color;
  const _PulseDot({required this.color});

  @override
  State<_PulseDot> createState() => _PulseDotState();
}

class _PulseDotState extends State<_PulseDot>
    with SingleTickerProviderStateMixin {
  late final AnimationController _ctrl;

  @override
  void initState() {
    super.initState();
    _ctrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 900),
    )..repeat(reverse: true);
  }

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return FadeTransition(
      opacity: _ctrl,
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 7,
            height: 7,
            decoration: BoxDecoration(
                shape: BoxShape.circle, color: widget.color),
          ),
          const SizedBox(width: 4),
          Text('LIVE',
              style: TextStyle(
                  fontSize: 10,
                  color: widget.color,
                  fontWeight: FontWeight.w700,
                  letterSpacing: 0.8)),
        ],
      ),
    );
  }
}

class _ConnectionBadge extends StatelessWidget {
  final BleService ble;
  const _ConnectionBadge({required this.ble});

  @override
  Widget build(BuildContext context) {
    final connected = ble.isConnected;
    final color = connected ? Colors.green : Colors.red;
    final label = connected
        ? (ble.connectedDevice!.platformName.isNotEmpty
            ? ble.connectedDevice!.platformName
            : 'Connected')
        : 'No device';
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        color: color.withOpacity(0.15),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: color, width: 1),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            connected ? Icons.bluetooth_connected : Icons.bluetooth_disabled,
            size: 13,
            color: color,
          ),
          const SizedBox(width: 5),
          Text(label, style: TextStyle(fontSize: 12, color: color)),
        ],
      ),
    );
  }
}

class _MetricCard extends StatelessWidget {
  final String label;
  final String value;
  final String unit;

  const _MetricCard(
      {required this.label, required this.value, required this.unit});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Card(
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 8),
        child: Column(
          children: [
            Text(label,
                textAlign: TextAlign.center,
                style: theme.textTheme.labelSmall?.copyWith(
                    color: theme.colorScheme.onSurface.withOpacity(0.5))),
            const SizedBox(height: 4),
            Text.rich(TextSpan(children: [
              TextSpan(
                  text: value,
                  style: theme.textTheme.titleMedium
                      ?.copyWith(color: theme.colorScheme.primary)),
              TextSpan(
                  text: ' $unit',
                  style: theme.textTheme.labelSmall
                      ?.copyWith(color: Colors.white38)),
            ])),
          ],
        ),
      ),
    );
  }
}
