import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../services/ble_service.dart';
import '../services/signal_service.dart';
import '../services/audio_capture_service.dart';
import '../theme/app_theme.dart';
import '../utils/signal_bands.dart';
import '../widgets/band_power_card.dart';
import '../widgets/processing_badge.dart';
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
        title: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Image.asset('assets/images/moda_logo.png', height: 32),
            const SizedBox(width: 10),
            const Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                Text('FastMODA'),
                Text(
                  'Signal Feature Extraction',
                  style: TextStyle(
                    fontSize: 10,
                    color: Colors.white54,
                    fontStyle: FontStyle.italic,
                  ),
                ),
              ],
            ),
          ],
        ),
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
                      const SizedBox(width: 8),
                      const ProcessingBadge(location: ProcessingLocation.device),
                      const Spacer(),
                      if (ble.isStreaming ||
                          signal.activeSource == InputSource.microphone)
                        _PulseDot(color: theme.colorScheme.secondary),
                    ],
                  ),
                  const SizedBox(height: 8),
                  SegmentedButton<InputSource>(
                    showSelectedIcon: false,
                    style: const ButtonStyle(
                      visualDensity: VisualDensity.compact,
                    ),
                    segments: const [
                      ButtonSegment(
                        value: InputSource.bluetooth,
                        label: Text('Bluetooth'),
                        icon: Icon(Icons.bluetooth, size: 16),
                      ),
                      ButtonSegment(
                        value: InputSource.microphone,
                        label: Text('Mic'),
                        icon: Icon(Icons.mic, size: 16),
                      ),
                    ],
                    selected: {signal.activeSource},
                    onSelectionChanged: (sel) =>
                        _switchSource(context, sel.first),
                  ),
                  const SizedBox(height: 8),
                  SignalChartWidget(
                    height: 160,
                    data: samples.isEmpty ? null : samples,
                    sampleRate: signal.sampleRate,
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
              for (int i = 0; i < 3; i++) ...[
                if (i > 0) const SizedBox(width: 8),
                Expanded(
                  child: BandPowerCard(
                    band: kBands[i].label(signal.signalType),
                    hz: kBands[i].hz(signal.signalType),
                    color: kBands[i].color,
                    power: norm(kBands[i].key),
                  ),
                ),
              ],
            ],
          ),
          const SizedBox(height: 8),
          Row(
            children: [
              for (int i = 3; i < 5; i++) ...[
                if (i > 3) const SizedBox(width: 8),
                Expanded(
                  child: BandPowerCard(
                    band: kBands[i].label(signal.signalType),
                    hz: kBands[i].hz(signal.signalType),
                    color: kBands[i].color,
                    power: norm(kBands[i].key),
                  ),
                ),
              ],
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
                  child: _DomFreqCard(signal: signal)),
              const SizedBox(width: 8),
              Expanded(
                  child: _QualityCard(signal: signal)),
            ],
          ),
          const SizedBox(height: 8),
          Row(
            children: [
              Expanded(
                child: _MetricCard(
                  label: 'Rhythmicity',
                  value: signal.hasData
                      ? (signal.rhythmicity * 100).toStringAsFixed(0)
                      : '—',
                  unit: '%',
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: _MetricCard(
                  label: 'Entropy',
                  value: signal.hasData
                      ? (signal.spectralEntropy * 100).toStringAsFixed(0)
                      : '—',
                  unit: '%',
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: _MetricCard(
                  label: 'Changepoints',
                  value: signal.changepoints.length.toString(),
                  unit: '',
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

/// Coordinates a source switch: starts/stops the mic and flips SignalService.
/// If the mic fails to start (e.g. permission denied), stays on the current
/// source — the error is surfaced via the AudioCaptureService error stream.
Future<void> _switchSource(BuildContext context, InputSource src) async {
  final signal = context.read<SignalService>();
  final audio = context.read<AudioCaptureService>();
  if (src == signal.activeSource) return;
  if (src == InputSource.microphone) {
    if (!await audio.start()) return;
    signal.setInputSource(InputSource.microphone);
  } else {
    await audio.stop();
    signal.setInputSource(InputSource.bluetooth);
  }
}

class _ServerDot extends StatelessWidget {
  final ServerStatus status;
  const _ServerDot({required this.status});

  @override
  Widget build(BuildContext context) {
    final color = switch (status) {
      ServerStatus.up => AppTheme.success,
      ServerStatus.down => AppTheme.danger,
      ServerStatus.checking => AppTheme.secondary,
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
    final color = connected ? AppTheme.success : AppTheme.danger;
    final label = connected
        ? (ble.connectedDevice!.platformName.isNotEmpty
            ? ble.connectedDevice!.platformName
            : 'Connected')
        : 'No device';
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.15),
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
                    color: theme.colorScheme.onSurface.withValues(alpha: 0.5))),
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

/// Dominant frequency card with a coloured band badge beneath the Hz value.
class _DomFreqCard extends StatelessWidget {
  final SignalService signal;
  const _DomFreqCard({required this.signal});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final band = signal.hasData ? bandForFreq(signal.dominantFreq, signal.signalType) : null;
    return Card(
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 8),
        child: Column(
          children: [
            Text('Dominant Freq',
                textAlign: TextAlign.center,
                style: theme.textTheme.labelSmall?.copyWith(
                    color: theme.colorScheme.onSurface.withValues(alpha: 0.5))),
            const SizedBox(height: 4),
            Text.rich(TextSpan(children: [
              TextSpan(
                  text: signal.hasData
                      ? signal.dominantFreq.toStringAsFixed(1)
                      : '—',
                  style: theme.textTheme.titleMedium
                      ?.copyWith(color: theme.colorScheme.primary)),
              TextSpan(
                  text: signal.hasData ? ' Hz' : '',
                  style: theme.textTheme.labelSmall
                      ?.copyWith(color: Colors.white38)),
            ])),
            if (band != null) ...[
              const SizedBox(height: 4),
              Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                decoration: BoxDecoration(
                  color: band.color.withValues(alpha: 0.18),
                  borderRadius: BorderRadius.circular(4),
                  border: Border.all(
                      color: band.color.withValues(alpha: 0.5), width: 0.8),
                ),
                child: Text(
                  band.label(signal.signalType),
                  style: TextStyle(
                      fontSize: 9,
                      color: band.color,
                      fontWeight: FontWeight.w700),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

/// Signal quality card with traffic-light colour coding.
class _QualityCard extends StatelessWidget {
  final SignalService signal;
  const _QualityCard({required this.signal});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final q = signal.hasData ? signal.signalQuality : -1;
    final color = q < 0
        ? Colors.white38
        : q >= 70
            ? AppTheme.success
            : q >= 40
                ? AppTheme.secondary
                : AppTheme.danger;
    return Card(
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 8),
        child: Column(
          children: [
            Text('Signal Quality',
                textAlign: TextAlign.center,
                style: theme.textTheme.labelSmall?.copyWith(
                    color: theme.colorScheme.onSurface.withValues(alpha: 0.5))),
            const SizedBox(height: 4),
            Text.rich(TextSpan(children: [
              TextSpan(
                  text: q >= 0 ? q.toStringAsFixed(0) : '—',
                  style: theme.textTheme.titleMedium?.copyWith(color: color)),
              TextSpan(
                  text: q >= 0 ? ' %' : '',
                  style:
                      theme.textTheme.labelSmall?.copyWith(color: Colors.white38)),
            ])),
          ],
        ),
      ),
    );
  }
}
