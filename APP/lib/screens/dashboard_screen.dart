import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../services/ble_service.dart';
import '../services/signal_service.dart';
import '../services/audio_capture_service.dart';
import '../theme/app_theme.dart';
import '../utils/signal_bands.dart';
import '../widgets/band_power_card.dart';
import '../widgets/plotly_chart_widget.dart';
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
          const SizedBox(height: 16),
          const _ServerAnalysisSection(),
          const SizedBox(height: 16),
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

// ── Server Analysis Section ───────────────────────────────────────────────────

class _ServerAnalysisSection extends StatefulWidget {
  const _ServerAnalysisSection();

  @override
  State<_ServerAnalysisSection> createState() => _ServerAnalysisSectionState();
}

class _ServerAnalysisSectionState extends State<_ServerAnalysisSection> {
  bool _autoFired = false;

  @override
  Widget build(BuildContext context) {
    final signal = context.watch<SignalService>();

    // Auto-fire analysis once when server comes up and data is available.
    if (!_autoFired &&
        signal.serverStatus == ServerStatus.up &&
        signal.hasData &&
        signal.lastResult == null &&
        !signal.isSubmitting) {
      _autoFired = true;
      WidgetsBinding.instance.addPostFrameCallback(
          (_) { if (mounted) signal.submitAnalysis(); });
    }

    if (signal.serverStatus != ServerStatus.up) return const SizedBox.shrink();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Section divider
        Row(children: [
          const Expanded(child: Divider()),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 10),
            child: Text(
              'SERVER ANALYSIS',
              style: TextStyle(
                fontSize: 10,
                letterSpacing: 1.4,
                fontWeight: FontWeight.w700,
                color: Theme.of(context).colorScheme.primary,
              ),
            ),
          ),
          const Expanded(child: Divider()),
        ]),
        const SizedBox(height: 8),

        // Spectral Analysis panel
        _AnalysisPanel(
          title: 'Spectral Analysis',
          action: signal.isSubmitting
              ? const SizedBox(
                  width: 16, height: 16,
                  child: CircularProgressIndicator(
                      strokeWidth: 2, color: Colors.white70))
              : IconButton(
                  icon: const Icon(Icons.refresh, size: 18, color: Colors.white70),
                  padding: EdgeInsets.zero,
                  constraints: const BoxConstraints(),
                  tooltip: 'Re-analyse',
                  onPressed: signal.hasData
                      ? () {
                          setState(() => _autoFired = true);
                          signal.submitAnalysis();
                        }
                      : null,
                ),
          child: _SpectralPanelBody(signal: signal),
        ),

        const SizedBox(height: 12),

        // MODWT panel
        _AnalysisPanel(
          title: 'Wavelet Decomposition',
          subtitle: 'MODWT · LA8',
          action: signal.isSubmittingModwt
              ? const SizedBox(
                  width: 16, height: 16,
                  child: CircularProgressIndicator(
                      strokeWidth: 2, color: Colors.white70))
              : IconButton(
                  icon: Icon(
                    signal.modwtResult != null
                        ? Icons.refresh
                        : Icons.play_arrow,
                    size: 18,
                    color: Colors.white70,
                  ),
                  padding: EdgeInsets.zero,
                  constraints: const BoxConstraints(),
                  tooltip: signal.modwtResult != null
                      ? 'Re-run MODWT'
                      : 'Run MODWT',
                  onPressed: signal.hasData && !signal.isSubmittingModwt
                      ? signal.submitModwt
                      : null,
                ),
          child: _ModwtPanelBody(signal: signal),
        ),
      ],
    );
  }
}

// ── MATLAB-style gradient panel container ─────────────────────────────────────

class _AnalysisPanel extends StatelessWidget {
  final String title;
  final String? subtitle;
  final Widget? action;
  final Widget child;

  const _AnalysisPanel({
    required this.title,
    required this.child,
    this.subtitle,
    this.action,
  });

  @override
  Widget build(BuildContext context) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(12),
      child: ColoredBox(
        color: AppTheme.surface,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Container(
              decoration: const BoxDecoration(
                gradient: LinearGradient(
                  colors: [AppTheme.primary, AppTheme.secondary],
                  begin: Alignment.centerLeft,
                  end: Alignment.centerRight,
                ),
              ),
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 9),
              child: Row(children: [
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Text(title,
                          style: Theme.of(context)
                              .textTheme
                              .titleSmall
                              ?.copyWith(
                                  color: Colors.white,
                                  fontWeight: FontWeight.w600)),
                      if (subtitle != null)
                        Text(subtitle!,
                            style: const TextStyle(
                                fontSize: 10, color: Colors.white70)),
                    ],
                  ),
                ),
                if (action != null) action!,
              ]),
            ),
            Padding(
              padding: const EdgeInsets.all(12),
              child: child,
            ),
          ],
        ),
      ),
    );
  }
}

// ── Spectral panel body ───────────────────────────────────────────────────────

class _SpectralPanelBody extends StatelessWidget {
  final SignalService signal;
  const _SpectralPanelBody({required this.signal});

  static bool _isFigureJson(dynamic v) =>
      v is String && v.trimLeft().startsWith('{') && v.contains('"data"');

  // Poll result may nest plots at top level or inside a 'results' sub-map.
  static Map<String, dynamic> _flatten(Map<String, dynamic> raw) {
    final out = Map<String, dynamic>.from(raw);
    final nested = raw['results'];
    if (nested is Map<String, dynamic>) out.addAll(nested);
    return out;
  }

  @override
  Widget build(BuildContext context) {
    if (signal.isSubmitting) {
      return const _Placeholder(
          icon: Icons.hourglass_top_outlined,
          message: 'Analysing signal on server…',
          animate: true);
    }
    if (!signal.hasData) {
      return const _Placeholder(
          icon: Icons.sensors_off_outlined,
          message: 'Connect a device and stream signal data');
    }
    final raw = signal.lastResult;
    if (raw == null) {
      return const _Placeholder(
          icon: Icons.cloud_outlined,
          message: 'Awaiting server analysis…',
          animate: true);
    }

    final r = _flatten(raw);
    final nCp = _toInt(r['num_changepoints']);
    final nWin = _toInt(r['num_windows']);
    final freqSummary = r['frequency_summary'] as List?;
    final domHz = (freqSummary != null && freqSummary.isNotEmpty)
        ? (freqSummary.first['frequency'] as num?)?.toStringAsFixed(1)
        : null;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        if (nCp != null || domHz != null)
          Padding(
            padding: const EdgeInsets.only(bottom: 10),
            child: Wrap(spacing: 16, runSpacing: 6, children: [
              if (nCp != null) _MetricPill(label: 'Changepoints', value: '$nCp'),
              if (nWin != null) _MetricPill(label: 'Windows', value: '$nWin'),
              if (domHz != null)
                _MetricPill(label: 'Dominant freq', value: '$domHz Hz'),
            ]),
          ),
        if (_isFigureJson(r['signal']))
          _NamedPlot(figureJson: r['signal'] as String, height: 240),
        if (_isFigureJson(r['spectrogram'])) ...[
          const SizedBox(height: 10),
          _NamedPlot(
              label: 'Time-Frequency Spectrogram',
              figureJson: r['spectrogram'] as String,
              height: 240),
        ],
        if (_isFigureJson(r['band_powers'])) ...[
          const SizedBox(height: 10),
          _NamedPlot(
              label: 'Band Powers',
              figureJson: r['band_powers'] as String,
              height: 200),
        ],
      ],
    );
  }

  static int? _toInt(dynamic v) =>
      v is int ? v : (v is num ? v.toInt() : null);
}

// ── MODWT panel body ──────────────────────────────────────────────────────────

class _ModwtPanelBody extends StatelessWidget {
  final SignalService signal;
  const _ModwtPanelBody({required this.signal});

  static bool _isFigureJson(dynamic v) =>
      v is String && v.trimLeft().startsWith('{') && v.contains('"data"');

  static Map<String, dynamic> _flatten(Map<String, dynamic> raw) {
    final out = Map<String, dynamic>.from(raw);
    final nested = raw['results'];
    if (nested is Map<String, dynamic>) out.addAll(nested);
    return out;
  }

  @override
  Widget build(BuildContext context) {
    if (signal.isSubmittingModwt) {
      return const _Placeholder(
          icon: Icons.hourglass_top_outlined,
          message: 'Running wavelet decomposition…',
          animate: true);
    }
    final raw = signal.modwtResult;
    if (raw == null) {
      return const _Placeholder(
          icon: Icons.layers_outlined,
          message: 'Press ▶ to run MODWT decomposition');
    }

    final r = _flatten(raw);
    final nLevels = r['n_levels'] as int?;
    final reconErr = (r['reconstruction_error'] as num?)?.toStringAsFixed(4);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        if (nLevels != null || reconErr != null)
          Padding(
            padding: const EdgeInsets.only(bottom: 10),
            child: Wrap(spacing: 16, runSpacing: 6, children: [
              if (nLevels != null)
                _MetricPill(label: 'Levels', value: '$nLevels'),
              if (reconErr != null)
                _MetricPill(label: 'Recon. error', value: reconErr),
            ]),
          ),
        if (_isFigureJson(r['coefficients_plot']))
          _NamedPlot(
              label: 'Wavelet Coefficients',
              figureJson: r['coefficients_plot'] as String,
              height: 300),
        if (_isFigureJson(r['energy_plot'])) ...[
          const SizedBox(height: 10),
          _NamedPlot(
              label: 'Band Energy',
              figureJson: r['energy_plot'] as String,
              height: 180),
        ],
      ],
    );
  }
}

// ── Shared helpers ────────────────────────────────────────────────────────────

class _Placeholder extends StatelessWidget {
  final IconData icon;
  final String message;
  final bool animate;
  const _Placeholder(
      {required this.icon, required this.message, this.animate = false});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 24),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          animate
              ? const SizedBox(
                  width: 28,
                  height: 28,
                  child: CircularProgressIndicator(strokeWidth: 2))
              : Icon(icon, size: 32, color: Colors.white24),
          const SizedBox(height: 10),
          Text(message,
              textAlign: TextAlign.center,
              style: const TextStyle(fontSize: 13, color: Colors.white38)),
        ],
      ),
    );
  }
}

class _MetricPill extends StatelessWidget {
  final String label;
  final String value;
  const _MetricPill({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    return RichText(
      text: TextSpan(
        style: DefaultTextStyle.of(context).style,
        children: [
          TextSpan(
              text: '$label: ',
              style: const TextStyle(fontSize: 11, color: Colors.white38)),
          TextSpan(
              text: value,
              style: const TextStyle(
                  fontSize: 11,
                  fontWeight: FontWeight.w600,
                  color: Colors.white70)),
        ],
      ),
    );
  }
}

class _NamedPlot extends StatelessWidget {
  final String figureJson;
  final double height;
  final String? label;
  const _NamedPlot(
      {required this.figureJson, this.height = 240, this.label});

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        if (label != null)
          Padding(
            padding: const EdgeInsets.only(bottom: 4),
            child: Text(label!,
                style: const TextStyle(
                    fontSize: 11,
                    color: Colors.white54,
                    fontWeight: FontWeight.w600)),
          ),
        ClipRRect(
          borderRadius: BorderRadius.circular(8),
          child: PlotlyChartWidget(figureJson: figureJson, height: height),
        ),
      ],
    );
  }
}
