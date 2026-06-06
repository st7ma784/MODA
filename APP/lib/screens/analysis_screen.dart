import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:file_picker/file_picker.dart';
import '../services/analysis_history_service.dart';
import '../services/signal_service.dart';
import '../utils/export.dart';
import '../utils/signal_bands.dart';
import '../widgets/signal_chart_widget.dart';
import '../widgets/spectrogram_widget.dart';

class AnalysisScreen extends StatefulWidget {
  const AnalysisScreen({super.key});

  @override
  State<AnalysisScreen> createState() => _AnalysisScreenState();
}

class _AnalysisScreenState extends State<AnalysisScreen>
    with SingleTickerProviderStateMixin {
  late final TabController _tabs;

  // Instance-scoped set — resets with the widget, not a global leak.
  final _savedTaskIds = <String>{};

  @override
  void initState() {
    super.initState();
    _tabs = TabController(length: 4, vsync: this);
    WidgetsBinding.instance.addPostFrameCallback((_) {
      context.read<SignalService>().addListener(_onSignalChange);
    });
  }

  void _onSignalChange() {
    if (!mounted) return;
    final signal = context.read<SignalService>();
    final history = context.read<AnalysisHistoryService>();
    _autoSave(history, signal);
  }

  void _autoSave(AnalysisHistoryService history, SignalService signal) {
    void tryStore(Map<String, dynamic>? result, String type) {
      if (result == null) return;
      final taskId = result['task_id'] as String? ?? type;
      if (_savedTaskIds.contains(taskId)) return;
      _savedTaskIds.add(taskId);
      history.save(AnalysisRecord.fromResult(
          taskId, type, result, signal.sampleRate, 512));
    }

    tryStore(signal.lastResult, 'spectral');
    tryStore(signal.bispectrumResult, 'bispectrum');
    tryStore(signal.coherenceResult, 'coherence');
    tryStore(signal.bayesianResult, 'bayesian');
    tryStore(signal.stftResult, 'stft');
    tryStore(signal.cwtResult, 'cwt');
    tryStore(signal.hilbertResult, 'hilbert');
    tryStore(signal.surrogatesResult, 'surrogates');
    tryStore(signal.featuresResult, 'features');
    tryStore(signal.syncMapResult, 'syncmap');
    tryStore(signal.biphaseResult, 'biphase');
    tryStore(signal.bispec4Result, 'bispectrum4');
    tryStore(signal.couplingResult, 'coupling');
    tryStore(signal.ridgeResult, 'ridge');
    tryStore(signal.filterResult, 'filter');
    tryStore(signal.wftResult, 'wft');
    tryStore(signal.modwtResult, 'modwt');
    tryStore(signal.groupResult, 'group');
  }

  @override
  void dispose() {
    if (mounted) context.read<SignalService>().removeListener(_onSignalChange);
    _tabs.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final signal = context.watch<SignalService>();

    return Scaffold(
      appBar: AppBar(
        title: const Text('Analysis'),
        backgroundColor: Colors.transparent,
        elevation: 0,
        bottom: TabBar(
          controller: _tabs,
          isScrollable: true,
          tabAlignment: TabAlignment.start,
          tabs: [
            const Tab(text: 'Live'),
            const Tab(text: 'Spectral'),
            Tab(
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Text('Server'),
                  const SizedBox(width: 6),
                  _statusDot(signal.serverStatus),
                ],
              ),
            ),
            const Tab(text: 'History'),
          ],
        ),
      ),
      body: TabBarView(
        controller: _tabs,
        children: [
          _LiveTab(signal: signal),
          _SpectralTab(signal: signal),
          _ServerTab(signal: signal),
          const _HistoryTab(),
        ],
      ),
    );
  }

  Widget _statusDot(ServerStatus status) {
    if (status == ServerStatus.checking) {
      return const SizedBox(
          width: 8,
          height: 8,
          child: CircularProgressIndicator(strokeWidth: 1.5));
    }
    final color = switch (status) {
      ServerStatus.up => Colors.green,
      ServerStatus.down => Colors.red,
      _ => Colors.white24,
    };
    return Container(
        width: 8,
        height: 8,
        decoration: BoxDecoration(shape: BoxShape.circle, color: color));
  }
}

class _LiveTab extends StatelessWidget {
  final SignalService signal;
  const _LiveTab({required this.signal});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final samples = signal.recentSamples;
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        Card(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('Real-Time Signal',
                    style: theme.textTheme.labelLarge
                        ?.copyWith(color: theme.colorScheme.primary)),
                const SizedBox(height: 8),
                SignalChartWidget(
                    height: 160,
                    data: samples.isEmpty ? null : samples),
              ],
            ),
          ),
        ),
        const SizedBox(height: 12),
        Card(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('Power Spectrum (FFT)',
                    style: theme.textTheme.labelLarge
                        ?.copyWith(color: theme.colorScheme.primary)),
                const SizedBox(height: 8),
                SignalChartWidget(
                  height: 140,
                  type: ChartType.spectrum,
                  data: signal.spectrum.isEmpty ? null : signal.spectrum,
                ),
              ],
            ),
          ),
        ),
        const SizedBox(height: 12),
        _ChangepointCard(signal: signal),
      ],
    );
  }
}

class _ChangepointCard extends StatelessWidget {
  final SignalService signal;
  const _ChangepointCard({required this.signal});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cps = signal.changepoints;

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Changepoints',
                style: theme.textTheme.labelLarge
                    ?.copyWith(color: theme.colorScheme.primary)),
            const SizedBox(height: 8),
            if (cps.isNotEmpty)
              Wrap(
                spacing: 6,
                runSpacing: 6,
                children: cps
                    .map((idx) => Chip(
                          label: Text(
                            '${(idx / signal.sampleRate).toStringAsFixed(2)}s',
                            style: const TextStyle(fontSize: 11),
                          ),
                          padding: EdgeInsets.zero,
                          visualDensity: VisualDensity.compact,
                        ))
                    .toList(),
              )
            else
              Container(
                height: 50,
                decoration: BoxDecoration(
                  color: Colors.white.withValues(alpha: 0.04),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: const Center(
                  child: Text('Collecting signal data…',
                      style: TextStyle(color: Colors.white38, fontSize: 13)),
                ),
              ),
          ],
        ),
      ),
    );
  }
}

class _SpectralTab extends StatelessWidget {
  final SignalService signal;
  const _SpectralTab({required this.signal});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final powers = signal.bandPowers;

    if (!signal.hasData) {
      return const Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.analytics, size: 64, color: Colors.white24),
            SizedBox(height: 16),
            Text('Connect a BLE device and start streaming',
                style: TextStyle(color: Colors.white54)),
            SizedBox(height: 6),
            Text('Full MODWT + PSD breakdown will appear here',
                style: TextStyle(color: Colors.white38, fontSize: 13)),
          ],
        ),
      );
    }

    final maxPower = powers.values.fold(0.0, math.max);

    double norm(String key) =>
        maxPower > 0 ? (powers[key]! / maxPower).clamp(0.0, 1.0) : 0.0;

    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        Card(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('Live Spectrogram',
                    style: theme.textTheme.labelLarge
                        ?.copyWith(color: theme.colorScheme.primary)),
                const SizedBox(height: 8),
                SpectrogramWidget(
                  history: signal.spectrogramHistory,
                  sampleRate: signal.sampleRate,
                  height: 140,
                ),
              ],
            ),
          ),
        ),
        const SizedBox(height: 12),
        Card(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('Power Spectral Density',
                    style: theme.textTheme.labelLarge
                        ?.copyWith(color: theme.colorScheme.primary)),
                const SizedBox(height: 8),
                SignalChartWidget(
                  height: 180,
                  type: ChartType.spectrum,
                  data: signal.spectrum.isEmpty ? null : signal.spectrum,
                ),
              ],
            ),
          ),
        ),
        const SizedBox(height: 12),
        Row(
          children: [
            Expanded(
              child: _MetricChip(
                label: 'Entropy',
                value: '${(signal.spectralEntropy * 100).toStringAsFixed(1)}%',
                tooltip: '0% = pure tone  ·  100% = white noise',
              ),
            ),
            const SizedBox(width: 8),
            Expanded(
              child: _MetricChip(
                label: 'Flatness',
                value: '${(signal.spectralFlatness * 100).toStringAsFixed(1)}%',
                tooltip: '0% = tonal  ·  100% = noise-like',
              ),
            ),
          ],
        ),
        const SizedBox(height: 12),
        Text('Band Power Breakdown', style: theme.textTheme.labelLarge),
        const SizedBox(height: 8),
        ...kBands.map((b) => Padding(
          padding: const EdgeInsets.only(bottom: 8),
          child: _BandRow(
            name: b.label(signal.signalType),
            hz: b.hz(signal.signalType),
            color: b.color,
            power: norm(b.key),
          ),
        )),
      ],
    );
  }
}

class _BandRow extends StatelessWidget {
  final String name, hz;
  final Color color;
  final double power;
  const _BandRow(
      {required this.name,
      required this.hz,
      required this.color,
      required this.power});

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
        child: Row(
          children: [
            Container(
                width: 10,
                height: 10,
                decoration:
                    BoxDecoration(color: color, shape: BoxShape.circle)),
            const SizedBox(width: 10),
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(name,
                    style: const TextStyle(
                        fontSize: 13, fontWeight: FontWeight.w600)),
                Text(hz,
                    style:
                        const TextStyle(fontSize: 11, color: Colors.white38)),
              ],
            ),
            const SizedBox(width: 12),
            Expanded(
              child: ClipRRect(
                borderRadius: BorderRadius.circular(4),
                child: LinearProgressIndicator(
                  value: power.clamp(0.0, 1.0),
                  backgroundColor: Colors.white10,
                  valueColor: AlwaysStoppedAnimation(color),
                  minHeight: 8,
                ),
              ),
            ),
            const SizedBox(width: 10),
            SizedBox(
              width: 38,
              child: Text(
                '${(power * 100).toStringAsFixed(0)}%',
                textAlign: TextAlign.right,
                style:
                    const TextStyle(fontSize: 12, color: Colors.white54),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _ServerTab extends StatelessWidget {
  final SignalService signal;
  const _ServerTab({required this.signal});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final status = signal.serverStatus;

    if (status == ServerStatus.down || status == ServerStatus.unknown) {
      return Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(Icons.cloud_off, size: 64, color: Colors.white24),
            const SizedBox(height: 16),
            const Text('Server not reachable',
                style: TextStyle(color: Colors.white54)),
            const SizedBox(height: 8),
            const Padding(
              padding: EdgeInsets.symmetric(horizontal: 48),
              child: Text(
                'Configure the server URL in Settings,\nor set up a local FastMODA instance.',
                textAlign: TextAlign.center,
                style: TextStyle(color: Colors.white38, fontSize: 13),
              ),
            ),
            const SizedBox(height: 24),
            OutlinedButton.icon(
              onPressed: () => signal.forceHealthCheck(),
              icon: const Icon(Icons.refresh),
              label: const Text('Retry'),
            ),
          ],
        ),
      );
    }

    final anyResult = signal.lastResult != null ||
        signal.bispectrumResult != null ||
        signal.stftResult != null ||
        signal.cwtResult != null ||
        signal.hilbertResult != null ||
        signal.surrogatesResult != null ||
        signal.featuresResult != null ||
        signal.biphaseResult != null ||
        signal.bispec4Result != null ||
        signal.couplingResult != null ||
        signal.ridgeResult != null ||
        signal.filterResult != null ||
        signal.wftResult != null ||
        signal.syncMapResult != null ||
        signal.coherenceResult != null ||
        signal.bayesianResult != null ||
        signal.modwtResult != null ||
        signal.groupResult != null;

    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        // ── Server status ─────────────────────────────────────────────
        Card(
          child: ListTile(
            leading: const Icon(Icons.check_circle, color: Colors.green),
            title: const Text('Server connected'),
            subtitle: Text(signal.gpuAvailable
                ? 'GPU backend active'
                : 'CPU backend (scipy fallback)'),
            trailing: status == ServerStatus.checking
                ? const SizedBox(
                    width: 18,
                    height: 18,
                    child: CircularProgressIndicator(strokeWidth: 2))
                : signal.gpuAvailable
                    ? Chip(
                        label: const Text('GPU',
                            style: TextStyle(
                                fontSize: 10, fontWeight: FontWeight.bold)),
                        backgroundColor: Colors.green.withValues(alpha: 0.2),
                        side: const BorderSide(color: Colors.green),
                        visualDensity: VisualDensity.compact,
                        padding: EdgeInsets.zero,
                      )
                    : null,
          ),
        ),
        const SizedBox(height: 16),

        // ── Full analysis (MODWT + changepoints) ──────────────────────
        Text('Signal Analysis', style: theme.textTheme.labelLarge),
        const SizedBox(height: 8),
        FilledButton.icon(
          onPressed: signal.isSubmitting || !signal.hasData
              ? null
              : () => signal.submitAnalysis(),
          icon: signal.isSubmitting
              ? const SizedBox(
                  width: 16,
                  height: 16,
                  child: CircularProgressIndicator(
                      strokeWidth: 2, color: Colors.black))
              : const Icon(Icons.send),
          label: Text(signal.isSubmitting
              ? signal.pendingTaskId != null
                  ? 'Analysing… (${signal.pendingTaskId!.substring(0, 8)})'
                  : 'Submitting…'
              : signal.hasData
                  ? 'Submit Signal for Analysis'
                  : 'No signal data yet'),
        ),
        if (signal.lastResult != null) ...[
          const SizedBox(height: 12),
          _SummaryCard(result: signal.lastResult!),
          if (_getFrequencySummary(signal.lastResult!) != null) ...[
            const SizedBox(height: 8),
            _FrequencySummaryCard(
                summary: _getFrequencySummary(signal.lastResult!)!),
          ],
          if (_getSurrogateStats(signal.lastResult!) != null) ...[
            const SizedBox(height: 8),
            _SurrogateStatsRow(
                stats: _getSurrogateStats(signal.lastResult!)!),
          ],
        ],
        const SizedBox(height: 20),

        // ── Channel import (unlocks coherence / bayesian) ──────────────
        _ChannelImportRow(signal: signal),
        const SizedBox(height: 16),

        // ── Single-channel analyses ────────────────────────────────────
        Text('Single-Channel Analyses', style: theme.textTheme.labelLarge),
        const SizedBox(height: 8),
        _AnalysisCard(
          title: 'STFT',
          subtitle: 'Short-Time Fourier Transform',
          icon: Icons.bar_chart,
          busy: signal.isSubmittingStft,
          canRun: signal.hasData,
          result: signal.stftResult,
          onRun: () => signal.submitStft(),
        ),
        const SizedBox(height: 8),
        _AnalysisCard(
          title: 'WFT (Gaussian STFT)',
          subtitle: 'Optimal time-frequency localisation',
          icon: Icons.grain,
          busy: signal.isSubmittingWft,
          canRun: signal.hasData,
          result: signal.wftResult,
          onRun: () => signal.submitWft(),
        ),
        const SizedBox(height: 8),
        _AnalysisCard(
          title: 'CWT',
          subtitle: 'Continuous Morlet Wavelet Transform',
          icon: Icons.waves,
          busy: signal.isSubmittingCwt,
          canRun: signal.hasData,
          result: signal.cwtResult,
          onRun: () => signal.submitCwt(),
        ),
        const SizedBox(height: 8),
        _AnalysisCard(
          title: 'Hilbert Phase',
          subtitle: 'Instantaneous amplitude, phase & frequency',
          icon: Icons.rotate_right,
          busy: signal.isSubmittingHilbert,
          canRun: signal.hasData,
          result: signal.hilbertResult,
          onRun: () => signal.submitHilbert(),
        ),
        const SizedBox(height: 8),
        _AnalysisCard(
          title: 'Ridge Extraction',
          subtitle: 'Instantaneous freq / amplitude / phase',
          icon: Icons.show_chart,
          busy: signal.isSubmittingRidge,
          canRun: signal.hasData,
          result: signal.ridgeResult,
          onRun: () => signal.submitRidge(),
        ),
        const SizedBox(height: 8),
        _AnalysisCard(
          title: 'Butterworth Filter',
          subtitle: 'Bandpass + polynomial detrend',
          icon: Icons.filter_alt,
          busy: signal.isSubmittingFilter,
          canRun: signal.hasData,
          result: signal.filterResult,
          onRun: () => _showFilterDialog(context, signal),
        ),
        const SizedBox(height: 8),
        _AnalysisCard(
          title: 'MODWT',
          subtitle: 'Maximal-overlap discrete wavelet decomposition',
          icon: Icons.layers,
          busy: signal.isSubmittingModwt,
          canRun: signal.hasData,
          result: signal.modwtResult,
          onRun: () => _showModwtDialog(context, signal),
        ),
        const SizedBox(height: 8),
        _AnalysisCard(
          title: 'Bispectrum',
          subtitle: 'Quadratic phase coupling',
          icon: Icons.grid_4x4,
          busy: signal.isSubmittingBispectrum,
          canRun: signal.hasData,
          result: signal.bispectrumResult,
          onRun: () => signal.submitBispectrum(),
        ),
        const SizedBox(height: 8),
        _AnalysisCard(
          title: 'Surrogate Test',
          subtitle: 'Statistical significance testing',
          icon: Icons.science,
          busy: signal.isSubmittingSurrogates,
          canRun: signal.hasData,
          result: signal.surrogatesResult,
          onRun: () => _showSurrogateDialog(context, signal),
        ),
        const SizedBox(height: 8),
        _AnalysisCard(
          title: 'Feature Extraction',
          subtitle: 'ML-ready spectral + phase feature vector',
          icon: Icons.table_rows,
          busy: signal.isSubmittingFeatures,
          canRun: signal.hasData,
          result: signal.featuresResult,
          onRun: () => signal.submitFeatures(),
        ),
        const SizedBox(height: 20),

        // ── Multi-channel analyses ─────────────────────────────────────
        Text('Multi-Channel Analyses', style: theme.textTheme.labelLarge),
        const SizedBox(height: 4),
        if (signal.channelCount < 2)
          Padding(
            padding: const EdgeInsets.only(bottom: 8),
            child: Row(
              children: [
                const Icon(Icons.arrow_upward, size: 14, color: Colors.orange),
                const SizedBox(width: 4),
                Text(
                  'Import a 2nd channel above to enable these',
                  style: TextStyle(
                      fontSize: 12, color: Colors.orange.withValues(alpha: 0.8)),
                ),
              ],
            ),
          ),
        const SizedBox(height: 4),
        _AnalysisCard(
          title: 'Synchronisation Map',
          subtitle: '1:1 phase-locking detection from coupling',
          icon: Icons.lock_clock,
          busy: signal.isSubmittingSyncMap,
          canRun: signal.channelCount >= 2,
          unavailableReason:
              signal.channelCount < 2 ? 'Requires 2nd channel' : null,
          result: signal.syncMapResult,
          onRun: () => _showSyncMapDialog(context, signal),
        ),
        const SizedBox(height: 8),
        _AnalysisCard(
          title: 'Biphase Time Series',
          subtitle: 'Time-resolved biphase at a frequency pair',
          icon: Icons.timeline,
          busy: signal.isSubmittingBiphase,
          canRun: signal.channelCount >= 2,
          unavailableReason:
              signal.channelCount < 2 ? 'Requires 2nd channel' : null,
          result: signal.biphaseResult,
          onRun: () => _showBiphaseDialog(context, signal),
        ),
        const SizedBox(height: 8),
        _AnalysisCard(
          title: '4-Way Bispectrum',
          subtitle: 'b111 / b222 / b122 / b211 cross-bispectrum',
          icon: Icons.grid_view,
          busy: signal.isSubmittingBispec4,
          canRun: signal.channelCount >= 2,
          unavailableReason:
              signal.channelCount < 2 ? 'Requires 2nd channel' : null,
          result: signal.bispec4Result,
          onRun: () => signal.submitBispectrum4(),
        ),
        const SizedBox(height: 8),
        _AnalysisCard(
          title: 'Coupling Functions',
          subtitle: 'Directional q21/q12 via Fourier OLS',
          icon: Icons.swap_horiz,
          busy: signal.isSubmittingCoupling,
          canRun: signal.channelCount >= 2,
          unavailableReason:
              signal.channelCount < 2 ? 'Requires 2nd channel' : null,
          result: signal.couplingResult,
          onRun: () => _showCouplingDialog(context, signal),
        ),
        const SizedBox(height: 8),
        _AnalysisCard(
          title: 'Phase Coherence',
          subtitle: 'Multi-signal synchrony',
          icon: Icons.sync,
          busy: signal.isSubmittingCoherence,
          canRun: signal.channelCount >= 2,
          unavailableReason:
              signal.channelCount < 2 ? 'Requires 2nd channel' : null,
          result: signal.coherenceResult,
          onRun: () => signal.submitCoherence(
              channelBytes: List.generate(
                  signal.channelCount, signal.bytesForChannel)),
        ),
        const SizedBox(height: 8),
        _AnalysisCard(
          title: 'Bayesian Inference',
          subtitle: 'Directional phase coupling',
          icon: Icons.psychology,
          busy: signal.isSubmittingBayesian,
          canRun: signal.channelCount >= 2,
          unavailableReason:
              signal.channelCount < 2 ? 'Requires 2nd channel' : null,
          result: signal.bayesianResult,
          onRun: () => signal.submitBayesian(
              ch1Bytes: signal.bytesForChannel(0),
              ch2Bytes: signal.bytesForChannel(1)),
        ),
        const SizedBox(height: 8),
        _AnalysisCard(
          title: 'Group Comparison',
          subtitle: 'Wilcoxon rank-sum of mean wavelet power',
          icon: Icons.compare_arrows,
          busy: signal.isSubmittingGroup,
          canRun: signal.channelCount >= 4,
          unavailableReason: signal.channelCount < 4
              ? 'Import ≥ 4 channels (2 per group)'
              : null,
          result: signal.groupResult,
          onRun: () => _showGroupDialog(context, signal),
        ),

        // ── Export ─────────────────────────────────────────────────────
        if (anyResult) ...[
          const SizedBox(height: 20),
          Text('Export', style: theme.textTheme.labelLarge),
          const SizedBox(height: 8),
          Row(
            children: [
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: signal.hasData
                      ? () => exportSignalCsv(
                          signal.recentSamples, signal.sampleRate)
                      : null,
                  icon: const Icon(Icons.table_chart, size: 18),
                  label: const Text('Signal CSV'),
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: signal.lastResult != null
                      ? () => exportResultJson(signal.lastResult!)
                      : null,
                  icon: const Icon(Icons.data_object, size: 18),
                  label: const Text('Result JSON'),
                ),
              ),
            ],
          ),
        ],
      ],
    );
  }
}

// Renders displayable scalar/short-string fields from a result map,
// skipping large Plotly JSON blobs.
class _SummaryCard extends StatelessWidget {
  final Map<String, dynamic> result;
  const _SummaryCard({required this.result});

  static bool _displayable(dynamic v) {
    if (v is num || v is bool) return true;
    if (v is String && v.length < 80 && !v.startsWith('{') && !v.startsWith('[')) {
      return true;
    }
    return false;
  }

  @override
  Widget build(BuildContext context) {
    final entries = result.entries
        .where((e) => e.key != 'status' && _displayable(e.value))
        .take(8)
        .toList();

    if (entries.isEmpty) {
      return const Padding(
        padding: EdgeInsets.symmetric(vertical: 4),
        child: Text('Analysis complete',
            style: TextStyle(fontSize: 12, color: Colors.white54)),
      );
    }

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: entries
              .map((e) => Padding(
                    padding: const EdgeInsets.symmetric(vertical: 2),
                    child: Row(
                      children: [
                        Text('${e.key}: ',
                            style: const TextStyle(
                                fontSize: 11, color: Colors.white38)),
                        Expanded(
                          child: Text(
                            '${e.value}',
                            style: const TextStyle(fontSize: 11),
                            overflow: TextOverflow.ellipsis,
                          ),
                        ),
                      ],
                    ),
                  ))
              .toList(),
        ),
      ),
    );
  }
}

class _AnalysisCard extends StatelessWidget {
  final String title;
  final String subtitle;
  final IconData icon;
  final bool busy;
  final bool canRun;
  final String? unavailableReason;
  final Map<String, dynamic>? result;
  final VoidCallback onRun;

  const _AnalysisCard({
    required this.title,
    required this.subtitle,
    required this.icon,
    required this.busy,
    required this.canRun,
    required this.result,
    required this.onRun,
    this.unavailableReason,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Card(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          ListTile(
            leading: busy
                ? SizedBox(
                    width: 22,
                    height: 22,
                    child: CircularProgressIndicator(
                        strokeWidth: 2,
                        color: theme.colorScheme.primary))
                : Icon(icon, color: theme.colorScheme.primary),
            title: Text(title),
            subtitle: Text(
              unavailableReason ?? subtitle,
              style: TextStyle(
                  fontSize: 12,
                  color: unavailableReason != null
                      ? Colors.orange.withValues(alpha: 0.8)
                      : Colors.white38),
            ),
            trailing: canRun
                ? IconButton(
                    icon: const Icon(Icons.play_arrow),
                    onPressed: onRun,
                    tooltip: 'Run $title',
                  )
                : null,
          ),
          if (result != null)
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 0, 16, 12),
              child: _SummaryCard(result: result!),
            ),
        ],
      ),
    );
  }
}

// ── Helper functions for _ServerTab ──────────────────────────────────────────

List<dynamic>? _getFrequencySummary(Map<String, dynamic> result) {
  final r = result['results'];
  if (r is Map) return r['frequency_summary'] as List?;
  return null;
}

Map<String, dynamic>? _getSurrogateStats(Map<String, dynamic> result) {
  final r = result['results'];
  if (r is Map) {
    final s = r['surrogate_stats'];
    if (s is Map && s['enabled'] == true) return Map<String, dynamic>.from(s);
  }
  return null;
}

Future<void> _showSurrogateDialog(
    BuildContext context, SignalService signal) async {
  String testType = 'spectral';
  String method = 'phase_randomization';
  int nSurr = 19;

  await showModalBottomSheet(
    context: context,
    isScrollControlled: true,
    backgroundColor: const Color(0xFF1E1E2E),
    shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20))),
    builder: (ctx) => StatefulBuilder(
      builder: (ctx, setState) => Padding(
        padding: const EdgeInsets.fromLTRB(24, 20, 24, 36),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('Surrogate Test Options',
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600)),
            const SizedBox(height: 16),
            const Text('Test type',
                style: TextStyle(fontSize: 12, color: Colors.white54)),
            const SizedBox(height: 6),
            DropdownButton<String>(
              value: testType,
              isExpanded: true,
              dropdownColor: const Color(0xFF2A2A3E),
              items: const [
                DropdownMenuItem(value: 'spectral', child: Text('Spectral peak')),
                DropdownMenuItem(
                    value: 'changepoints', child: Text('Changepoints')),
                DropdownMenuItem(
                    value: 'phase_coherence', child: Text('Phase coherence')),
                DropdownMenuItem(value: 'bispectrum', child: Text('Bispectrum')),
              ],
              onChanged: (v) => setState(() => testType = v!),
            ),
            const SizedBox(height: 12),
            const Text('Surrogate method',
                style: TextStyle(fontSize: 12, color: Colors.white54)),
            const SizedBox(height: 6),
            DropdownButton<String>(
              value: method,
              isExpanded: true,
              dropdownColor: const Color(0xFF2A2A3E),
              items: const [
                DropdownMenuItem(
                    value: 'phase_randomization',
                    child: Text('Phase randomization')),
                DropdownMenuItem(value: 'iaaft', child: Text('IAAFT')),
                DropdownMenuItem(value: 'bootstrap', child: Text('Bootstrap')),
                DropdownMenuItem(
                    value: 'time_shift', child: Text('Time shift')),
              ],
              onChanged: (v) => setState(() => method = v!),
            ),
            const SizedBox(height: 12),
            Text('Surrogates: $nSurr',
                style: const TextStyle(fontSize: 12, color: Colors.white54)),
            Slider(
              value: nSurr.toDouble(),
              min: 9,
              max: 99,
              divisions: 10,
              label: '$nSurr',
              onChanged: (v) => setState(() => nSurr = v.round()),
            ),
            const SizedBox(height: 16),
            FilledButton(
              onPressed: () {
                Navigator.pop(ctx);
                signal.submitSurrogates(
                    testType: testType,
                    nSurrogates: nSurr,
                    surrogateMethod: method);
              },
              style: FilledButton.styleFrom(minimumSize: const Size.fromHeight(48)),
              child: const Text('Run Surrogate Test'),
            ),
          ],
        ),
      ),
    ),
  );
}

Future<void> _showFilterDialog(
    BuildContext context, SignalService signal) async {
  double fLow = 8.0, fHigh = 12.0;
  int order = 4, detrend = 0;

  await showModalBottomSheet(
    context: context,
    isScrollControlled: true,
    backgroundColor: const Color(0xFF1E1E2E),
    shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20))),
    builder: (ctx) => StatefulBuilder(
      builder: (ctx, setState) => Padding(
        padding: const EdgeInsets.fromLTRB(24, 20, 24, 36),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('Butterworth Filter',
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600)),
            const SizedBox(height: 16),
            Row(children: [
              Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                const Text('Low cutoff (Hz)', style: TextStyle(fontSize: 12, color: Colors.white54)),
                const SizedBox(height: 4),
                TextField(
                  decoration: const InputDecoration(isDense: true),
                  keyboardType: TextInputType.number,
                  controller: TextEditingController(text: fLow.toStringAsFixed(1)),
                  onChanged: (v) => fLow = double.tryParse(v) ?? fLow,
                  style: const TextStyle(fontSize: 13),
                ),
              ])),
              const SizedBox(width: 16),
              Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                const Text('High cutoff (Hz)', style: TextStyle(fontSize: 12, color: Colors.white54)),
                const SizedBox(height: 4),
                TextField(
                  decoration: const InputDecoration(isDense: true),
                  keyboardType: TextInputType.number,
                  controller: TextEditingController(text: fHigh.toStringAsFixed(1)),
                  onChanged: (v) => fHigh = double.tryParse(v) ?? fHigh,
                  style: const TextStyle(fontSize: 13),
                ),
              ])),
            ]),
            const SizedBox(height: 12),
            Text('Filter order: $order', style: const TextStyle(fontSize: 12, color: Colors.white54)),
            Slider(value: order.toDouble(), min: 1, max: 8, divisions: 7, label: '$order',
                onChanged: (v) => setState(() => order = v.round())),
            const SizedBox(height: 4),
            Text('Detrend degree: $detrend', style: const TextStyle(fontSize: 12, color: Colors.white54)),
            Slider(value: detrend.toDouble(), min: 0, max: 3, divisions: 3, label: '$detrend',
                onChanged: (v) => setState(() => detrend = v.round())),
            const SizedBox(height: 16),
            FilledButton(
              onPressed: () {
                Navigator.pop(ctx);
                signal.submitFilterButter(
                    fLow: fLow, fHigh: fHigh,
                    order: order, detrendDegree: detrend);
              },
              style: FilledButton.styleFrom(minimumSize: const Size.fromHeight(48)),
              child: const Text('Apply Filter'),
            ),
          ],
        ),
      ),
    ),
  );
}

Future<void> _showBiphaseDialog(
    BuildContext context, SignalService signal) async {
  double f1 = 6.0, f2 = 10.0;
  String wavelet = 'lognorm';

  await showModalBottomSheet(
    context: context,
    isScrollControlled: true,
    backgroundColor: const Color(0xFF1E1E2E),
    shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20))),
    builder: (ctx) => StatefulBuilder(
      builder: (ctx, setState) => Padding(
        padding: const EdgeInsets.fromLTRB(24, 20, 24, 36),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('Biphase Time Series',
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600)),
            const SizedBox(height: 16),
            Row(children: [
              Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                const Text('f1 (Hz)', style: TextStyle(fontSize: 12, color: Colors.white54)),
                const SizedBox(height: 4),
                TextField(
                  decoration: const InputDecoration(isDense: true),
                  keyboardType: TextInputType.number,
                  controller: TextEditingController(text: f1.toStringAsFixed(1)),
                  onChanged: (v) => f1 = double.tryParse(v) ?? f1,
                  style: const TextStyle(fontSize: 13),
                ),
              ])),
              const SizedBox(width: 16),
              Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                const Text('f2 (Hz)', style: TextStyle(fontSize: 12, color: Colors.white54)),
                const SizedBox(height: 4),
                TextField(
                  decoration: const InputDecoration(isDense: true),
                  keyboardType: TextInputType.number,
                  controller: TextEditingController(text: f2.toStringAsFixed(1)),
                  onChanged: (v) => f2 = double.tryParse(v) ?? f2,
                  style: const TextStyle(fontSize: 13),
                ),
              ])),
            ]),
            const SizedBox(height: 12),
            const Text('Wavelet', style: TextStyle(fontSize: 12, color: Colors.white54)),
            const SizedBox(height: 6),
            DropdownButton<String>(
              value: wavelet,
              isExpanded: true,
              dropdownColor: const Color(0xFF2A2A3E),
              items: const [
                DropdownMenuItem(value: 'lognorm', child: Text('Log-normal')),
                DropdownMenuItem(value: 'morlet', child: Text('Morlet')),
              ],
              onChanged: (v) => setState(() => wavelet = v!),
            ),
            const SizedBox(height: 16),
            FilledButton(
              onPressed: () {
                Navigator.pop(ctx);
                signal.submitBiphase(f1: f1, f2: f2, wavelet: wavelet);
              },
              style: FilledButton.styleFrom(minimumSize: const Size.fromHeight(48)),
              child: const Text('Run Biphase'),
            ),
          ],
        ),
      ),
    ),
  );
}

Future<void> _showSyncMapDialog(
    BuildContext context, SignalService signal) async {
  double b1Low = 8.0, b1High = 12.0, b2Low = 8.0, b2High = 12.0;

  await showModalBottomSheet(
    context: context,
    isScrollControlled: true,
    backgroundColor: const Color(0xFF1E1E2E),
    shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20))),
    builder: (ctx) => Padding(
      padding: const EdgeInsets.fromLTRB(24, 20, 24, 36),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('Synchronisation Map',
              style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600)),
          const SizedBox(height: 16),
          const Text('Channel 1 band (Hz)',
              style: TextStyle(fontSize: 12, color: Colors.white54)),
          const SizedBox(height: 6),
          Row(children: [
            Expanded(child: TextField(
              decoration: const InputDecoration(labelText: 'Low', isDense: true),
              keyboardType: TextInputType.number,
              controller: TextEditingController(text: b1Low.toStringAsFixed(1)),
              onChanged: (v) => b1Low = double.tryParse(v) ?? b1Low,
              style: const TextStyle(fontSize: 13),
            )),
            const SizedBox(width: 12),
            Expanded(child: TextField(
              decoration: const InputDecoration(labelText: 'High', isDense: true),
              keyboardType: TextInputType.number,
              controller: TextEditingController(text: b1High.toStringAsFixed(1)),
              onChanged: (v) => b1High = double.tryParse(v) ?? b1High,
              style: const TextStyle(fontSize: 13),
            )),
          ]),
          const SizedBox(height: 12),
          const Text('Channel 2 band (Hz)',
              style: TextStyle(fontSize: 12, color: Colors.white54)),
          const SizedBox(height: 6),
          Row(children: [
            Expanded(child: TextField(
              decoration: const InputDecoration(labelText: 'Low', isDense: true),
              keyboardType: TextInputType.number,
              controller: TextEditingController(text: b2Low.toStringAsFixed(1)),
              onChanged: (v) => b2Low = double.tryParse(v) ?? b2Low,
              style: const TextStyle(fontSize: 13),
            )),
            const SizedBox(width: 12),
            Expanded(child: TextField(
              decoration: const InputDecoration(labelText: 'High', isDense: true),
              keyboardType: TextInputType.number,
              controller: TextEditingController(text: b2High.toStringAsFixed(1)),
              onChanged: (v) => b2High = double.tryParse(v) ?? b2High,
              style: const TextStyle(fontSize: 13),
            )),
          ]),
          const SizedBox(height: 20),
          FilledButton(
            onPressed: () {
              Navigator.pop(ctx);
              signal.submitSyncMap(
                  band1Low: b1Low, band1High: b1High,
                  band2Low: b2Low, band2High: b2High);
            },
            style: FilledButton.styleFrom(minimumSize: const Size.fromHeight(48)),
            child: const Text('Run Sync Map'),
          ),
        ],
      ),
    ),
  );
}

Future<void> _showCouplingDialog(
    BuildContext context, SignalService signal) async {
  double b1Low = 8.0, b1High = 12.0, b2Low = 8.0, b2High = 12.0;
  int bn = 2;
  double winS = 1.0;

  await showModalBottomSheet(
    context: context,
    isScrollControlled: true,
    backgroundColor: const Color(0xFF1E1E2E),
    shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20))),
    builder: (ctx) => StatefulBuilder(
      builder: (ctx, setState) => Padding(
        padding: const EdgeInsets.fromLTRB(24, 20, 24, 36),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('Coupling Functions',
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600)),
            const SizedBox(height: 16),
            const Text('Channel 1 band (Hz)',
                style: TextStyle(fontSize: 12, color: Colors.white54)),
            const SizedBox(height: 6),
            Row(children: [
              Expanded(child: TextField(
                decoration: const InputDecoration(labelText: 'Low', isDense: true),
                keyboardType: TextInputType.number,
                controller: TextEditingController(text: b1Low.toStringAsFixed(1)),
                onChanged: (v) => b1Low = double.tryParse(v) ?? b1Low,
                style: const TextStyle(fontSize: 13),
              )),
              const SizedBox(width: 12),
              Expanded(child: TextField(
                decoration: const InputDecoration(labelText: 'High', isDense: true),
                keyboardType: TextInputType.number,
                controller: TextEditingController(text: b1High.toStringAsFixed(1)),
                onChanged: (v) => b1High = double.tryParse(v) ?? b1High,
                style: const TextStyle(fontSize: 13),
              )),
            ]),
            const SizedBox(height: 12),
            const Text('Channel 2 band (Hz)',
                style: TextStyle(fontSize: 12, color: Colors.white54)),
            const SizedBox(height: 6),
            Row(children: [
              Expanded(child: TextField(
                decoration: const InputDecoration(labelText: 'Low', isDense: true),
                keyboardType: TextInputType.number,
                controller: TextEditingController(text: b2Low.toStringAsFixed(1)),
                onChanged: (v) => b2Low = double.tryParse(v) ?? b2Low,
                style: const TextStyle(fontSize: 13),
              )),
              const SizedBox(width: 12),
              Expanded(child: TextField(
                decoration: const InputDecoration(labelText: 'High', isDense: true),
                keyboardType: TextInputType.number,
                controller: TextEditingController(text: b2High.toStringAsFixed(1)),
                onChanged: (v) => b2High = double.tryParse(v) ?? b2High,
                style: const TextStyle(fontSize: 13),
              )),
            ]),
            const SizedBox(height: 12),
            Text('Fourier terms (bn): $bn',
                style: const TextStyle(fontSize: 12, color: Colors.white54)),
            Slider(
              value: bn.toDouble(), min: 1, max: 5, divisions: 4, label: '$bn',
              onChanged: (v) => setState(() => bn = v.round()),
            ),
            const SizedBox(height: 4),
            Text('Window size: ${winS.toStringAsFixed(1)} s',
                style: const TextStyle(fontSize: 12, color: Colors.white54)),
            Slider(
              value: winS, min: 0.5, max: 5.0, divisions: 9,
              label: '${winS.toStringAsFixed(1)} s',
              onChanged: (v) => setState(() => winS = v),
            ),
            const SizedBox(height: 16),
            FilledButton(
              onPressed: () {
                Navigator.pop(ctx);
                signal.submitCoupling(
                    band1Low: b1Low, band1High: b1High,
                    band2Low: b2Low, band2High: b2High,
                    bn: bn, winS: winS);
              },
              style: FilledButton.styleFrom(minimumSize: const Size.fromHeight(48)),
              child: const Text('Run Coupling Functions'),
            ),
          ],
        ),
      ),
    ),
  );
}

// ── History tab ───────────────────────────────────────────────────────────────

class _HistoryTab extends StatelessWidget {
  const _HistoryTab();

  @override
  Widget build(BuildContext context) {
    final history = context.watch<AnalysisHistoryService>();
    final records = history.records;

    if (records.isEmpty) {
      return const Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.history, size: 64, color: Colors.white24),
            SizedBox(height: 16),
            Text('No recorded sessions yet',
                style: TextStyle(color: Colors.white54)),
            SizedBox(height: 6),
            Text('Results are saved automatically after each analysis',
                style: TextStyle(color: Colors.white38, fontSize: 12)),
          ],
        ),
      );
    }

    return Column(
      children: [
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
          child: Row(
            children: [
              Text('${records.length} sessions',
                  style: const TextStyle(fontSize: 12, color: Colors.white38)),
              const Spacer(),
              TextButton(
                onPressed: () => _confirmClearHistory(context, history),
                child: const Text('Clear all',
                    style: TextStyle(fontSize: 12, color: Colors.red)),
              ),
            ],
          ),
        ),
        Expanded(
          child: ListView.builder(
            padding: const EdgeInsets.symmetric(horizontal: 12),
            itemCount: records.length,
            itemBuilder: (_, i) => _HistoryTile(record: records[i]),
          ),
        ),
      ],
    );
  }

  void _confirmClearHistory(
      BuildContext context, AnalysisHistoryService history) {
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Clear all history?'),
        content: const Text(
            'This permanently removes all recorded analysis sessions.'),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(ctx),
              child: const Text('Cancel')),
          TextButton(
              onPressed: () {
                Navigator.pop(ctx);
                history.clearAll();
              },
              child: const Text('Clear all',
                  style: TextStyle(color: Colors.red))),
        ],
      ),
    );
  }
}

String _fmtTimestamp(DateTime ts) {
  final d = ts.day.toString().padLeft(2, '0');
  final mo = ts.month.toString().padLeft(2, '0');
  final h = ts.hour.toString().padLeft(2, '0');
  final mi = ts.minute.toString().padLeft(2, '0');
  return '$d/$mo ${ts.year} $h:$mi';
}

class _HistoryTile extends StatelessWidget {
  final AnalysisRecord record;
  const _HistoryTile({required this.record});

  @override
  Widget build(BuildContext context) {
    final history = context.read<AnalysisHistoryService>();
    final label = _fmtTimestamp(record.timestamp);

    return Card(
      child: ListTile(
        leading: _typeIcon(context, record.analysisType),
        title: Text(record.analysisType.toUpperCase(),
            style:
                const TextStyle(fontSize: 13, fontWeight: FontWeight.w600)),
        subtitle: Text(
          '$label · ${record.samplingRate.toStringAsFixed(0)} Hz'
          '${record.gpuUsed ? ' · GPU' : ''}',
          style: const TextStyle(fontSize: 11, color: Colors.white38),
        ),
        trailing: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            if (record.scalars.isNotEmpty)
              IconButton(
                icon: const Icon(Icons.data_object, size: 18),
                tooltip: 'Export JSON',
                onPressed: () => exportResultJson(record.scalars),
              ),
            IconButton(
              icon: const Icon(Icons.delete_outline,
                  size: 18, color: Colors.red),
              onPressed: () => history.delete(record.id!),
            ),
          ],
        ),
        onTap: () => _showDetail(context, record),
      ),
    );
  }

  static Widget _typeIcon(BuildContext context, String type) {
    const icons = {
      'spectral':    Icons.analytics,
      'modwt':       Icons.waves,
      'stft':        Icons.bar_chart,
      'cwt':         Icons.water,
      'hilbert':     Icons.rotate_right,
      'bispectrum':  Icons.grid_4x4,
      'bispectrum4': Icons.grid_view,
      'coherence':   Icons.sync,
      'bayesian':    Icons.psychology,
      'surrogates':  Icons.science,
      'features':    Icons.table_rows,
      'syncmap':     Icons.lock_clock,
      'biphase':     Icons.timeline,
      'coupling':    Icons.swap_horiz,
      'ridge':       Icons.show_chart,
      'filter':      Icons.filter_alt,
      'wft':         Icons.grain,
    };
    return Icon(icons[type] ?? Icons.insert_chart,
        color: Theme.of(context).colorScheme.primary);
  }

  void _showDetail(BuildContext ctx, AnalysisRecord r) {
    showModalBottomSheet(
      context: ctx,
      backgroundColor: const Color(0xFF1E1E2E),
      shape: const RoundedRectangleBorder(
          borderRadius: BorderRadius.vertical(top: Radius.circular(20))),
      builder: (_) => _HistoryDetailSheet(record: r),
    );
  }
}

class _HistoryDetailSheet extends StatelessWidget {
  final AnalysisRecord record;
  const _HistoryDetailSheet({required this.record});

  @override
  Widget build(BuildContext context) {
    final scalars = record.scalars;
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 20, 20, 36),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(record.analysisType.toUpperCase(),
              style: const TextStyle(
                  fontSize: 16, fontWeight: FontWeight.w600)),
          Text(_fmtTimestamp(record.timestamp),
              style: const TextStyle(fontSize: 11, color: Colors.white38)),
          const SizedBox(height: 12),
          if (scalars.isNotEmpty) _SummaryCard(result: scalars),
          if (record.frequencySummary != null) ...[
            const SizedBox(height: 8),
            _FrequencySummaryCard(summary: record.frequencySummary!),
          ],
        ],
      ),
    );
  }
}

// ── Shared widgets ────────────────────────────────────────────────────────────

class _FrequencySummaryCard extends StatelessWidget {
  final List<dynamic> summary;
  const _FrequencySummaryCard({required this.summary});

  static const _bandColors = {
    'delta': Colors.purple,
    'theta': Colors.blue,
    'alpha': Colors.teal,
    'beta': Colors.orange,
    'gamma': Colors.red,
  };

  @override
  Widget build(BuildContext context) {
    if (summary.isEmpty) return const SizedBox.shrink();
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Top Frequencies',
                style: Theme.of(context)
                    .textTheme
                    .labelSmall
                    ?.copyWith(color: Theme.of(context).colorScheme.primary)),
            const SizedBox(height: 8),
            ...summary.take(5).map((item) {
              if (item is! Map) return const SizedBox.shrink();
              final hz = (item['frequency'] as num?)?.toStringAsFixed(1) ?? '—';
              final band = (item['band'] as String?) ?? '';
              final rank = item['rank'] ?? '—';
              final color = _bandColors[band] ?? Colors.white38;
              return Padding(
                padding: const EdgeInsets.symmetric(vertical: 3),
                child: Row(
                  children: [
                    SizedBox(
                      width: 20,
                      child: Text('#$rank',
                          style: const TextStyle(
                              fontSize: 10, color: Colors.white38)),
                    ),
                    Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 6, vertical: 2),
                      decoration: BoxDecoration(
                        color: color.withValues(alpha: 0.15),
                        border: Border.all(
                            color: color.withValues(alpha: 0.5)),
                        borderRadius: BorderRadius.circular(4),
                      ),
                      child: Text(band,
                          style: TextStyle(
                              fontSize: 9,
                              color: color,
                              fontWeight: FontWeight.w600)),
                    ),
                    const SizedBox(width: 8),
                    Text('$hz Hz',
                        style: const TextStyle(
                            fontSize: 12, fontWeight: FontWeight.w600)),
                    const Spacer(),
                    if (item['duration_pct'] != null)
                      Text(
                          '${(item['duration_pct'] as num).toStringAsFixed(0)}%',
                          style: const TextStyle(
                              fontSize: 11, color: Colors.white38)),
                  ],
                ),
              );
            }),
          ],
        ),
      ),
    );
  }
}

class _SurrogateStatsRow extends StatelessWidget {
  final Map<String, dynamic> stats;
  const _SurrogateStatsRow({required this.stats});

  @override
  Widget build(BuildContext context) {
    final p95 = stats['pct_significant_95'];
    final p99 = stats['pct_significant_99'];
    final method = stats['method'] ?? '';
    final n = stats['n_surrogates'] ?? '—';
    return Row(
      children: [
        Expanded(
          child: _MetricChip(
            label: '95% sig.',
            value: p95 != null ? '${(p95 as num).toStringAsFixed(1)}%' : '—',
            tooltip: '$n $method surrogates',
          ),
        ),
        const SizedBox(width: 8),
        Expanded(
          child: _MetricChip(
            label: '99% sig.',
            value: p99 != null ? '${(p99 as num).toStringAsFixed(1)}%' : '—',
            tooltip: 'Spectrum bins significant at 99% confidence',
          ),
        ),
      ],
    );
  }
}

class _ChannelImportRow extends StatelessWidget {
  final SignalService signal;
  const _ChannelImportRow({required this.signal});

  Future<void> _pickFile(BuildContext context) async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['csv', 'txt'],
      withData: true,
    );
    if (result == null || result.files.isEmpty) return;
    final bytes = result.files.first.bytes;
    if (bytes == null) return;

    // Parse CSV: one double per line
    final text = String.fromCharCodes(bytes);
    final samples = text
        .split(RegExp(r'[\r\n,]+'))
        .map((s) => double.tryParse(s.trim()))
        .whereType<double>()
        .toList();

    if (samples.isEmpty) {
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('No numeric data found in file')),
        );
      }
      return;
    }
    signal.addChannel(samples);
  }

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
        child: Row(
          children: [
            const Icon(Icons.layers, size: 18, color: Colors.white38),
            const SizedBox(width: 10),
            Text(
              'Channels: ${signal.channelCount}',
              style: const TextStyle(fontSize: 13),
            ),
            const Spacer(),
            if (signal.channelCount > 1)
              TextButton(
                onPressed: signal.clearExtraChannels,
                child: const Text('Clear',
                    style: TextStyle(fontSize: 12, color: Colors.red)),
              ),
            TextButton.icon(
              onPressed: () => _pickFile(context),
              icon: const Icon(Icons.upload_file, size: 16),
              label: const Text('Import CSV', style: TextStyle(fontSize: 12)),
            ),
          ],
        ),
      ),
    );
  }
}

Future<void> _showModwtDialog(
    BuildContext context, SignalService signal) async {
  String wavelet = 'la8';
  int level = 5;

  await showModalBottomSheet(
    context: context,
    isScrollControlled: true,
    backgroundColor: const Color(0xFF1E1E2E),
    shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20))),
    builder: (ctx) => StatefulBuilder(
      builder: (ctx, setState) => Padding(
        padding: const EdgeInsets.fromLTRB(24, 20, 24, 36),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('MODWT Parameters',
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600)),
            const SizedBox(height: 16),
            const Text('Wavelet', style: TextStyle(fontSize: 12, color: Colors.white54)),
            DropdownButton<String>(
              value: wavelet,
              isExpanded: true,
              items: const [
                DropdownMenuItem(value: 'la8',  child: Text('LA8 (least-asym 8)')),
                DropdownMenuItem(value: 'la16', child: Text('LA16 (least-asym 16)')),
                DropdownMenuItem(value: 'd4',   child: Text('D4 (Daubechies 4)')),
                DropdownMenuItem(value: 'd6',   child: Text('D6 (Daubechies 6)')),
              ],
              onChanged: (v) => setState(() => wavelet = v ?? wavelet),
            ),
            const SizedBox(height: 12),
            Text('Decomposition level: $level',
                style: const TextStyle(fontSize: 12, color: Colors.white54)),
            Slider(
              value: level.toDouble(), min: 2, max: 10, divisions: 8,
              label: '$level',
              onChanged: (v) => setState(() => level = v.round()),
            ),
            const SizedBox(height: 16),
            FilledButton(
              onPressed: () {
                Navigator.pop(ctx);
                signal.submitModwt(wavelet: wavelet, level: level);
              },
              style: FilledButton.styleFrom(minimumSize: const Size.fromHeight(48)),
              child: const Text('Run MODWT'),
            ),
          ],
        ),
      ),
    ),
  );
}

Future<void> _showGroupDialog(
    BuildContext context, SignalService signal) async {
  final total = signal.channelCount;
  // Default split: first half vs second half.
  final mid = (total / 2).floor();
  final g1 = <int>{for (int i = 0; i < mid; i++) i};
  final g2 = <int>{for (int i = mid; i < total; i++) i};
  double freqMin = 0.5;
  int nFreqs = 50;

  await showModalBottomSheet(
    context: context,
    isScrollControlled: true,
    backgroundColor: const Color(0xFF1E1E2E),
    shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20))),
    builder: (ctx) => StatefulBuilder(
      builder: (ctx, setState) => Padding(
        padding: const EdgeInsets.fromLTRB(24, 20, 24, 36),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('Group Comparison',
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600)),
            const SizedBox(height: 6),
            const Text(
              'Assign each channel to Group 1 or Group 2. Each group needs ≥ 2 channels.',
              style: TextStyle(fontSize: 12, color: Colors.white54),
            ),
            const SizedBox(height: 12),
            ConstrainedBox(
              constraints: const BoxConstraints(maxHeight: 220),
              child: SingleChildScrollView(
                child: Column(
                  children: List.generate(total, (i) {
                    final inG1 = g1.contains(i);
                    final inG2 = g2.contains(i);
                    return Row(
                      children: [
                        Expanded(child: Text('Ch $i',
                            style: const TextStyle(fontSize: 13))),
                        ChoiceChip(
                          label: const Text('G1'), selected: inG1,
                          onSelected: (sel) => setState(() {
                            if (sel) { g1.add(i); g2.remove(i); }
                            else { g1.remove(i); }
                          }),
                        ),
                        const SizedBox(width: 6),
                        ChoiceChip(
                          label: const Text('G2'), selected: inG2,
                          onSelected: (sel) => setState(() {
                            if (sel) { g2.add(i); g1.remove(i); }
                            else { g2.remove(i); }
                          }),
                        ),
                      ],
                    );
                  }),
                ),
              ),
            ),
            const SizedBox(height: 12),
            Row(children: [
              Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                const Text('Min freq (Hz)', style: TextStyle(fontSize: 12, color: Colors.white54)),
                TextField(
                  decoration: const InputDecoration(isDense: true),
                  keyboardType: TextInputType.number,
                  controller: TextEditingController(text: freqMin.toStringAsFixed(2)),
                  onChanged: (v) => freqMin = double.tryParse(v) ?? freqMin,
                  style: const TextStyle(fontSize: 13),
                ),
              ])),
              const SizedBox(width: 16),
              Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Text('Frequency bins: $nFreqs',
                    style: const TextStyle(fontSize: 12, color: Colors.white54)),
                Slider(
                  value: nFreqs.toDouble(), min: 10, max: 200, divisions: 19,
                  label: '$nFreqs',
                  onChanged: (v) => setState(() => nFreqs = v.round()),
                ),
              ])),
            ]),
            const SizedBox(height: 16),
            FilledButton(
              onPressed: (g1.length >= 2 && g2.length >= 2)
                  ? () {
                      Navigator.pop(ctx);
                      signal.submitGroupComparison(
                        group1Indices: g1.toList()..sort(),
                        group2Indices: g2.toList()..sort(),
                        freqMin: freqMin,
                        nFreqs: nFreqs,
                      );
                    }
                  : null,
              style: FilledButton.styleFrom(minimumSize: const Size.fromHeight(48)),
              child: const Text('Compare Groups'),
            ),
          ],
        ),
      ),
    ),
  );
}

class _MetricChip extends StatelessWidget {
  final String label;
  final String value;
  final String tooltip;
  const _MetricChip(
      {required this.label, required this.value, required this.tooltip});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Tooltip(
      message: tooltip,
      child: Card(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(label,
                  style: const TextStyle(fontSize: 12, color: Colors.white54)),
              Text(value,
                  style: TextStyle(
                      fontSize: 13,
                      fontWeight: FontWeight.w600,
                      color: theme.colorScheme.primary)),
            ],
          ),
        ),
      ),
    );
  }
}
