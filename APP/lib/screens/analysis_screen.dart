import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../services/signal_service.dart';
import '../widgets/signal_chart_widget.dart';

class AnalysisScreen extends StatefulWidget {
  const AnalysisScreen({super.key});

  @override
  State<AnalysisScreen> createState() => _AnalysisScreenState();
}

class _AnalysisScreenState extends State<AnalysisScreen>
    with SingleTickerProviderStateMixin {
  late final TabController _tabs;

  @override
  void initState() {
    super.initState();
    _tabs = TabController(length: 4, vsync: this);
  }

  @override
  void dispose() {
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
    final result = signal.lastResult;
    final changepoints = result?['changepoints'] as List?;

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
            if (changepoints != null && changepoints.isNotEmpty)
              Wrap(
                spacing: 6,
                runSpacing: 6,
                children: changepoints
                    .map((cp) => Chip(
                          label: Text('${cp}s',
                              style: const TextStyle(fontSize: 11)),
                          padding: EdgeInsets.zero,
                          visualDensity: VisualDensity.compact,
                        ))
                    .toList(),
              )
            else
              Container(
                height: 50,
                decoration: BoxDecoration(
                  color: Colors.white.withOpacity(0.04),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: const Center(
                  child: Text('Run server analysis to detect changepoints',
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
        Text('Band Power Breakdown', style: theme.textTheme.labelLarge),
        const SizedBox(height: 8),
        ...[
          ('Delta', '0.5–4 Hz', Colors.purple, 'delta'),
          ('Theta', '4–8 Hz', Colors.blue, 'theta'),
          ('Alpha', '8–12 Hz', Colors.teal, 'alpha'),
          ('Beta', '12–30 Hz', Colors.orange, 'beta'),
          ('Gamma', '30–100 Hz', Colors.red, 'gamma'),
        ].map((e) {
          final (name, hz, color, key) = e;
          return Padding(
            padding: const EdgeInsets.only(bottom: 8),
            child: _BandRow(
                name: name, hz: hz, color: color, power: norm(key)),
          );
        }),
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
    final result = signal.lastResult;

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

    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        Card(
          child: ListTile(
            leading: const Icon(Icons.check_circle, color: Colors.green),
            title: const Text('Server connected'),
            subtitle: const Text('Advanced analysis available'),
            trailing: signal.serverStatus == ServerStatus.checking
                ? const SizedBox(
                    width: 18,
                    height: 18,
                    child: CircularProgressIndicator(strokeWidth: 2))
                : null,
          ),
        ),
        const SizedBox(height: 16),
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
        if (result != null) ...[
          const SizedBox(height: 16),
          Text('Last Result', style: theme.textTheme.labelLarge),
          const SizedBox(height: 8),
          _ResultCard(result: result),
        ],
        const SizedBox(height: 16),
        Text('Available Analyses', style: theme.textTheme.labelLarge),
        const SizedBox(height: 8),
        const _ServerAnalysisCard(
            title: 'Full MODWT',
            subtitle: 'All decomposition levels',
            icon: Icons.waves),
        const SizedBox(height: 8),
        const _ServerAnalysisCard(
            title: 'Phase Coherence',
            subtitle: 'Multi-signal synchrony',
            icon: Icons.sync),
        const SizedBox(height: 8),
        const _ServerAnalysisCard(
            title: 'Bispectrum',
            subtitle: 'Quadratic phase coupling',
            icon: Icons.grid_4x4),
        const SizedBox(height: 8),
        const _ServerAnalysisCard(
            title: 'Bayesian Inference',
            subtitle: 'Directional coupling',
            icon: Icons.psychology),
      ],
    );
  }
}

class _ResultCard extends StatelessWidget {
  final Map<String, dynamic> result;
  const _ResultCard({required this.result});

  @override
  Widget build(BuildContext context) {
    final entries = result.entries
        .where((e) => e.key != 'status')
        .take(6)
        .toList();

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: entries
              .map((e) => Padding(
                    padding: const EdgeInsets.symmetric(vertical: 3),
                    child: Row(
                      children: [
                        Text('${e.key}:',
                            style: const TextStyle(
                                fontSize: 12, color: Colors.white54)),
                        const SizedBox(width: 8),
                        Expanded(
                          child: Text(
                            '${e.value}',
                            style: const TextStyle(fontSize: 12),
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

class _ServerAnalysisCard extends StatelessWidget {
  final String title;
  final String subtitle;
  final IconData icon;

  const _ServerAnalysisCard(
      {required this.title, required this.subtitle, required this.icon});

  @override
  Widget build(BuildContext context) {
    return Card(
      child: ListTile(
        leading: Icon(icon, color: Theme.of(context).colorScheme.primary),
        title: Text(title),
        subtitle: Text(subtitle,
            style: const TextStyle(fontSize: 12, color: Colors.white38)),
        trailing: const Icon(Icons.chevron_right),
        onTap: () {},
      ),
    );
  }
}

class _HistoryTab extends StatelessWidget {
  const _HistoryTab();

  @override
  Widget build(BuildContext context) {
    return const Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(Icons.history, size: 64, color: Colors.white24),
          SizedBox(height: 16),
          Text('No recorded sessions yet',
              style: TextStyle(color: Colors.white54)),
        ],
      ),
    );
  }
}
