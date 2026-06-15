import 'package:flutter/material.dart';

import 'plotly_chart_widget.dart';

/// Scans an analysis result map for Plotly figure JSON produced by the
/// FastMODA backend and renders them — keys ending in `_plot` are a single
/// figure, keys ending in `_plots` are a named group of figures (e.g.
/// coherence's `pair_plots`, one per channel pair).
class ResultPlots extends StatelessWidget {
  final Map<String, dynamic> result;
  const ResultPlots({super.key, required this.result});

  static bool _looksLikeFigure(dynamic v) =>
      v is String && v.trimLeft().startsWith('{') && v.contains('"data"');

  static String _titleFromKey(String key) {
    var base = key;
    if (base.endsWith('_plots')) {
      base = base.substring(0, base.length - 6);
    } else if (base.endsWith('_plot')) {
      base = base.substring(0, base.length - 5);
    }
    const acronyms = {'cwt', 'stft', 'wft', 'modwt'};
    return base
        .split('_')
        .where((w) => w.isNotEmpty)
        .map((w) =>
            acronyms.contains(w) ? w.toUpperCase() : '${w[0].toUpperCase()}${w.substring(1)}')
        .join(' ');
  }

  @override
  Widget build(BuildContext context) {
    final singlePlots = <MapEntry<String, String>>[];
    final groupPlots = <MapEntry<String, Map<String, dynamic>>>[];

    result.forEach((key, value) {
      if (key.endsWith('_plot') && _looksLikeFigure(value)) {
        singlePlots.add(MapEntry(key, value as String));
      } else if (key.endsWith('_plots') && value is Map) {
        final group = Map<String, dynamic>.from(value);
        if (group.values.any(_looksLikeFigure)) {
          groupPlots.add(MapEntry(key, group));
        }
      }
    });

    if (singlePlots.isEmpty && groupPlots.isEmpty) {
      return const SizedBox.shrink();
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        for (final entry in singlePlots) ...[
          const SizedBox(height: 10),
          _PlotLabel(text: _titleFromKey(entry.key)),
          const SizedBox(height: 4),
          ClipRRect(
            borderRadius: BorderRadius.circular(8),
            child: PlotlyChartWidget(figureJson: entry.value),
          ),
        ],
        for (final entry in groupPlots) ...[
          const SizedBox(height: 10),
          _PairPlotGroup(title: _titleFromKey(entry.key), plots: entry.value),
        ],
      ],
    );
  }
}

class _PlotLabel extends StatelessWidget {
  final String text;
  const _PlotLabel({required this.text});

  @override
  Widget build(BuildContext context) {
    return Text(text,
        style: const TextStyle(
            fontSize: 12, fontWeight: FontWeight.w600, color: Colors.white70));
  }
}

/// A group of named figures (e.g. one per channel pair) with a chip selector.
class _PairPlotGroup extends StatefulWidget {
  final String title;
  final Map<String, dynamic> plots;
  const _PairPlotGroup({required this.title, required this.plots});

  @override
  State<_PairPlotGroup> createState() => _PairPlotGroupState();
}

class _PairPlotGroupState extends State<_PairPlotGroup> {
  late String _selected;

  @override
  void initState() {
    super.initState();
    _selected = widget.plots.keys.first;
  }

  @override
  Widget build(BuildContext context) {
    final keys = widget.plots.keys.toList();
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _PlotLabel(text: widget.title),
        if (keys.length > 1) ...[
          const SizedBox(height: 6),
          Wrap(
            spacing: 6,
            runSpacing: 4,
            children: keys
                .map((k) => ChoiceChip(
                      label: Text(k, style: const TextStyle(fontSize: 11)),
                      selected: _selected == k,
                      visualDensity: VisualDensity.compact,
                      onSelected: (_) => setState(() => _selected = k),
                    ))
                .toList(),
          ),
        ],
        const SizedBox(height: 4),
        ClipRRect(
          borderRadius: BorderRadius.circular(8),
          child: PlotlyChartWidget(figureJson: widget.plots[_selected] as String),
        ),
      ],
    );
  }
}
