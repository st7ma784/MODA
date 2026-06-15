import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:webview_flutter/webview_flutter.dart';

import '../theme/app_theme.dart';

/// Renders a Plotly figure — as produced by `plotly.utils.PlotlyJSONEncoder`
/// on the FastMODA backend — inside an embedded WebView, using the bundled
/// Plotly.js asset (`assets/plotly/plotly-cartesian.min.js`).
class PlotlyChartWidget extends StatefulWidget {
  final String figureJson;
  final double height;

  const PlotlyChartWidget({
    super.key,
    required this.figureJson,
    this.height = 280,
  });

  @override
  State<PlotlyChartWidget> createState() => _PlotlyChartWidgetState();
}

class _PlotlyChartWidgetState extends State<PlotlyChartWidget> {
  static final String _bg = _colorToHex(AppTheme.surface);

  late final WebViewController _controller;
  bool _pageReady = false;
  String? _error;

  static String _colorToHex(Color c) =>
      '#${c.toARGB32().toRadixString(16).padLeft(8, '0').substring(2)}';

  @override
  void initState() {
    super.initState();
    _controller = WebViewController()
      ..setJavaScriptMode(JavaScriptMode.unrestricted)
      ..setBackgroundColor(AppTheme.surface)
      ..setNavigationDelegate(NavigationDelegate(
        onPageFinished: (_) {
          _pageReady = true;
          _render();
        },
      ))
      ..loadFlutterAsset('assets/plotly/chart.html');
  }

  @override
  void didUpdateWidget(covariant PlotlyChartWidget oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.figureJson != widget.figureJson && _pageReady) {
      _render();
    }
  }

  Future<void> _render() async {
    try {
      final fig = jsonDecode(widget.figureJson) as Map<String, dynamic>;
      final data = fig['data'] ?? <dynamic>[];
      final layout = Map<String, dynamic>.from(fig['layout'] as Map? ?? {});

      // Dark-theme overrides so charts blend into the app's surface.
      layout['paper_bgcolor'] = _bg;
      layout['plot_bgcolor'] = _bg;
      layout['font'] = {'color': '#E6E1DE', 'size': 10};
      layout['margin'] ??= const {'t': 32, 'b': 36, 'l': 48, 'r': 16};

      final dataArg = jsonEncode(jsonEncode(data));
      final layoutArg = jsonEncode(jsonEncode(layout));
      await _controller.runJavaScript('renderPlot($dataArg, $layoutArg)');
      if (mounted) setState(() {});
    } catch (e) {
      if (mounted) setState(() => _error = e.toString());
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_error != null) {
      return SizedBox(
        height: widget.height,
        child: Center(
          child: Text('Chart error: $_error',
              style: const TextStyle(fontSize: 11, color: Colors.redAccent)),
        ),
      );
    }
    return SizedBox(
      height: widget.height,
      child: Stack(
        children: [
          WebViewWidget(controller: _controller),
          if (!_pageReady)
            const Center(
              child: SizedBox(
                width: 22,
                height: 22,
                child: CircularProgressIndicator(strokeWidth: 2),
              ),
            ),
        ],
      ),
    );
  }
}
