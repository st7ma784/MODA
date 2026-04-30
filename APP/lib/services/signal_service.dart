import 'dart:async';
import 'dart:math' as math;
import 'package:flutter/foundation.dart';
import 'fastmoda_client.dart';
import '../utils/dft.dart';
import '../utils/npy.dart';

enum ServerStatus { unknown, checking, up, down }

class SignalService extends ChangeNotifier {
  static const int _bufferSize = 512;
  static const int _dftSize = 256;

  final _buf = List<double>.filled(_bufferSize, 0.0, growable: false);
  int _head = 0;
  int _total = 0;
  double _sampleRate = 256.0;

  List<double> _spectrum = List.filled(_dftSize ~/ 2, 0.0);
  final _bandPowers = <String, double>{
    'delta': 0, 'theta': 0, 'alpha': 0, 'beta': 0, 'gamma': 0,
  };
  double _dominantFreq = 0.0;
  double _signalQuality = 0.0;

  ServerStatus _serverStatus = ServerStatus.unknown;
  FastModaClient? _client;
  Timer? _healthTimer;
  bool _submitting = false;
  String? _pendingTaskId;
  Map<String, dynamic>? _lastResult;
  bool _dftPending = false;

  StreamSubscription<List<double>>? _bleSub;
  final _errorController = StreamController<String>.broadcast();

  // ── Public getters ──────────────────────────────────────────────────────────

  ServerStatus get serverStatus => _serverStatus;
  Map<String, double> get bandPowers => Map.unmodifiable(_bandPowers);
  double get dominantFreq => _dominantFreq;
  double get signalQuality => _signalQuality;
  double get sampleRate => _sampleRate;
  List<double> get spectrum => List.unmodifiable(_spectrum);
  Map<String, dynamic>? get lastResult => _lastResult;
  bool get isSubmitting => _submitting;
  String? get pendingTaskId => _pendingTaskId;
  bool get hasData => _total >= 64;

  /// Errors suitable for display as snackbars. Broadcast; subscribe once.
  Stream<String> get errors => _errorController.stream;

  set sampleRate(double fs) {
    _sampleRate = fs;
    notifyListeners();
  }

  List<double> get recentSamples {
    final count = math.min(_total, _bufferSize);
    if (_total < _bufferSize) return List<double>.from(_buf.sublist(0, count));
    final pos = _head % _bufferSize;
    return [..._buf.sublist(pos), ..._buf.sublist(0, pos)];
  }

  // ── Signal ingestion ────────────────────────────────────────────────────────

  void addSamples(List<double> values) {
    for (final v in values) {
      _buf[_head % _bufferSize] = v;
      _head++;
    }
    _total += values.length;
    notifyListeners();
    if (!_dftPending && _total >= _dftSize) _scheduleRecompute();
  }

  void _scheduleRecompute() async {
    _dftPending = true;
    final samples = recentSamples;
    final n = math.min(samples.length, _dftSize);
    final window = samples.length > n
        ? samples.sublist(samples.length - n)
        : List<double>.from(samples);

    // Run DFT in a separate isolate so the UI thread stays smooth.
    final result = await compute(dftWorker, {'data': window, 'fs': _sampleRate});

    _spectrum = List<double>.from(result['mags'] as List);
    _bandPowers['delta'] = (result['delta'] as num).toDouble();
    _bandPowers['theta'] = (result['theta'] as num).toDouble();
    _bandPowers['alpha'] = (result['alpha'] as num).toDouble();
    _bandPowers['beta'] = (result['beta'] as num).toDouble();
    _bandPowers['gamma'] = (result['gamma'] as num).toDouble();
    _dominantFreq = (result['dominant'] as num).toDouble();
    _signalQuality = (result['quality'] as num).toDouble();
    _dftPending = false;
    if (hasListeners) notifyListeners();
  }

  // ── Service wiring ──────────────────────────────────────────────────────────

  void bindBleStream(Stream<List<double>> stream) {
    _bleSub?.cancel();
    _bleSub = stream.listen(addSamples);
  }

  void bindClient(FastModaClient client) {
    _client = client;
    _serverStatus = ServerStatus.checking;
    notifyListeners();
    _pollHealth();
    _healthTimer?.cancel();
    _healthTimer =
        Timer.periodic(const Duration(seconds: 30), (_) => _pollHealth());
  }

  Future<void> forceHealthCheck() => _pollHealth();

  Future<void> _pollHealth() async {
    if (_client == null) return;
    try {
      await _client!.checkHealth();
      if (_serverStatus != ServerStatus.up) {
        _serverStatus = ServerStatus.up;
        if (hasListeners) notifyListeners();
      }
    } catch (e) {
      if (_serverStatus != ServerStatus.down) {
        _serverStatus = ServerStatus.down;
        _errorController.add('Server unreachable — ${_friendly(e)}');
        if (hasListeners) notifyListeners();
      }
    }
  }

  // ── Analysis submission ─────────────────────────────────────────────────────

  Future<void> submitAnalysis() async {
    if (_client == null || _submitting || !hasData) return;
    _submitting = true;
    notifyListeners();
    try {
      final bytes = packNpy(recentSamples);
      final taskId = await _client!.submitAnalysis(
        signalBytes: bytes,
        samplingRate: _sampleRate,
      );
      _pendingTaskId = taskId;
      notifyListeners();
      await _awaitTask(taskId);
    } catch (e) {
      _errorController.add('Analysis submission failed — ${_friendly(e)}');
    } finally {
      _submitting = false;
      _pendingTaskId = null;
      if (hasListeners) notifyListeners();
    }
  }

  Future<void> _awaitTask(String taskId) async {
    for (int i = 0; i < 60; i++) {
      await Future.delayed(const Duration(seconds: 2));
      try {
        final status = await _client!.pollStatus(taskId);
        final s = status['status'] as String?;
        if (s == 'done' || s == 'complete' || s == 'success') {
          _lastResult = status;
          return;
        }
        if (s == 'error' || s == 'failed') {
          _errorController.add('Server analysis returned an error.');
          return;
        }
      } catch (_) {}
    }
    _errorController.add('Analysis timed out after 2 minutes.');
  }

  // ── Helpers ─────────────────────────────────────────────────────────────────

  static String _friendly(Object e) {
    final msg = e.toString();
    if (msg.contains('SocketException') || msg.contains('Connection refused')) {
      return 'server not reachable';
    }
    if (msg.contains('401') || msg.contains('403')) return 'authentication error';
    if (msg.contains('timeout')) return 'request timed out';
    return msg.split('\n').first;
  }

  @override
  void dispose() {
    _healthTimer?.cancel();
    _bleSub?.cancel();
    _errorController.close();
    super.dispose();
  }
}
