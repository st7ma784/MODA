import 'dart:async';
import 'dart:math' as math;
import 'package:flutter/foundation.dart';
import 'fastmoda_client.dart';
import '../utils/changepoint.dart';
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

  // Cached snapshot rebuilt in addSamples — avoids allocating a new list on every getter call.
  List<double> _recentSamplesSnapshot = const [];

  List<double> _spectrum = List.filled(_dftSize ~/ 2, 0.0);
  final _bandPowers = <String, double>{
    'delta': 0, 'theta': 0, 'alpha': 0, 'beta': 0, 'gamma': 0,
  };

  // Cached unmodifiable views — recreated only when the underlying data changes.
  Map<String, double>? _bandPowersCache;
  List<double>? _spectrumCache;
  List<int>? _changepointsCache;

  // Per-channel NPY cache — invalidated when samples or channel list changes.
  final Map<int, List<int>> _npyCache = {};
  double _dominantFreq = 0.0;
  double _signalQuality = 0.0;
  double _spectralEntropy = 0.0;
  double _spectralFlatness = 0.0;

  ServerStatus _serverStatus = ServerStatus.unknown;
  FastModaClient? _client;
  Timer? _healthTimer;
  bool _submitting = false;
  String? _pendingTaskId;
  Map<String, dynamic>? _lastResult;
  bool _dftPending = false;
  bool _changepointPending = false;
  int _lastChangepointTotal = 0;
  List<int> _changepoints = [];

  bool _submittingSyncMap = false;
  Map<String, dynamic>? _syncMapResult;

  bool _submittingBiphase = false;
  bool _submittingBispec4 = false;
  bool _submittingCoupling = false;
  Map<String, dynamic>? _biphaseResult;
  Map<String, dynamic>? _bispec4Result;
  Map<String, dynamic>? _couplingResult;

  bool _submittingRidge = false;
  bool _submittingFilter = false;
  bool _submittingWft = false;
  bool _submittingModwt = false;
  bool _submittingGroup = false;
  Map<String, dynamic>? _ridgeResult;
  Map<String, dynamic>? _filterResult;
  Map<String, dynamic>? _wftResult;
  Map<String, dynamic>? _modwtResult;
  Map<String, dynamic>? _groupResult;

  bool _submittingBispectrum = false;
  bool _submittingCoherence = false;
  bool _submittingBayesian = false;
  bool _submittingStft = false;
  bool _submittingCwt = false;
  bool _submittingHilbert = false;
  bool _submittingSurrogates = false;
  bool _submittingFeatures = false;
  Map<String, dynamic>? _bispectrumResult;
  Map<String, dynamic>? _coherenceResult;
  Map<String, dynamic>? _bayesianResult;
  Map<String, dynamic>? _stftResult;
  Map<String, dynamic>? _cwtResult;
  Map<String, dynamic>? _hilbertResult;
  Map<String, dynamic>? _surrogatesResult;
  Map<String, dynamic>? _featuresResult;

  // Extra channels loaded from imported files (index 0 = primary stream)
  final List<List<double>> _extraChannels = [];
  bool _gpuAvailable = false;

  StreamSubscription<List<double>>? _bleSub;
  final _errorController = StreamController<String>.broadcast();

  // ── Public getters ──────────────────────────────────────────────────────────

  ServerStatus get serverStatus => _serverStatus;
  Map<String, double> get bandPowers => _bandPowersCache ??= Map.unmodifiable(_bandPowers);
  double get dominantFreq => _dominantFreq;
  double get signalQuality => _signalQuality;
  double get spectralEntropy => _spectralEntropy;
  double get spectralFlatness => _spectralFlatness;
  double get sampleRate => _sampleRate;
  List<double> get spectrum => _spectrumCache ??= List.unmodifiable(_spectrum);
  Map<String, dynamic>? get lastResult => _lastResult;
  bool get isSubmitting => _submitting;
  String? get pendingTaskId => _pendingTaskId;
  bool get hasData => _total >= 64;
  List<int> get changepoints => _changepointsCache ??= List.unmodifiable(_changepoints);

  bool get isSubmittingSyncMap   => _submittingSyncMap;
  Map<String, dynamic>? get syncMapResult => _syncMapResult;

  bool get isSubmittingBiphase   => _submittingBiphase;
  bool get isSubmittingBispec4   => _submittingBispec4;
  bool get isSubmittingCoupling  => _submittingCoupling;
  Map<String, dynamic>? get biphaseResult  => _biphaseResult;
  Map<String, dynamic>? get bispec4Result  => _bispec4Result;
  Map<String, dynamic>? get couplingResult => _couplingResult;

  bool get isSubmittingRidge   => _submittingRidge;
  bool get isSubmittingFilter  => _submittingFilter;
  bool get isSubmittingWft     => _submittingWft;
  bool get isSubmittingModwt   => _submittingModwt;
  bool get isSubmittingGroup   => _submittingGroup;
  Map<String, dynamic>? get ridgeResult  => _ridgeResult;
  Map<String, dynamic>? get filterResult => _filterResult;
  Map<String, dynamic>? get wftResult    => _wftResult;
  Map<String, dynamic>? get modwtResult  => _modwtResult;
  Map<String, dynamic>? get groupResult  => _groupResult;

  bool get isSubmittingBispectrum  => _submittingBispectrum;
  bool get isSubmittingCoherence   => _submittingCoherence;
  bool get isSubmittingBayesian    => _submittingBayesian;
  bool get isSubmittingStft        => _submittingStft;
  bool get isSubmittingCwt         => _submittingCwt;
  bool get isSubmittingHilbert     => _submittingHilbert;
  bool get isSubmittingSurrogates  => _submittingSurrogates;
  bool get isSubmittingFeatures    => _submittingFeatures;
  Map<String, dynamic>? get bispectrumResult  => _bispectrumResult;
  Map<String, dynamic>? get coherenceResult   => _coherenceResult;
  Map<String, dynamic>? get bayesianResult    => _bayesianResult;
  Map<String, dynamic>? get stftResult        => _stftResult;
  Map<String, dynamic>? get cwtResult         => _cwtResult;
  Map<String, dynamic>? get hilbertResult     => _hilbertResult;
  Map<String, dynamic>? get surrogatesResult  => _surrogatesResult;
  Map<String, dynamic>? get featuresResult    => _featuresResult;
  bool get gpuAvailable   => _gpuAvailable;
  int  get channelCount   => 1 + _extraChannels.length;
  List<List<double>> get extraChannels => List.unmodifiable(_extraChannels);

  void addChannel(List<double> samples) {
    _npyCache.remove(_extraChannels.length + 1);
    _extraChannels.add(samples);
    notifyListeners();
  }

  void clearExtraChannels() {
    _extraChannels.clear();
    _npyCache.removeWhere((k, _) => k > 0);
    notifyListeners();
  }

  List<int> bytesForChannel(int idx) {
    return _npyCache.putIfAbsent(idx, () {
      if (idx == 0) return packNpy(_recentSamplesSnapshot);
      return packNpy(_extraChannels[idx - 1]);
    });
  }

  /// Errors suitable for display as snackbars. Broadcast; subscribe once.
  Stream<String> get errors => _errorController.stream;

  set sampleRate(double fs) {
    _sampleRate = fs;
    notifyListeners();
  }

  List<double> get recentSamples => _recentSamplesSnapshot;

  void _rebuildSnapshot() {
    final count = math.min(_total, _bufferSize);
    if (_total < _bufferSize) {
      _recentSamplesSnapshot = List<double>.from(_buf.sublist(0, count));
    } else {
      final pos = _head % _bufferSize;
      _recentSamplesSnapshot = [..._buf.sublist(pos), ..._buf.sublist(0, pos)];
    }
  }

  // ── Signal ingestion ────────────────────────────────────────────────────────

  void addSamples(List<double> values) {
    for (final v in values) {
      _buf[_head % _bufferSize] = v;
      _head++;
    }
    _total += values.length;
    _rebuildSnapshot();
    _npyCache.remove(0); // invalidate channel-0 NPY cache
    notifyListeners();
    if (!_dftPending && _total >= _dftSize) _scheduleRecompute();
    if (!_changepointPending &&
        _total - _lastChangepointTotal >= _dftSize &&
        _total >= 64) {
      _scheduleChangepoint();
    }
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
    _spectralEntropy = (result['entropy'] as num).toDouble();
    _spectralFlatness = (result['flatness'] as num).toDouble();
    _bandPowersCache = null;
    _spectrumCache = null;
    _dftPending = false;
    if (hasListeners) notifyListeners();
  }

  void _scheduleChangepoint() async {
    _changepointPending = true;
    _lastChangepointTotal = _total;
    final samples = recentSamples;
    final result =
        await compute(changepointWorker, {'data': samples});
    _changepoints = List<int>.from(result['changepoints'] as List);
    _changepointsCache = null;
    _changepointPending = false;
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
    _fetchGpuInfo();
    _healthTimer?.cancel();
    _healthTimer =
        Timer.periodic(const Duration(seconds: 30), (_) => _pollHealth());
  }

  Future<void> _fetchGpuInfo() async {
    if (_client == null) return;
    try {
      final info = await _client!.getGpuInfo();
      _gpuAvailable = info['cuda_available'] == true ||
          info['pytorch_available'] == true;
      if (hasListeners) notifyListeners();
    } catch (_) {}
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
      _lastResult = await _awaitTask(taskId);
    } catch (e) {
      _errorController.add('Analysis submission failed — ${_friendly(e)}');
    } finally {
      _submitting = false;
      _pendingTaskId = null;
      if (hasListeners) notifyListeners();
    }
  }

  Future<Map<String, dynamic>?> _awaitTask(String taskId) async {
    for (int i = 0; i < 60; i++) {
      await Future.delayed(const Duration(seconds: 2));
      try {
        final status = await _client!.pollStatus(taskId);
        final s = status['status'] as String?;
        if (s == 'done' || s == 'complete' || s == 'success') return status;
        if (s == 'error' || s == 'failed') {
          _errorController.add('Server analysis returned an error.');
          return null;
        }
      } catch (_) {}
    }
    _errorController.add('Analysis timed out after 2 minutes.');
    return null;
  }

  Future<void> submitBispectrum() async {
    if (_client == null || _submittingBispectrum || !hasData) return;
    _submittingBispectrum = true;
    notifyListeners();
    try {
      final bytes = packNpy(recentSamples);
      final taskId = await _client!.submitBispectrum(
        signalBytes: bytes,
        samplingRate: _sampleRate,
      );
      _bispectrumResult = await _awaitTask(taskId);
    } catch (e) {
      _errorController.add('Bispectrum failed — ${_friendly(e)}');
    } finally {
      _submittingBispectrum = false;
      if (hasListeners) notifyListeners();
    }
  }

  Future<void> submitSyncMap({
    double band1Low = 8.0,
    double band1High = 12.0,
    double band2Low = 8.0,
    double band2High = 12.0,
  }) async {
    if (_client == null || _submittingSyncMap || channelCount < 2) return;
    _submittingSyncMap = true;
    notifyListeners();
    try {
      final taskId = await _client!.submitSyncMap(
        signal1Bytes: bytesForChannel(0),
        signal2Bytes: bytesForChannel(1),
        samplingRate: _sampleRate,
        band1Low: band1Low,
        band1High: band1High,
        band2Low: band2Low,
        band2High: band2High,
      );
      _syncMapResult = await _awaitTask(taskId);
    } catch (e) {
      _errorController.add('Sync map failed — ${_friendly(e)}');
    } finally {
      _submittingSyncMap = false;
      if (hasListeners) notifyListeners();
    }
  }

  Future<void> submitBiphase({
    double f1 = 6.0,
    double f2 = 10.0,
    String wavelet = 'lognorm',
  }) async {
    if (_client == null || _submittingBiphase || channelCount < 2) return;
    _submittingBiphase = true;
    notifyListeners();
    try {
      final taskId = await _client!.submitBiphase(
        signal1Bytes: bytesForChannel(0),
        signal2Bytes: bytesForChannel(1),
        samplingRate: _sampleRate,
        f1: f1,
        f2: f2,
        wavelet: wavelet,
      );
      _biphaseResult = await _awaitTask(taskId);
    } catch (e) {
      _errorController.add('Biphase failed — ${_friendly(e)}');
    } finally {
      _submittingBiphase = false;
      if (hasListeners) notifyListeners();
    }
  }

  Future<void> submitBispectrum4() async {
    if (_client == null || _submittingBispec4 || channelCount < 2) return;
    _submittingBispec4 = true;
    notifyListeners();
    try {
      final taskId = await _client!.submitBispectrum4(
        signal1Bytes: bytesForChannel(0),
        signal2Bytes: bytesForChannel(1),
        samplingRate: _sampleRate,
      );
      _bispec4Result = await _awaitTask(taskId);
    } catch (e) {
      _errorController.add('4-way bispectrum failed — ${_friendly(e)}');
    } finally {
      _submittingBispec4 = false;
      if (hasListeners) notifyListeners();
    }
  }

  Future<void> submitCoupling({
    double band1Low = 8.0,
    double band1High = 12.0,
    double band2Low = 8.0,
    double band2High = 12.0,
    int bn = 2,
    double winS = 1.0,
  }) async {
    if (_client == null || _submittingCoupling || channelCount < 2) return;
    _submittingCoupling = true;
    notifyListeners();
    try {
      final taskId = await _client!.submitCoupling(
        signal1Bytes: bytesForChannel(0),
        signal2Bytes: bytesForChannel(1),
        samplingRate: _sampleRate,
        bn: bn,
        winS: winS,
        band1Low: band1Low,
        band1High: band1High,
        band2Low: band2Low,
        band2High: band2High,
      );
      _couplingResult = await _awaitTask(taskId);
    } catch (e) {
      _errorController.add('Coupling functions failed — ${_friendly(e)}');
    } finally {
      _submittingCoupling = false;
      if (hasListeners) notifyListeners();
    }
  }

  Future<void> submitRidge({
    double freqMin = 0.5,
    double? freqMax,
    int nFreqs = 64,
    int smoothLen = 5,
  }) async {
    if (_client == null || _submittingRidge || !hasData) return;
    _submittingRidge = true;
    notifyListeners();
    try {
      final taskId = await _client!.submitRidge(
        signalBytes: packNpy(recentSamples),
        samplingRate: _sampleRate,
        freqMin: freqMin,
        freqMax: freqMax,
        nFreqs: nFreqs,
        smoothLen: smoothLen,
      );
      _ridgeResult = await _awaitTask(taskId);
    } catch (e) {
      _errorController.add('Ridge extraction failed — ${_friendly(e)}');
    } finally {
      _submittingRidge = false;
      if (hasListeners) notifyListeners();
    }
  }

  Future<void> submitFilterButter({
    double fLow = 0.5,
    double? fHigh,
    int order = 4,
    int detrendDegree = 0,
  }) async {
    if (_client == null || _submittingFilter || !hasData) return;
    _submittingFilter = true;
    notifyListeners();
    try {
      final taskId = await _client!.submitFilterButter(
        signalBytes: packNpy(recentSamples),
        samplingRate: _sampleRate,
        fLow: fLow,
        fHigh: fHigh,
        order: order,
        detrendDegree: detrendDegree,
      );
      _filterResult = await _awaitTask(taskId);
    } catch (e) {
      _errorController.add('Butterworth filter failed — ${_friendly(e)}');
    } finally {
      _submittingFilter = false;
      if (hasListeners) notifyListeners();
    }
  }

  Future<void> submitWft({int windowSize = 256, int hopSize = 128}) async {
    if (_client == null || _submittingWft || !hasData) return;
    _submittingWft = true;
    notifyListeners();
    try {
      final taskId = await _client!.submitWft(
        signalBytes: packNpy(recentSamples),
        samplingRate: _sampleRate,
        windowSize: windowSize,
        hopSize: hopSize,
      );
      _wftResult = await _awaitTask(taskId);
    } catch (e) {
      _errorController.add('WFT failed — ${_friendly(e)}');
    } finally {
      _submittingWft = false;
      if (hasListeners) notifyListeners();
    }
  }

  Future<void> submitModwt({String wavelet = 'la8', int level = 5}) async {
    if (_client == null || _submittingModwt || !hasData) return;
    _submittingModwt = true;
    notifyListeners();
    try {
      final taskId = await _client!.submitModwt(
        signalBytes: packNpy(recentSamples),
        samplingRate: _sampleRate,
        wavelet: wavelet,
        level: level,
      );
      _modwtResult = await _awaitTask(taskId);
    } catch (e) {
      _errorController.add('MODWT failed — ${_friendly(e)}');
    } finally {
      _submittingModwt = false;
      if (hasListeners) notifyListeners();
    }
  }

  /// Two-group comparison of mean wavelet power.
  /// Group 1 = the primary channel + any extra channels indexed in [group1Indices].
  /// Group 2 = extra channels indexed in [group2Indices].
  /// Both groups need ≥ 2 signals.
  Future<void> submitGroupComparison({
    required List<int> group1Indices,
    required List<int> group2Indices,
    double freqMin = 0.5,
    double? freqMax,
    int nFreqs = 50,
    String wavelet = 'lognorm',
  }) async {
    if (_client == null || _submittingGroup) return;
    if (group1Indices.length < 2 || group2Indices.length < 2) {
      _errorController.add('Group comparison needs ≥ 2 signals per group.');
      return;
    }
    _submittingGroup = true;
    notifyListeners();
    try {
      final g1 = [for (final i in group1Indices) bytesForChannel(i)];
      final g2 = [for (final i in group2Indices) bytesForChannel(i)];
      final taskId = await _client!.submitGroupComparison(
        group1: g1,
        group2: g2,
        samplingRate: _sampleRate,
        freqMin: freqMin,
        freqMax: freqMax,
        nFreqs: nFreqs,
        wavelet: wavelet,
      );
      _groupResult = await _awaitTask(taskId);
    } catch (e) {
      _errorController.add('Group comparison failed — ${_friendly(e)}');
    } finally {
      _submittingGroup = false;
      if (hasListeners) notifyListeners();
    }
  }

  Future<void> submitStft({int windowSize = 256, int hopSize = 128}) async {
    if (_client == null || _submittingStft || !hasData) return;
    _submittingStft = true;
    notifyListeners();
    try {
      final taskId = await _client!.submitStft(
        signalBytes: packNpy(recentSamples),
        samplingRate: _sampleRate,
        windowSize: windowSize,
        hopSize: hopSize,
      );
      _stftResult = await _awaitTask(taskId);
    } catch (e) {
      _errorController.add('STFT failed — ${_friendly(e)}');
    } finally {
      _submittingStft = false;
      if (hasListeners) notifyListeners();
    }
  }

  Future<void> submitCwt({double freqMin = 0.5, double? freqMax, int nFreqs = 50}) async {
    if (_client == null || _submittingCwt || !hasData) return;
    _submittingCwt = true;
    notifyListeners();
    try {
      final taskId = await _client!.submitCwt(
        signalBytes: packNpy(recentSamples),
        samplingRate: _sampleRate,
        freqMin: freqMin,
        freqMax: freqMax,
        nFreqs: nFreqs,
      );
      _cwtResult = await _awaitTask(taskId);
    } catch (e) {
      _errorController.add('CWT failed — ${_friendly(e)}');
    } finally {
      _submittingCwt = false;
      if (hasListeners) notifyListeners();
    }
  }

  Future<void> submitHilbert() async {
    if (_client == null || _submittingHilbert || !hasData) return;
    _submittingHilbert = true;
    notifyListeners();
    try {
      final taskId = await _client!.submitHilbert(
        signalBytes: packNpy(recentSamples),
        samplingRate: _sampleRate,
      );
      _hilbertResult = await _awaitTask(taskId);
    } catch (e) {
      _errorController.add('Hilbert failed — ${_friendly(e)}');
    } finally {
      _submittingHilbert = false;
      if (hasListeners) notifyListeners();
    }
  }

  Future<void> submitSurrogates({
    String testType = 'spectral',
    int nSurrogates = 19,
    String surrogateMethod = 'phase_randomization',
    double? targetFreq,
  }) async {
    if (_client == null || _submittingSurrogates || !hasData) return;
    _submittingSurrogates = true;
    notifyListeners();
    try {
      final taskId = await _client!.submitSurrogates(
        signalBytes: packNpy(recentSamples),
        samplingRate: _sampleRate,
        testType: testType,
        nSurrogates: nSurrogates,
        surrogateMethod: surrogateMethod,
        targetFreq: targetFreq,
      );
      _surrogatesResult = await _awaitTask(taskId);
    } catch (e) {
      _errorController.add('Surrogate test failed — ${_friendly(e)}');
    } finally {
      _submittingSurrogates = false;
      if (hasListeners) notifyListeners();
    }
  }

  Future<void> submitFeatures({
    List<String> analyses = const ['spectral', 'phase'],
  }) async {
    if (_client == null || _submittingFeatures || !hasData) return;
    _submittingFeatures = true;
    notifyListeners();
    try {
      final taskId = await _client!.submitFeatures(
        signalBytes: packNpy(recentSamples),
        samplingRate: _sampleRate,
        analyses: analyses,
      );
      _featuresResult = await _awaitTask(taskId);
    } catch (e) {
      _errorController.add('Feature extraction failed — ${_friendly(e)}');
    } finally {
      _submittingFeatures = false;
      if (hasListeners) notifyListeners();
    }
  }

  Future<void> submitCoherence({List<List<int>>? channelBytes}) async {
    final bytes = channelBytes ??
        List.generate(channelCount, bytesForChannel);
    if (_client == null || _submittingCoherence || bytes.length < 2) return;
    _submittingCoherence = true;
    notifyListeners();
    try {
      final taskId = await _client!.submitCoherence(
        signalBytesPerChannel: bytes,
        samplingRate: _sampleRate,
      );
      _coherenceResult = await _awaitTask(taskId);
    } catch (e) {
      _errorController.add('Coherence failed — ${_friendly(e)}');
    } finally {
      _submittingCoherence = false;
      if (hasListeners) notifyListeners();
    }
  }

  Future<void> submitBayesian(
      {required List<int> ch1Bytes, required List<int> ch2Bytes}) async {
    if (_client == null || _submittingBayesian) return;
    _submittingBayesian = true;
    notifyListeners();
    try {
      final taskId = await _client!.submitBayesian(
        signal1Bytes: ch1Bytes,
        signal2Bytes: ch2Bytes,
        samplingRate: _sampleRate,
      );
      _bayesianResult = await _awaitTask(taskId);
    } catch (e) {
      _errorController.add('Bayesian failed — ${_friendly(e)}');
    } finally {
      _submittingBayesian = false;
      if (hasListeners) notifyListeners();
    }
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
