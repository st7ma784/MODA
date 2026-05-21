import 'package:dio/dio.dart';
import '../config/app_config.dart';

class FastModaClient {
  late Dio _dio;
  String _baseUrl;

  FastModaClient({String? baseUrl})
      : _baseUrl = baseUrl ?? kFastModaDefaultUrl {
    _initDio();
  }

  void _initDio() {
    _dio = Dio(BaseOptions(
      baseUrl: _baseUrl,
      connectTimeout: kApiTimeout,
      receiveTimeout: kAnalysisReceiveTimeout,
    ));
    _dio.interceptors.add(_ApiKeyInterceptor());
  }

  void setBaseUrl(String url) {
    _baseUrl = url;
    _initDio();
  }

  String get baseUrl => _baseUrl;

  Future<Map<String, dynamic>> checkHealth() async {
    final res = await _dio.get('/health');
    return res.data as Map<String, dynamic>;
  }

  /// Submits signal data for asynchronous analysis. Returns task_id.
  Future<String> submitAnalysis({
    required List<int> signalBytes,
    required double samplingRate,
    double windowSize = 1.0,
    String penalty = 'auto',
  }) async {
    final form = FormData.fromMap({
      'file': MultipartFile.fromBytes(signalBytes, filename: 'signal.npy'),
      'fs': samplingRate.toString(),
      'win': windowSize.toString(),
      'pen': penalty,
    });
    final res = await _dio.post('/analyze', data: form);
    return (res.data as Map<String, dynamic>)['task_id'] as String;
  }

  Future<Map<String, dynamic>> pollStatus(String taskId) async {
    final res = await _dio.get('/status/$taskId');
    return res.data as Map<String, dynamic>;
  }

  /// MODWT (Maximal Overlap Discrete Wavelet Transform).
  /// Returns task_id; poll with [pollStatus].
  Future<String> submitModwt({
    required List<int> signalBytes,
    required double samplingRate,
    String wavelet = 'la8',
    int level = 5,
  }) async {
    final form = FormData.fromMap({
      'file': MultipartFile.fromBytes(signalBytes, filename: 'signal.npy'),
      'fs': samplingRate.toString(),
      'wavelet': wavelet,
      'level': level.toString(),
    });
    final res = await _dio.post('/analyze_modwt', data: form);
    return (res.data as Map<String, dynamic>)['task_id'] as String;
  }

  /// Two-group statistical comparison of mean wavelet power across frequencies.
  /// `group1` and `group2` must each contain ≥ 2 signal payloads.
  /// Returns task_id; poll with [pollStatus].
  Future<String> submitGroupComparison({
    required List<List<int>> group1,
    required List<List<int>> group2,
    required double samplingRate,
    double freqMin = 0.5,
    double? freqMax,
    int nFreqs = 50,
    String wavelet = 'lognorm',
  }) async {
    final form = FormData.fromMap({
      'g1': [
        for (int i = 0; i < group1.length; i++)
          MultipartFile.fromBytes(group1[i], filename: 'g1_$i.npy'),
      ],
      'g2': [
        for (int i = 0; i < group2.length; i++)
          MultipartFile.fromBytes(group2[i], filename: 'g2_$i.npy'),
      ],
      'fs': samplingRate.toString(),
      'freq_min': freqMin.toString(),
      if (freqMax != null) 'freq_max': freqMax.toString(),
      'n_freqs': nFreqs.toString(),
      'wavelet': wavelet,
    });
    final res = await _dio.post('/analyze_group', data: form);
    return (res.data as Map<String, dynamic>)['task_id'] as String;
  }

  /// Bispectrum analysis — single signal, server self-pairs it.
  /// Returns task_id; poll with [pollStatus].
  Future<String> submitBispectrum({
    required List<int> signalBytes,
    required double samplingRate,
    double freqMin = 0.5,
    double? freqMax,
    int nFreqs = 50,
    String bispecType = '122',
  }) async {
    final form = FormData.fromMap({
      'files': MultipartFile.fromBytes(signalBytes, filename: 'signal.npy'),
      'fs': samplingRate.toString(),
      'freq_min': freqMin.toString(),
      if (freqMax != null) 'freq_max': freqMax.toString(),
      'n_freqs': nFreqs.toString(),
      'bispec_type': bispecType,
    });
    final res = await _dio.post('/analyze_bispectrum', data: form);
    return (res.data as Map<String, dynamic>)['task_id'] as String;
  }

  /// Coherence analysis — requires 2–6 signals (separate channels).
  /// Returns task_id; poll with [pollStatus].
  Future<String> submitCoherence({
    required List<List<int>> signalBytesPerChannel,
    required double samplingRate,
    double windowSize = 1.0,
    double overlap = 0.5,
    int numcycles = 10,
  }) async {
    final form = FormData.fromMap({
      'files': [
        for (int i = 0; i < signalBytesPerChannel.length; i++)
          MultipartFile.fromBytes(signalBytesPerChannel[i],
              filename: 'signal_$i.npy'),
      ],
      'fs': samplingRate.toString(),
      'win': windowSize.toString(),
      'overlap': overlap.toString(),
      'numcycles': numcycles.toString(),
    });
    final res = await _dio.post('/analyze_coherence', data: form);
    return (res.data as Map<String, dynamic>)['task_id'] as String;
  }

  /// Bayesian phase-coupling inference — requires exactly 2 signals.
  /// Returns task_id; poll with [pollStatus].
  Future<String> submitBayesian({
    required List<int> signal1Bytes,
    required List<int> signal2Bytes,
    required double samplingRate,
    List<double> band1 = const [0.5, 2.0],
    List<double> band2 = const [0.5, 2.0],
    double windowS = 40.0,
    int nSurrogates = 19,
  }) async {
    final form = FormData.fromMap({
      'files': [
        MultipartFile.fromBytes(signal1Bytes, filename: 'signal_0.npy'),
        MultipartFile.fromBytes(signal2Bytes, filename: 'signal_1.npy'),
      ],
      'fs': samplingRate.toString(),
      'band1_low': band1[0].toString(),
      'band1_high': band1[1].toString(),
      'band2_low': band2[0].toString(),
      'band2_high': band2[1].toString(),
      'window_s': windowS.toString(),
      'n_surrogates': nSurrogates.toString(),
    });
    final res = await _dio.post('/analyze_bayesian', data: form);
    return (res.data as Map<String, dynamic>)['task_id'] as String;
  }

  /// Ridge extraction: instantaneous frequency, amplitude, phase, reconstruction.
  Future<String> submitRidge({
    required List<int> signalBytes,
    required double samplingRate,
    double freqMin = 0.5,
    double? freqMax,
    int nFreqs = 64,
    int smoothLen = 5,
    double nCycles = 6.0,
    String wavelet = 'lognorm',
    bool cutEdges = true,
  }) async {
    final form = FormData.fromMap({
      'file': MultipartFile.fromBytes(signalBytes, filename: 'signal.npy'),
      'fs': samplingRate.toString(),
      'freq_min': freqMin.toString(),
      if (freqMax != null) 'freq_max': freqMax.toString(),
      'n_freqs': nFreqs.toString(),
      'smooth_len': smoothLen.toString(),
      'n_cycles': nCycles.toString(),
      'wavelet': wavelet,
      'cut_edges': cutEdges.toString(),
    });
    final res = await _dio.post('/analyze_ridge', data: form);
    return (res.data as Map<String, dynamic>)['task_id'] as String;
  }

  /// Butterworth bandpass filter with optional polynomial detrend.
  Future<String> submitFilterButter({
    required List<int> signalBytes,
    required double samplingRate,
    double fLow = 0.5,
    double? fHigh,
    int order = 4,
    int detrendDegree = 0,
  }) async {
    final form = FormData.fromMap({
      'file': MultipartFile.fromBytes(signalBytes, filename: 'signal.npy'),
      'fs': samplingRate.toString(),
      'f_low': fLow.toString(),
      if (fHigh != null) 'f_high': fHigh.toString(),
      'order': order.toString(),
      'detrend_degree': detrendDegree.toString(),
    });
    final res = await _dio.post('/filter_butter', data: form);
    return (res.data as Map<String, dynamic>)['task_id'] as String;
  }

  /// Synchronisation map — detect 1:1 phase-locking from coupling functions.
  Future<String> submitSyncMap({
    required List<int> signal1Bytes,
    required List<int> signal2Bytes,
    required double samplingRate,
    int bn = 3,
    double winS = 1.0,
    double band1Low = 8.0,
    double band1High = 12.0,
    double band2Low = 8.0,
    double band2High = 12.0,
  }) async {
    final form = FormData.fromMap({
      'files': [
        MultipartFile.fromBytes(signal1Bytes, filename: 'signal_0.npy'),
        MultipartFile.fromBytes(signal2Bytes, filename: 'signal_1.npy'),
      ],
      'fs': samplingRate.toString(),
      'bn': bn.toString(),
      'win_s': winS.toString(),
      'band1_low': band1Low.toString(),
      'band1_high': band1High.toString(),
      'band2_low': band2Low.toString(),
      'band2_high': band2High.toString(),
    });
    final res = await _dio.post('/analyze_syncmap', data: form);
    return (res.data as Map<String, dynamic>)['task_id'] as String;
  }

  /// Biphase time series at a specific frequency pair.
  Future<String> submitBiphase({
    required List<int> signal1Bytes,
    required List<int> signal2Bytes,
    required double samplingRate,
    required double f1,
    required double f2,
    String wavelet = 'lognorm',
    double nCycles = 6.0,
  }) async {
    final form = FormData.fromMap({
      'files': [
        MultipartFile.fromBytes(signal1Bytes, filename: 'signal_0.npy'),
        MultipartFile.fromBytes(signal2Bytes, filename: 'signal_1.npy'),
      ],
      'fs': samplingRate.toString(),
      'f1': f1.toString(),
      'f2': f2.toString(),
      'wavelet': wavelet,
      'n_cycles': nCycles.toString(),
    });
    final res = await _dio.post('/analyze_biphase', data: form);
    return (res.data as Map<String, dynamic>)['task_id'] as String;
  }

  /// Four-way cross-bispectrum (b111/b222/b122/b211).
  Future<String> submitBispectrum4({
    required List<int> signal1Bytes,
    required List<int> signal2Bytes,
    required double samplingRate,
    int nfft = 256,
  }) async {
    final form = FormData.fromMap({
      'files': [
        MultipartFile.fromBytes(signal1Bytes, filename: 'signal_0.npy'),
        MultipartFile.fromBytes(signal2Bytes, filename: 'signal_1.npy'),
      ],
      'fs': samplingRate.toString(),
      'nfft': nfft.toString(),
    });
    final res = await _dio.post('/analyze_bispectrum4', data: form);
    return (res.data as Map<String, dynamic>)['task_id'] as String;
  }

  /// Coupling function estimation (Bayesian-style OLS).
  Future<String> submitCoupling({
    required List<int> signal1Bytes,
    required List<int> signal2Bytes,
    required double samplingRate,
    int bn = 3,
    double winS = 40.0,
    double overlap = 0.5,
    double band1Low = 0.5,
    double band1High = 2.0,
    double band2Low = 0.5,
    double band2High = 2.0,
  }) async {
    final form = FormData.fromMap({
      'files': [
        MultipartFile.fromBytes(signal1Bytes, filename: 'signal_0.npy'),
        MultipartFile.fromBytes(signal2Bytes, filename: 'signal_1.npy'),
      ],
      'fs': samplingRate.toString(),
      'bn': bn.toString(),
      'win_s': winS.toString(),
      'overlap': overlap.toString(),
      'band1_low': band1Low.toString(),
      'band1_high': band1High.toString(),
      'band2_low': band2Low.toString(),
      'band2_high': band2High.toString(),
    });
    final res = await _dio.post('/analyze_coupling', data: form);
    return (res.data as Map<String, dynamic>)['task_id'] as String;
  }

  /// Windowed Fourier Transform (Gaussian-windowed STFT).
  Future<String> submitWft({
    required List<int> signalBytes,
    required double samplingRate,
    int windowSize = 256,
    int hopSize = 128,
  }) async {
    final form = FormData.fromMap({
      'file': MultipartFile.fromBytes(signalBytes, filename: 'signal.npy'),
      'fs': samplingRate.toString(),
      'window_size': windowSize.toString(),
      'hop_size': hopSize.toString(),
    });
    final res = await _dio.post('/analyze_wft', data: form);
    return (res.data as Map<String, dynamic>)['task_id'] as String;
  }

  Future<Map<String, dynamic>> getGpuInfo() async {
    final res = await _dio.get('/api/gpu-info');
    return res.data as Map<String, dynamic>;
  }

  Future<String> submitStft({
    required List<int> signalBytes,
    required double samplingRate,
    int windowSize = 256,
    int hopSize = 128,
    String window = 'hann',
  }) async {
    final form = FormData.fromMap({
      'file': MultipartFile.fromBytes(signalBytes, filename: 'signal.npy'),
      'fs': samplingRate.toString(),
      'window_size': windowSize.toString(),
      'hop_size': hopSize.toString(),
      'window': window,
    });
    final res = await _dio.post('/analyze_stft', data: form);
    return (res.data as Map<String, dynamic>)['task_id'] as String;
  }

  Future<String> submitCwt({
    required List<int> signalBytes,
    required double samplingRate,
    double freqMin = 0.5,
    double? freqMax,
    int nFreqs = 50,
  }) async {
    final form = FormData.fromMap({
      'file': MultipartFile.fromBytes(signalBytes, filename: 'signal.npy'),
      'fs': samplingRate.toString(),
      'freq_min': freqMin.toString(),
      if (freqMax != null) 'freq_max': freqMax.toString(),
      'n_freqs': nFreqs.toString(),
    });
    final res = await _dio.post('/analyze_cwt', data: form);
    return (res.data as Map<String, dynamic>)['task_id'] as String;
  }

  Future<String> submitHilbert({
    required List<int> signalBytes,
    required double samplingRate,
  }) async {
    final form = FormData.fromMap({
      'file': MultipartFile.fromBytes(signalBytes, filename: 'signal.npy'),
      'fs': samplingRate.toString(),
    });
    final res = await _dio.post('/analyze_hilbert', data: form);
    return (res.data as Map<String, dynamic>)['task_id'] as String;
  }

  Future<String> submitSurrogates({
    required List<int> signalBytes,
    required double samplingRate,
    String testType = 'spectral',
    int nSurrogates = 19,
    String surrogateMethod = 'iaaft',
    double? targetFreq,
  }) async {
    final form = FormData.fromMap({
      'file': MultipartFile.fromBytes(signalBytes, filename: 'signal.npy'),
      'fs': samplingRate.toString(),
      'test_type': testType,
      'n_surrogates': nSurrogates.toString(),
      'surrogate_method': surrogateMethod,
      if (targetFreq != null) 'target_freq': targetFreq.toString(),
    });
    final res = await _dio.post('/analyze_surrogates', data: form);
    return (res.data as Map<String, dynamic>)['task_id'] as String;
  }

  Future<String> submitFeatures({
    required List<int> signalBytes,
    required double samplingRate,
    List<String> analyses = const ['spectral', 'phase'],
  }) async {
    final form = FormData.fromMap({
      'file': MultipartFile.fromBytes(signalBytes, filename: 'signal.npy'),
      'fs': samplingRate.toString(),
      'analyses': analyses.join(','),
    });
    final res = await _dio.post('/analyze_features', data: form);
    return (res.data as Map<String, dynamic>)['task_id'] as String;
  }
}

class _ApiKeyInterceptor extends Interceptor {
  @override
  void onRequest(RequestOptions options, RequestInterceptorHandler handler) {
    options.headers['X-API-Key'] = kFastModaApiKey;
    handler.next(options);
  }
}
