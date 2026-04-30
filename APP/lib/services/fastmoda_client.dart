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

  Future<Map<String, dynamic>> submitModwt({
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
    return res.data as Map<String, dynamic>;
  }
}

class _ApiKeyInterceptor extends Interceptor {
  @override
  void onRequest(RequestOptions options, RequestInterceptorHandler handler) {
    options.headers['X-API-Key'] = kFastModaApiKey;
    handler.next(options);
  }
}
