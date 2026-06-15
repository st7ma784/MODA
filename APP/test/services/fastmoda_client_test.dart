import 'dart:convert';
import 'dart:typed_data';

import 'package:dio/dio.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:moda_mobile/config/app_config.dart';
import 'package:moda_mobile/services/fastmoda_client.dart';

/// Records every request it sees and replies with a canned JSON body chosen
/// by [responder], so [FastModaClient] methods can be exercised without a
/// real network connection.
class _FakeAdapter implements HttpClientAdapter {
  final Map<String, dynamic> Function(RequestOptions options) responder;
  final List<RequestOptions> requests = [];

  _FakeAdapter(this.responder);

  @override
  Future<ResponseBody> fetch(
    RequestOptions options,
    Stream<Uint8List>? requestStream,
    Future<void>? cancelFuture,
  ) async {
    requests.add(options);
    final body = responder(options);
    return ResponseBody.fromString(
      jsonEncode(body),
      200,
      headers: {Headers.contentTypeHeader: ['application/json']},
    );
  }

  @override
  void close({bool force = false}) {}
}

FastModaClient _clientWith(Map<String, dynamic> Function(RequestOptions) responder,
    {required _FakeAdapter Function(_FakeAdapter) capture}) {
  final adapter = _FakeAdapter(responder);
  capture(adapter);
  final client = FastModaClient(baseUrl: 'http://fastmoda.test');
  client.httpClientAdapter = adapter;
  return client;
}

void main() {
  group('FastModaClient — auth header', () {
    test('every request carries the X-API-Key header', () async {
      late _FakeAdapter fake;
      final client = _clientWith(
        (o) => {'status': 'ok'},
        capture: (a) => fake = a,
      );

      await client.checkHealth();

      expect(fake.requests.single.headers['X-API-Key'], kFastModaApiKey);
    });
  });

  group('FastModaClient — recordings', () {
    test('uploadRecording posts multipart form and returns recording_id', () async {
      late _FakeAdapter fake;
      final client = _clientWith(
        (o) => {'recording_id': 'rec-123'},
        capture: (a) => fake = a,
      );

      final id = await client.uploadRecording(
        signalBytes: [1, 2, 3, 4],
        samplingRate: 256.0,
        deviceId: 'device-1',
        signalType: 'eeg',
        isBaseline: true,
      );

      expect(id, 'rec-123');
      final req = fake.requests.single;
      expect(req.method, 'POST');
      expect(req.path, '/recordings');
      final form = req.data as FormData;
      final fields = Map.fromEntries(form.fields);
      expect(fields['device_id'], 'device-1');
      expect(fields['fs'], '256.0');
      expect(fields['signal_type'], 'eeg');
      expect(fields['is_baseline'], 'true');
      expect(form.files.single.key, 'file');
    });

    test('listRecordings returns the recordings list', () async {
      late _FakeAdapter fake;
      final client = _clientWith(
        (o) => {
          'recordings': [
            {'id': 'rec-1', 'signal_type': 'eeg', 'is_baseline': 0},
            {'id': 'rec-2', 'signal_type': 'eeg', 'is_baseline': 1},
          ],
        },
        capture: (a) => fake = a,
      );

      final recordings = await client.listRecordings('device-1');

      expect(recordings, hasLength(2));
      expect(recordings.first['id'], 'rec-1');
      expect(fake.requests.single.path, '/recordings/device-1');
    });
  });

  group('FastModaClient — baseline', () {
    test('getBaseline returns n_samples and features', () async {
      late _FakeAdapter fake;
      final client = _clientWith(
        (o) => {
          'device_id': 'device-1',
          'n_samples': 3,
          'features': {'hr_mean': {'mean': 72.0, 'std': 4.0, 'n': 3}},
        },
        capture: (a) => fake = a,
      );

      final baseline = await client.getBaseline('device-1');

      expect(baseline['n_samples'], 3);
      expect((baseline['features'] as Map)['hr_mean'], isNotNull);
      expect(fake.requests.single.path, '/baseline/device-1');
      expect(fake.requests.single.method, 'GET');
    });

    test('calibrateBaseline posts recording_id and returns updated counts', () async {
      late _FakeAdapter fake;
      final client = _clientWith(
        (o) => {
          'device_id': 'device-1',
          'recording_id': 'rec-123',
          'n_samples': 4,
          'n_features': 42,
        },
        capture: (a) => fake = a,
      );

      final result = await client.calibrateBaseline(
        deviceId: 'device-1',
        recordingId: 'rec-123',
      );

      expect(result['n_samples'], 4);
      expect(result['n_features'], 42);
      final req = fake.requests.single;
      expect(req.method, 'POST');
      expect(req.path, '/baseline/device-1/calibrate');
      expect((req.data as Map)['recording_id'], 'rec-123');
    });
  });

  group('FastModaClient — classification & labelling', () {
    test('classify returns conditions and deviations', () async {
      late _FakeAdapter fake;
      final client = _clientWith(
        (o) => {
          'device_id': 'device-1',
          'recording_id': 'rec-123',
          'used_baseline': true,
          'conditions': {
            'afib': {
              'probability': 0.82,
              'top_features': [
                {'name': 'rmssd', 'value': 12.0, 'deviation': -2.1, 'contribution': 0.9},
              ],
            },
            'normal': {'probability': 0.1, 'top_features': []},
          },
          'deviations': [
            {'name': 'rmssd', 'value': 12.0, 'deviation': -2.1},
          ],
        },
        capture: (a) => fake = a,
      );

      final result = await client.classify(
        recordingId: 'rec-123',
        deviceId: 'device-1',
      );

      expect(result['used_baseline'], isTrue);
      final conditions = result['conditions'] as Map<String, dynamic>;
      expect((conditions['afib'] as Map)['probability'], 0.82);
      final req = fake.requests.single;
      expect(req.path, '/classify');
      expect((req.data as Map)['recording_id'], 'rec-123');
      expect((req.data as Map)['device_id'], 'device-1');
    });

    test('submitLabel posts the label payload', () async {
      late _FakeAdapter fake;
      final client = _clientWith(
        (o) => <String, dynamic>{},
        capture: (a) => fake = a,
      );

      await client.submitLabel(
        recordingId: 'rec-123',
        condition: 'normal',
        source: 'self',
        confidence: 0.9,
      );

      final req = fake.requests.single;
      expect(req.method, 'POST');
      expect(req.path, '/recordings/rec-123/label');
      final body = req.data as Map;
      expect(body['condition'], 'normal');
      expect(body['source'], 'self');
      expect(body['confidence'], 0.9);
      expect(body.containsKey('severity'), isFalse);
    });
  });
}
