import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import '../config/app_config.dart';

class AppSettings {
  static const _urlKey = 'server_url';
  static const _bleUuidKey = 'ble_char_uuid';
  static const _sampleRateKey = 'sample_rate';
  static const _dataFormatKey = 'ble_data_format';

  final _storage = const FlutterSecureStorage(
    aOptions: AndroidOptions(encryptedSharedPreferences: true),
  );

  Future<String> getServerUrl() async =>
      (await _storage.read(key: _urlKey)) ?? kFastModaDefaultUrl;

  Future<void> setServerUrl(String url) =>
      _storage.write(key: _urlKey, value: url);

  Future<String> getBleCharUuid() async =>
      (await _storage.read(key: _bleUuidKey)) ?? '';

  Future<void> setBleCharUuid(String uuid) =>
      _storage.write(key: _bleUuidKey, value: uuid);

  Future<double> getSampleRate() async {
    final s = await _storage.read(key: _sampleRateKey);
    return s != null ? double.tryParse(s) ?? kDefaultSampleRate : kDefaultSampleRate;
  }

  Future<void> setSampleRate(double fs) =>
      _storage.write(key: _sampleRateKey, value: fs.toString());

  Future<String> getDataFormat() async =>
      (await _storage.read(key: _dataFormatKey)) ?? 'int16';

  Future<void> setDataFormat(String format) =>
      _storage.write(key: _dataFormatKey, value: format);
}
