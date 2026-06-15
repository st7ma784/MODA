import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:uuid/uuid.dart';
import '../config/app_config.dart';
import 'signal_service.dart' show SignalType, ChangepointMode;

class AppSettings {
  static const _urlKey = 'server_url';
  static const _bleUuidKey = 'ble_char_uuid';
  static const _sampleRateKey = 'sample_rate';
  static const _changepointThresholdKey = 'changepoint_threshold';
  static const _dftSizeKey = 'dft_size';
  static const _dataFormatKey = 'ble_data_format';
  static const _signalTypeKey = 'signal_type';
  static const _changepointModeKey = 'changepoint_mode';
  static const _deviceIdKey = 'moda_device_id';

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

    Future<double> getChangepointThreshold() async {
        final s = await _storage.read(key: _changepointThresholdKey);
        return s != null ? double.tryParse(s) ?? 1.0 : 1.0;
    }

    Future<void> setChangepointThreshold(double t) =>
            _storage.write(key: _changepointThresholdKey, value: t.toString());

    Future<int> getDftSize() async {
        final s = await _storage.read(key: _dftSizeKey);
        return s != null ? int.tryParse(s) ?? 256 : 256;
    }

    Future<void> setDftSize(int n) =>
            _storage.write(key: _dftSizeKey, value: n.toString());

  Future<void> setSampleRate(double fs) =>
      _storage.write(key: _sampleRateKey, value: fs.toString());

  Future<String> getDataFormat() async =>
      (await _storage.read(key: _dataFormatKey)) ?? 'int16';

  Future<void> setDataFormat(String format) =>
      _storage.write(key: _dataFormatKey, value: format);

  Future<SignalType> getSignalType() async {
    final s = await _storage.read(key: _signalTypeKey);
    return s == 'generic' ? SignalType.generic : SignalType.eeg;
  }

  Future<void> setSignalType(SignalType t) =>
      _storage.write(key: _signalTypeKey, value: t.name);

  Future<ChangepointMode> getChangepointMode() async {
    final s = await _storage.read(key: _changepointModeKey);
    return switch (s) {
      'envelope'  => ChangepointMode.envelope,
      'frequency' => ChangepointMode.frequency,
      _           => ChangepointMode.raw,
    };
  }

  Future<void> setChangepointMode(ChangepointMode m) =>
      _storage.write(key: _changepointModeKey, value: m.name);

  /// Stable per-install identifier used to associate uploaded recordings and
  /// baselines with this device/patient on the FastMODA server. Generated
  /// once on first call and persisted thereafter.
  Future<String> getDeviceId() async {
    final existing = await _storage.read(key: _deviceIdKey);
    if (existing != null && existing.isNotEmpty) return existing;
    final id = const Uuid().v4();
    await _storage.write(key: _deviceIdKey, value: id);
    return id;
  }
}
