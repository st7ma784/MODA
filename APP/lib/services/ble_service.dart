import 'dart:async';
import 'dart:convert';
import 'dart:io' show Platform;
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import 'package:permission_handler/permission_handler.dart';
import '../config/app_config.dart';
import '../utils/moda_protocol.dart';

// ── MODA device data classes ──────────────────────────────────────────────────

class ModeDeviceInfo {
  final String deviceName;
  final String firmwareVersion;
  final int maxChannels;
  final int maxSamplingRate;
  final int batteryLevel; // -1 = unknown
  final List<String> supportedFormats;

  const ModeDeviceInfo({
    this.deviceName = 'MODA Sensor',
    this.firmwareVersion = '',
    this.maxChannels = 1,
    this.maxSamplingRate = 256,
    this.batteryLevel = -1,
    this.supportedFormats = const ['int16'],
  });
}

class ModeSignalConfig {
  final int samplingRate;
  final int numChannels;
  final int dataFormat; // 0 = int16, 1 = float32
  final int packetSize;
  final int gain;
  final bool filterEnabled;
  final double filterCutoffLow;
  final double filterCutoffHigh;

  const ModeSignalConfig({
    this.samplingRate = 256,
    this.numChannels = 1,
    this.dataFormat = 0,
    this.packetSize = 10,
    this.gain = 1,
    this.filterEnabled = false,
    this.filterCutoffLow = 0.5,
    this.filterCutoffHigh = 100.0,
  });
}

class ModeDeviceStatus {
  final int state; // 0=idle 1=streaming 2=error 3=calibrating 4=low_battery
  final int batteryLevel;
  final int signalQuality; // 0–100
  final int errorCode;
  final int packetsSent;
  final int packetsLost;
  final double temperature;

  const ModeDeviceStatus({
    this.state = 0,
    this.batteryLevel = 100,
    this.signalQuality = 0,
    this.errorCode = 0,
    this.packetsSent = 0,
    this.packetsLost = 0,
    this.temperature = 0,
  });

  bool get isStreaming => state == 1;
  bool get hasError => state == 2;
  String get stateLabel => switch (state) {
        0 => 'Idle',
        1 => 'Streaming',
        2 => 'Error',
        3 => 'Calibrating',
        4 => 'Low Battery',
        _ => 'Unknown ($state)',
      };
}

// ── Generic enum (non-MODA devices) ──────────────────────────────────────────

enum BleDataFormat { int16LE, float32LE }

// ── BleService ────────────────────────────────────────────────────────────────

class BleService extends ChangeNotifier {
  // Scan
  final List<ScanResult> _scanResults = [];
  StreamSubscription<List<ScanResult>>? _scanSub;
  bool _isScanning = false;

  // Connection
  BluetoothDevice? _connectedDevice;

  // MODA protocol state
  bool _isModaDevice = false;
  ModeDeviceInfo? _deviceInfo;
  ModeSignalConfig? _signalConfig;
  ModeDeviceStatus? _deviceStatus;
  BluetoothCharacteristic? _controlChar;
  StreamSubscription<List<int>>? _dataNotifySub;
  StreamSubscription<List<int>>? _statusNotifySub;
  int _expectedSeq = 0;
  int _appPacketsLost = 0;
  bool _streaming = false;

  // Generic (non-MODA) state
  List<BluetoothCharacteristic> _characteristics = [];
  BluetoothCharacteristic? _activeChar;
  StreamSubscription<List<int>>? _genericNotifySub;
  BleDataFormat _genericFormat = BleDataFormat.int16LE;

  // Output streams
  final _sampleController = StreamController<List<double>>.broadcast();
  final _errorController = StreamController<String>.broadcast();

  // ── Public getters ──────────────────────────────────────────────────────────

  List<ScanResult> get scanResults => List.unmodifiable(_scanResults);
  BluetoothDevice? get connectedDevice => _connectedDevice;
  bool get isScanning => _isScanning;
  bool get isConnected => _connectedDevice != null;
  bool get isModaDevice => _isModaDevice;
  bool get isStreaming => _streaming || _activeChar != null;
  int get packetsLost => _appPacketsLost;

  ModeDeviceInfo? get deviceInfo => _deviceInfo;
  ModeSignalConfig? get signalConfig => _signalConfig;
  ModeDeviceStatus? get deviceStatus => _deviceStatus;

  List<BluetoothCharacteristic> get characteristics =>
      List.unmodifiable(_characteristics);
  BluetoothCharacteristic? get activeCharacteristic => _activeChar;

  Stream<List<double>> get sampleStream => _sampleController.stream;

  /// Errors suitable for display as snackbars. Broadcast; subscribe once.
  Stream<String> get errors => _errorController.stream;

  /// Current adapter state stream — use in a StreamBuilder for the BT-off banner.
  Stream<BluetoothAdapterState> get adapterState =>
      FlutterBluePlus.adapterState;

  // ── Permissions + adapter check ─────────────────────────────────────────────

  Future<bool> _ensurePermissions() async {
    if (!Platform.isAndroid && !Platform.isIOS) return true;

    final List<Permission> perms = Platform.isAndroid
        ? [
            Permission.bluetoothScan,
            Permission.bluetoothConnect,
            Permission.locationWhenInUse, // required on Android < 12
          ]
        : [Permission.bluetooth];

    final results = await perms.request();

    for (final entry in results.entries) {
      final isBtPerm = entry.key == Permission.bluetoothScan ||
          entry.key == Permission.bluetoothConnect ||
          entry.key == Permission.bluetooth;
      if (isBtPerm &&
          (entry.value.isDenied || entry.value.isPermanentlyDenied)) {
        _errorController.add(
          'Bluetooth permission denied. '
          'Go to Settings → Apps → MODA → Permissions to enable it.',
        );
        return false;
      }
    }
    return true;
  }

  Future<bool> _checkAdapterOn() async {
    try {
      final state = await FlutterBluePlus.adapterState
          .first
          .timeout(const Duration(seconds: 3));
      if (state != BluetoothAdapterState.on) {
        _errorController.add(
          state == BluetoothAdapterState.off
              ? 'Bluetooth is disabled. Please enable it and try again.'
              : 'Bluetooth is unavailable on this device.',
        );
        return false;
      }
      return true;
    } catch (_) {
      return true; // timeout — assume on and let the scan fail naturally
    }
  }

  // ── Scanning ────────────────────────────────────────────────────────────────

  Future<void> startScan({Duration timeout = const Duration(seconds: 10)}) async {
    if (!await _ensurePermissions()) return;
    if (!await _checkAdapterOn()) return;

    _scanResults.clear();
    _isScanning = true;
    notifyListeners();

    try {
      // Subscribe BEFORE starting scan — results arrive during the scan
      // and FlutterBluePlus.startScan awaits until timeout elapses.
      _scanSub = FlutterBluePlus.scanResults.listen((results) {
        for (final r in results) {
          final idx = _scanResults
              .indexWhere((e) => e.device.remoteId == r.device.remoteId);
          if (idx >= 0) {
            _scanResults[idx] = r;
          } else {
            _scanResults.add(r);
          }
        }
        notifyListeners();
      });
      await FlutterBluePlus.startScan(timeout: timeout);
      // startScan future completes when timeout elapses — no extra delay needed.
    } catch (e) {
      _errorController.add('Scan failed: ${_friendly(e)}');
    } finally {
      await stopScan();
    }
  }

  Future<void> stopScan() async {
    await FlutterBluePlus.stopScan().catchError((_) {});
    await _scanSub?.cancel();
    _scanSub = null;
    _isScanning = false;
    notifyListeners();
  }

  // ── Connection ──────────────────────────────────────────────────────────────

  Future<void> connect(BluetoothDevice device) async {
    try {
      await device.connect(autoConnect: false);
    } catch (e) {
      final name = device.platformName.isNotEmpty
          ? device.platformName
          : device.remoteId.toString();
      _errorController.add('Could not connect to $name: ${_friendly(e)}');
      return;
    }
    _connectedDevice = device;
    _isModaDevice = false;
    _deviceInfo = null;
    _signalConfig = null;
    _deviceStatus = null;
    _appPacketsLost = 0;
    notifyListeners();
    await _discoverAndInit();
  }

  Future<void> _discoverAndInit() async {
    if (_connectedDevice == null) return;
    try {
      final services = await _connectedDevice!.discoverServices();

      final modaSvc = services
          .where((s) =>
              s.uuid.toString().toLowerCase() ==
              kModaServiceUuid.toLowerCase())
          .firstOrNull;

      if (modaSvc != null) {
        _isModaDevice = true;
        _characteristics = [];
        notifyListeners();
        await _initModaProtocol(modaSvc);
      } else {
        _isModaDevice = false;
        _characteristics = services
            .expand((s) => s.characteristics)
            .where((c) => c.properties.notify || c.properties.indicate)
            .toList();
        notifyListeners();
      }
    } catch (e) {
      _errorController.add('Service discovery failed: ${_friendly(e)}');
    }
  }

  // ── MODA Protocol ───────────────────────────────────────────────────────────

  Future<void> _initModaProtocol(BluetoothService svc) async {
    BluetoothCharacteristic? find(String uuid) => svc.characteristics
        .where((c) => c.uuid.toString().toLowerCase() == uuid.toLowerCase())
        .firstOrNull;

    try {
      // 1. Read Device Info (JSON)
      final infoChar = find(kModaDeviceInfoUuid);
      if (infoChar != null && infoChar.properties.read) {
        try {
          _deviceInfo = _parseDeviceInfo(await infoChar.read());
        } catch (e) {
          debugPrint('Device info read error: $e');
          _deviceInfo = const ModeDeviceInfo();
        }
      }

      // 2. Read Signal Config (16-byte binary struct)
      final configChar = find(kModaSignalConfigUuid);
      if (configChar != null && configChar.properties.read) {
        try {
          _signalConfig = _parseSignalConfig(await configChar.read());
        } catch (e) {
          debugPrint('Signal config read error: $e');
          _signalConfig = const ModeSignalConfig();
        }
      }
      _signalConfig ??= const ModeSignalConfig();

      // 3. Store Control characteristic
      _controlChar = find(kModaControlUuid);

      // 4. Subscribe to Status
      final statusChar = find(kModaStatusUuid);
      if (statusChar != null &&
          (statusChar.properties.notify || statusChar.properties.indicate)) {
        await statusChar.setNotifyValue(true);
        _statusNotifySub =
            statusChar.onValueReceived.listen(_onModaStatus);
      }

      // 5. Subscribe to Signal Data
      final dataChar = find(kModaSignalDataUuid);
      if (dataChar != null &&
          (dataChar.properties.notify || dataChar.properties.indicate)) {
        await dataChar.setNotifyValue(true);
        _dataNotifySub = dataChar.onValueReceived.listen(_onModaPacket);
      }

      notifyListeners();

      // 6. START_STREAMING
      await _sendControl(0x01);
    } catch (e) {
      _errorController.add('MODA protocol error: ${_friendly(e)}');
    }
  }

  Future<void> _sendControl(int commandId,
      {List<int> payload = const []}) async {
    if (_controlChar == null) return;
    try {
      await _controlChar!.write([commandId, ...payload]);
      if (commandId == 0x01) {
        _streaming = true;
        _expectedSeq = 0;
        _appPacketsLost = 0;
      } else if (commandId == 0x02) {
        _streaming = false;
      }
      notifyListeners();
    } catch (e) {
      _errorController.add('Control write failed: ${_friendly(e)}');
    }
  }

  void _onModaPacket(List<int> bytes) {
    final isFloat32 = (_signalConfig?.dataFormat ?? 0) == 1;
    final packet = parseModaPacket(bytes, isFloat32: isFloat32);
    if (packet == null) return;

    // Detect gaps in sequence numbers (wraps at 255)
    final gap = (packet.sequenceNum - _expectedSeq) & 0xff;
    if (gap > 0) _appPacketsLost += gap;
    _expectedSeq = (packet.sequenceNum + 1) & 0xff;

    if (packet.channel0.isNotEmpty) {
      _sampleController.add(packet.channel0);
    }
  }

  void _onModaStatus(List<int> bytes) {
    if (bytes.length < 16) return;
    final bd = ByteData.view(Uint8List.fromList(bytes).buffer);
    _deviceStatus = ModeDeviceStatus(
      state: bytes[0],
      batteryLevel: bytes[1],
      signalQuality: bytes[2],
      errorCode: bytes[3],
      packetsSent: bd.getUint32(4, Endian.little),
      packetsLost: bd.getUint32(8, Endian.little),
      temperature: bd.getFloat32(12, Endian.little),
    );
    if (_deviceStatus!.hasError) {
      _errorController.add(
          'MODA sensor error (code ${_deviceStatus!.errorCode}). '
          'Try disconnecting and reconnecting.');
    }
    notifyListeners();
  }

  ModeDeviceInfo _parseDeviceInfo(List<int> bytes) {
    try {
      final map =
          jsonDecode(utf8.decode(bytes)) as Map<String, dynamic>;
      return ModeDeviceInfo(
        deviceName: map['device_name'] as String? ?? 'MODA Sensor',
        firmwareVersion: map['firmware_version'] as String? ?? '',
        maxChannels: map['max_channels'] as int? ?? 1,
        maxSamplingRate: map['max_sampling_rate'] as int? ?? 256,
        batteryLevel: map['battery_level'] as int? ?? -1,
        supportedFormats:
            (map['supported_formats'] as List?)?.cast<String>() ??
                ['int16'],
      );
    } catch (_) {
      return const ModeDeviceInfo();
    }
  }

  ModeSignalConfig _parseSignalConfig(List<int> bytes) {
    if (bytes.length < 16) return const ModeSignalConfig();
    final bd = ByteData.view(Uint8List.fromList(bytes).buffer);
    return ModeSignalConfig(
      samplingRate: bd.getUint16(0, Endian.little),
      numChannels: bytes[2],
      dataFormat: bytes[3],
      packetSize: bd.getUint16(4, Endian.little),
      gain: bytes[6],
      filterEnabled: bytes[7] != 0,
      filterCutoffLow: bd.getFloat32(8, Endian.little),
      filterCutoffHigh: bd.getFloat32(12, Endian.little),
    );
  }

  // ── Generic characteristic streaming (non-MODA devices) ────────────────────

  Future<void> subscribeToCharacteristic(
    BluetoothCharacteristic char, {
    BleDataFormat format = BleDataFormat.int16LE,
  }) async {
    try {
      await _genericNotifySub?.cancel();
      await _activeChar?.setNotifyValue(false).catchError((_) {});
      _genericFormat = format;
      _activeChar = char;
      await char.setNotifyValue(true);
      _genericNotifySub = char.onValueReceived.listen((bytes) {
        if (bytes.isEmpty) return;
        final samples = _parseGenericBytes(bytes);
        if (samples.isNotEmpty) _sampleController.add(samples);
      });
      notifyListeners();
    } catch (e) {
      _errorController.add(
          'Could not subscribe to characteristic: ${_friendly(e)}');
    }
  }

  Future<void> stopGenericStreaming() async {
    await _genericNotifySub?.cancel();
    _genericNotifySub = null;
    await _activeChar?.setNotifyValue(false).catchError((_) {});
    _activeChar = null;
    notifyListeners();
  }

  List<double> _parseGenericBytes(List<int> bytes) {
    switch (_genericFormat) {
      case BleDataFormat.int16LE:
        final result = <double>[];
        for (int i = 0; i + 1 < bytes.length; i += 2) {
          result.add(
              (bytes[i] | (bytes[i + 1] << 8)).toSigned(16) / 32768.0);
        }
        return result;
      case BleDataFormat.float32LE:
        if (bytes.length < 4) return [];
        final bd = ByteData.view(Uint8List.fromList(bytes).buffer);
        return List.generate(
            bytes.length ~/ 4,
            (i) => bd.getFloat32(i * 4, Endian.little));
    }
  }

  // ── Stop / disconnect ───────────────────────────────────────────────────────

  Future<void> stopStreaming() async {
    if (_isModaDevice) {
      await _sendControl(0x02);
    } else {
      await stopGenericStreaming();
    }
  }

  Future<void> disconnect() async {
    await stopStreaming();
    await _dataNotifySub?.cancel();
    await _statusNotifySub?.cancel();
    _dataNotifySub = null;
    _statusNotifySub = null;
    _controlChar = null;
    try {
      await _connectedDevice?.disconnect();
    } catch (_) {}
    _connectedDevice = null;
    _isModaDevice = false;
    _streaming = false;
    _characteristics = [];
    _deviceInfo = null;
    _signalConfig = null;
    _deviceStatus = null;
    notifyListeners();
  }

  // ── Helpers ─────────────────────────────────────────────────────────────────

  static String _friendly(Object e) {
    final msg = e.toString();
    if (msg.contains('device is disconnected') ||
        msg.contains('GATT_DISCONNECT')) return 'device disconnected';
    if (msg.contains('timeout')) return 'operation timed out';
    if (msg.contains('permission')) return 'permission denied';
    return msg.split('\n').first;
  }

  @override
  void dispose() {
    _scanSub?.cancel();
    _genericNotifySub?.cancel();
    _dataNotifySub?.cancel();
    _statusNotifySub?.cancel();
    _sampleController.close();
    _errorController.close();
    super.dispose();
  }
}
