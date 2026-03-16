# MODA Bluetooth Protocol Specification

## Overview

This document defines the Bluetooth Low Energy (BLE) protocol for transmitting signal data from external devices to the MODA mobile application.

**Version:** 1.0
**Protocol Name:** MODA-BLE-SP (MODA Bluetooth Low Energy Signal Protocol)

---

## 1. BLE Service & Characteristics

### 1.1 GATT Service Definition

**Service UUID:** `0000MOD-0000-1000-8000-00805F9B34FB`
**Service Name:** MODA Signal Service

### 1.2 Characteristics

| Characteristic | UUID | Properties | Description |
|----------------|------|------------|-------------|
| **Device Info** | `0000MODINF-0000-1000-8000-00805F9B34FB` | Read | Device metadata (name, version, capabilities) |
| **Signal Config** | `0000MODCFG-0000-1000-8000-00805F9B34FB` | Read, Write, Notify | Sampling rate, channel count, data format |
| **Signal Data** | `0000MODDAT-0000-1000-8000-00805F9B34FB` | Notify | Real-time signal data stream |
| **Control** | `0000MODCTL-0000-1000-8000-00805F9B34FB` | Write | Start/stop streaming, commands |
| **Status** | `0000MODSTS-0000-1000-8000-00805F9B34FB` | Read, Notify | Device status, battery, errors |

---

## 2. Data Formats

### 2.1 Device Info (Read)

**Format:** JSON string (UTF-8 encoded)

```json
{
  "device_name": "MODA Sensor v1.0",
  "manufacturer": "Lancaster Physics",
  "firmware_version": "1.2.3",
  "hardware_version": "2.0",
  "max_channels": 6,
  "max_sampling_rate": 1000,
  "supported_formats": ["int16", "float32"],
  "battery_level": 85
}
```

**Max Size:** 512 bytes

### 2.2 Signal Config (Read/Write)

**Format:** Binary structure (little-endian)

```
Offset | Size | Type    | Field              | Description
-------|------|---------|--------------------|--------------------------
0      | 2    | uint16  | sampling_rate      | Samples per second (Hz)
2      | 1    | uint8   | num_channels       | Number of active channels (1-6)
3      | 1    | uint8   | data_format        | 0=int16, 1=float32
4      | 2    | uint16  | packet_size        | Samples per packet
6      | 1    | uint8   | gain               | Amplification factor (0-255)
7      | 1    | uint8   | filter_enabled     | 0=off, 1=on
8      | 4    | float32 | filter_cutoff_low  | High-pass cutoff (Hz)
12     | 4    | float32 | filter_cutoff_high | Low-pass cutoff (Hz)
```

**Total Size:** 16 bytes

**Example Configuration:**
```
Sampling Rate: 100 Hz
Channels: 2
Format: int16
Packet Size: 10 samples
```

### 2.3 Signal Data (Notify)

**Format:** Binary packet with header

#### Packet Structure

```
Offset | Size    | Type    | Field           | Description
-------|---------|---------|-----------------|-----------------------------
0      | 1       | uint8   | packet_type     | 0x01 = Signal Data
1      | 1       | uint8   | sequence_num    | Incremental counter (wraps at 255)
2      | 4       | uint32  | timestamp_ms    | Milliseconds since start
6      | 2       | uint16  | num_samples     | Samples in this packet
8      | 1       | uint8   | num_channels    | Number of channels
9      | 1       | uint8   | flags           | Bit 0: Loss detected, Bit 1: Overflow
10     | N       | varies  | data            | Interleaved channel data
```

**Header Size:** 10 bytes
**Max Payload:** 512 bytes (BLE MTU limit)

#### Data Layout (Interleaved)

For `num_channels=2`, `num_samples=3`, `format=int16`:

```
[Ch1_S1][Ch2_S1][Ch1_S2][Ch2_S2][Ch1_S3][Ch2_S3]
```

Each sample is 2 bytes (int16) or 4 bytes (float32).

**Example Packet (2 channels, 10 samples, int16):**
```
Packet Type: 0x01
Sequence: 42
Timestamp: 1234567 ms
Num Samples: 10
Num Channels: 2
Flags: 0x00
Data: [40 bytes = 2 channels × 10 samples × 2 bytes]
Total: 50 bytes
```

### 2.4 Control Commands (Write)

**Format:** Binary command structure

```
Offset | Size | Type   | Field      | Description
-------|------|--------|------------|-----------------------------
0      | 1    | uint8  | command_id | Command identifier
1      | N    | varies | payload    | Command-specific data
```

#### Supported Commands

| Command ID | Name | Payload | Description |
|------------|------|---------|-------------|
| `0x01` | START_STREAMING | None | Begin signal transmission |
| `0x02` | STOP_STREAMING | None | Stop signal transmission |
| `0x03` | RESET | None | Reset device state |
| `0x04` | SET_CONFIG | 16 bytes | Update configuration (same format as Signal Config) |
| `0x05` | CALIBRATE | None | Trigger calibration routine |
| `0x10` | PING | None | Test connectivity |

**Response:** Status characteristic will be updated with command result.

### 2.5 Status (Read/Notify)

**Format:** Binary structure

```
Offset | Size | Type   | Field           | Description
-------|------|--------|-----------------|---------------------------
0      | 1    | uint8  | state           | 0=Idle, 1=Streaming, 2=Error
1      | 1    | uint8  | battery_level   | 0-100%
2      | 1    | uint8  | signal_quality  | 0-100 (0=poor, 100=excellent)
3      | 1    | uint8  | error_code      | 0=None (see error codes below)
4      | 4    | uint32 | packets_sent    | Total packets transmitted
8      | 4    | uint32 | packets_lost    | Estimated packet loss count
12     | 4    | float32| temperature     | Device temperature (°C)
```

**Total Size:** 16 bytes

#### State Codes
- `0x00`: Idle (ready to stream)
- `0x01`: Streaming (actively sending data)
- `0x02`: Error (see error_code)
- `0x03`: Calibrating
- `0x04`: Low Battery

#### Error Codes
- `0x00`: No error
- `0x01`: Configuration invalid
- `0x02`: Sensor malfunction
- `0x03`: Battery critical
- `0x04`: Memory overflow
- `0x10`: Communication timeout

---

## 3. Communication Flow

### 3.1 Connection & Discovery

```
1. Mobile App → Scan for BLE devices
2. Device → Advertise with Service UUID 0000MOD...
3. Mobile App → Connect to device
4. Mobile App → Discover services and characteristics
5. Mobile App → Read Device Info
6. Mobile App → Subscribe to Status notifications
```

### 3.2 Configuration

```
1. Mobile App → Read current Signal Config
2. Mobile App → Modify config (sampling rate, channels, etc.)
3. Mobile App → Write new Signal Config
4. Device → Update Status (state = Idle, error_code = 0x00 if valid)
5. Mobile App → Verify configuration accepted
```

### 3.3 Data Streaming

```
1. Mobile App → Subscribe to Signal Data notifications
2. Mobile App → Write Control: START_STREAMING
3. Device → Update Status (state = Streaming)
4. Device → Begin sending Signal Data packets at configured rate
5. Mobile App → Buffer incoming packets, check sequence numbers
6. [User initiates stop]
7. Mobile App → Write Control: STOP_STREAMING
8. Device → Update Status (state = Idle)
```

### 3.4 Error Handling

```
1. Device detects error (e.g., sensor malfunction)
2. Device → Update Status (state = Error, error_code = 0x02)
3. Device → Stop streaming
4. Mobile App receives Status notification
5. Mobile App → Display error to user
6. Mobile App → Write Control: RESET (if recoverable)
7. Device → Reset and return to Idle state
```

---

## 4. Timing & Performance

### 4.1 Latency Requirements

| Metric | Target | Notes |
|--------|--------|-------|
| **Connection Time** | <3 seconds | From scan to ready |
| **Config Update** | <500 ms | Write config → acknowledgment |
| **Stream Start Latency** | <200 ms | Command → first packet |
| **Packet Interval** | Consistent ±5% | Based on sampling rate |
| **End-to-End Latency** | <100 ms | Sensor → app display |

### 4.2 Data Rate Examples

| Sampling Rate | Channels | Format | Data Rate | Packets/sec |
|---------------|----------|--------|-----------|-------------|
| 100 Hz | 1 | int16 | 200 bytes/s | 10 |
| 100 Hz | 2 | int16 | 400 bytes/s | 20 |
| 250 Hz | 4 | int16 | 2000 bytes/s | 50 |
| 1000 Hz | 1 | float32 | 4000 bytes/s | 100 |

**BLE Throughput:** Typical BLE 4.2+ supports 10-20 KB/s effective throughput.

**Recommended Packet Size:** 10-50 samples per packet to balance latency and overhead.

---

## 5. Security

### 5.1 Pairing & Encryption

- **Pairing Method:** Numeric Comparison or Passkey Entry
- **Encryption:** AES-128 (BLE standard)
- **Security Level:** BLE Security Mode 1, Level 3 (authenticated encryption)

### 5.2 Authentication (Optional)

For medical or sensitive applications:

```
1. Mobile App → Read Device Info (includes device_id)
2. Mobile App → Verify device_id against whitelist
3. Mobile App → Write Control: AUTH_CHALLENGE [16-byte nonce]
4. Device → Compute HMAC-SHA256(nonce, shared_secret)
5. Device → Update Status with HMAC in custom field
6. Mobile App → Verify HMAC
```

---

## 6. Example Implementation

### 6.1 Mobile App (Flutter Pseudo-code)

```dart
import 'package:flutter_blue_plus/flutter_blue_plus.dart';

class MODABluetoothService {
  static const SERVICE_UUID = "0000MOD-0000-1000-8000-00805F9B34FB";
  static const SIGNAL_DATA_UUID = "0000MODDAT-0000-1000-8000-00805F9B34FB";
  static const CONTROL_UUID = "0000MODCTL-0000-1000-8000-00805F9B34FB";

  BluetoothDevice? device;
  BluetoothCharacteristic? signalDataChar;
  BluetoothCharacteristic? controlChar;

  Future<void> connect(BluetoothDevice device) async {
    await device.connect();
    List<BluetoothService> services = await device.discoverServices();

    for (var service in services) {
      if (service.uuid.toString() == SERVICE_UUID) {
        signalDataChar = service.characteristics
            .firstWhere((c) => c.uuid.toString() == SIGNAL_DATA_UUID);
        controlChar = service.characteristics
            .firstWhere((c) => c.uuid.toString() == CONTROL_UUID);
      }
    }

    // Subscribe to notifications
    await signalDataChar?.setNotifyValue(true);
    signalDataChar?.value.listen(_onSignalData);
  }

  Future<void> startStreaming() async {
    await controlChar?.write([0x01]); // START_STREAMING command
  }

  void _onSignalData(List<int> data) {
    // Parse packet header
    int packetType = data[0];
    int sequenceNum = data[1];
    int timestamp = ByteData.sublistView(Uint8List.fromList(data), 2, 6)
        .getUint32(0, Endian.little);
    int numSamples = ByteData.sublistView(Uint8List.fromList(data), 6, 8)
        .getUint16(0, Endian.little);
    int numChannels = data[8];

    // Extract signal data (assuming int16 format)
    List<List<double>> signals = List.generate(numChannels, (_) => []);
    int offset = 10; // Header size

    for (int i = 0; i < numSamples; i++) {
      for (int ch = 0; ch < numChannels; ch++) {
        int value = ByteData.sublistView(Uint8List.fromList(data), offset, offset + 2)
            .getInt16(0, Endian.little);
        signals[ch].add(value / 32768.0); // Normalize to [-1, 1]
        offset += 2;
      }
    }

    // Pass to signal processor
    processSignals(signals);
  }
}
```

### 6.2 Device Firmware (Arduino/C++ Pseudo-code)

```cpp
#include <BLEDevice.h>
#include <BLEServer.h>

#define SERVICE_UUID        "0000MOD-0000-1000-8000-00805F9B34FB"
#define SIGNAL_DATA_UUID    "0000MODDAT-0000-1000-8000-00805F9B34FB"
#define CONTROL_UUID        "0000MODCTL-0000-1000-8000-00805F9B34FB"

BLECharacteristic *pSignalDataChar;
bool streaming = false;
uint8_t sequenceNum = 0;

void setup() {
  BLEDevice::init("MODA Sensor");
  BLEServer *pServer = BLEDevice::createServer();
  BLEService *pService = pServer->createService(SERVICE_UUID);

  pSignalDataChar = pService->createCharacteristic(
    SIGNAL_DATA_UUID,
    BLECharacteristic::PROPERTY_NOTIFY
  );

  BLECharacteristic *pControlChar = pService->createCharacteristic(
    CONTROL_UUID,
    BLECharacteristic::PROPERTY_WRITE
  );
  pControlChar->setCallbacks(new ControlCallbacks());

  pService->start();
  BLEDevice::startAdvertising();
}

void loop() {
  if (streaming) {
    // Read sensor data (example: 2 channels, 10 samples)
    int16_t data[20]; // 2 channels × 10 samples
    readSensorData(data, 20);

    // Build packet
    uint8_t packet[50];
    packet[0] = 0x01; // Packet type
    packet[1] = sequenceNum++;
    *(uint32_t*)&packet[2] = millis(); // Timestamp
    *(uint16_t*)&packet[6] = 10; // Num samples
    packet[8] = 2; // Num channels
    packet[9] = 0x00; // Flags
    memcpy(&packet[10], data, 40); // Copy data

    // Send notification
    pSignalDataChar->setValue(packet, 50);
    pSignalDataChar->notify();

    delay(100); // 10 packets/sec for 100 Hz sampling
  }
}

class ControlCallbacks: public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic *pCharacteristic) {
    uint8_t *data = pCharacteristic->getData();
    uint8_t cmd = data[0];

    if (cmd == 0x01) {
      streaming = true;
      sequenceNum = 0;
    } else if (cmd == 0x02) {
      streaming = false;
    }
  }
};
```

---

## 7. Testing & Validation

### 7.1 Compliance Tests

- BLE 4.2+ compliance (use BLE sniffer tools)
- MTU negotiation (test with 23, 185, 512-byte MTUs)
- Connection stability (maintain connection for 1 hour)
- Packet loss measurement (<1% under normal conditions)

### 7.2 Interoperability

Test with:
- iOS devices (iPhone 8+, iPad Pro)
- Android devices (Android 8.0+)
- Multiple BLE chipsets (Nordic nRF52, ESP32, TI CC2640)

### 7.3 Performance Benchmarks

- Measure end-to-end latency with oscilloscope
- Validate data integrity (send known test signal)
- Test maximum sustained data rate
- Battery life testing (target: 8+ hours continuous streaming)

---

## 8. Future Enhancements

- **Multi-device Support:** Simultaneous connections to multiple sensors
- **Compression:** Real-time data compression for higher sampling rates
- **OTA Updates:** Over-the-air firmware updates via BLE
- **Advanced Filters:** On-device filtering and preprocessing
- **Sync Protocol:** Precise timestamp synchronization across devices

---

**Document Version:** 1.0
**Last Updated:** 2026-03-04
**Maintained by:** MODA Development Team
