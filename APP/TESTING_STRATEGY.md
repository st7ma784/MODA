# MODA Mobile App Testing Strategy

## Overview

This document outlines the comprehensive testing strategy for the MODA mobile application, covering unit tests, integration tests, performance tests, and cross-platform validation.

**Target Coverage:** 80%+ code coverage
**Testing Frameworks:** Flutter Test, Mockito, Integration Test, XCTest, Espresso

---

## 1. Testing Pyramid

```
                    ┌─────────────────┐
                    │   E2E Tests     │  5% (Manual + Automated)
                    │   (10 tests)    │
                    └─────────────────┘
                   ┌───────────────────┐
                   │ Integration Tests │  15% (Automated)
                   │   (50 tests)      │
                   └───────────────────┘
              ┌──────────────────────────┐
              │   Widget/UI Tests        │  30% (Automated)
              │   (100 tests)            │
              └──────────────────────────┘
         ┌─────────────────────────────────┐
         │     Unit Tests                  │  50% (Automated)
         │     (300+ tests)                │
         └─────────────────────────────────┘
```

---

## 2. Unit Testing

### 2.1 Scope

Test individual functions, methods, and classes in isolation.

### 2.2 Components to Test

#### Bluetooth Layer
- **Connection Manager**
  - Device discovery logic
  - Pairing state machine
  - Reconnection logic
  - Error handling

```dart
// Example: test/bluetooth/connection_manager_test.dart
import 'package:flutter_test/flutter_test.dart';
import 'package:mockito/mockito.dart';

void main() {
  group('ConnectionManager', () {
    late ConnectionManager manager;
    late MockBluetoothDevice mockDevice;

    setUp(() {
      mockDevice = MockBluetoothDevice();
      manager = ConnectionManager();
    });

    test('should connect to device successfully', () async {
      when(mockDevice.connect()).thenAnswer((_) async => true);

      final result = await manager.connectToDevice(mockDevice);

      expect(result, isTrue);
      verify(mockDevice.connect()).called(1);
    });

    test('should retry on connection failure', () async {
      when(mockDevice.connect())
          .thenThrow(Exception('Connection failed'))
          .thenAnswer((_) async => true); // Succeed on retry

      final result = await manager.connectToDevice(mockDevice);

      expect(result, isTrue);
      verify(mockDevice.connect()).called(2);
    });

    test('should timeout after max retries', () async {
      when(mockDevice.connect()).thenThrow(Exception('Connection failed'));

      expect(
        () => manager.connectToDevice(mockDevice, maxRetries: 3),
        throwsA(isA<ConnectionTimeoutException>()),
      );
    });
  });
}
```

#### Signal Processing
- **Buffer Management**
  - Circular buffer operations
  - Data overflow handling
  - Sample rate conversion

```dart
// test/signal/buffer_test.dart
test('circular buffer should handle overflow correctly', () {
  final buffer = CircularBuffer<double>(capacity: 100);

  // Add 150 samples
  for (int i = 0; i < 150; i++) {
    buffer.add(i.toDouble());
  }

  expect(buffer.length, equals(100));
  expect(buffer.first, equals(50.0)); // First 50 dropped
  expect(buffer.last, equals(149.0));
});
```

#### Data Models
- **Packet Parser**
  - Binary data parsing
  - Endianness handling
  - Checksum validation

```dart
// test/models/signal_packet_test.dart
test('should parse signal packet correctly', () {
  final bytes = Uint8List.fromList([
    0x01, 0x2A, // Packet type, sequence
    0x87, 0xD6, 0x12, 0x00, // Timestamp (1234567)
    0x0A, 0x00, // Num samples (10)
    0x02, 0x00, // Num channels (2)
    ...// Data bytes
  ]);

  final packet = SignalPacket.fromBytes(bytes);

  expect(packet.packetType, equals(0x01));
  expect(packet.sequenceNum, equals(42));
  expect(packet.timestamp, equals(1234567));
  expect(packet.numSamples, equals(10));
  expect(packet.numChannels, equals(2));
});
```

#### API Client
- **HTTP Request Building**
  - Endpoint construction
  - Multipart form data
  - Error response parsing

```dart
// test/api/fastmoda_client_test.dart
test('should build analysis request correctly', () {
  final client = FastMODAClient(baseUrl: 'http://localhost:5000');
  final signal = Float64List.fromList([1.0, 2.0, 3.0]);

  final request = client.buildAnalysisRequest(
    signal: signal,
    samplingRate: 100.0,
    windowSize: 1.0,
  );

  expect(request.method, equals('POST'));
  expect(request.url.path, equals('/analyze'));
  expect(request.fields['fs'], equals('100.0'));
});
```

### 2.3 Testing Tools

- **Framework:** `flutter_test`
- **Mocking:** `mockito` or `mocktail`
- **Coverage:** `flutter test --coverage`

### 2.4 Success Criteria

- **Coverage:** >85% for critical paths (BLE, API, signal processing)
- **Coverage:** >70% overall
- **Execution Time:** <30 seconds for full suite

---

## 3. Widget/UI Testing

### 3.1 Scope

Test UI components, user interactions, and state changes.

### 3.2 Components to Test

#### Dashboard Screen
```dart
// test/widgets/dashboard_test.dart
testWidgets('should display signal quality indicator', (tester) async {
  await tester.pumpWidget(MaterialApp(
    home: Dashboard(signalQuality: 85),
  ));

  expect(find.text('Signal Quality: 85%'), findsOneWidget);
  expect(find.byType(CircularProgressIndicator), findsOneWidget);
});

testWidgets('should show connection status', (tester) async {
  final mockBloc = MockBluetoothBloc();
  when(mockBloc.state).thenReturn(BluetoothConnected());

  await tester.pumpWidget(MaterialApp(
    home: BlocProvider.value(
      value: mockBloc,
      child: Dashboard(),
    ),
  ));

  expect(find.byIcon(Icons.bluetooth_connected), findsOneWidget);
  expect(find.text('Connected'), findsOneWidget);
});
```

#### Chart Widgets
```dart
// test/widgets/spectrogram_test.dart
testWidgets('spectrogram should render with data', (tester) async {
  final data = SpectrogramData(
    frequencies: [0, 10, 20, 30],
    times: [0, 1, 2],
    values: [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], ...],
  );

  await tester.pumpWidget(MaterialApp(
    home: Spectrogram(data: data),
  ));

  expect(find.byType(CustomPaint), findsOneWidget);
  // Verify chart renders (check for canvas operations)
});
```

#### Settings Screen
```dart
testWidgets('should update sampling rate preference', (tester) async {
  await tester.pumpWidget(MaterialApp(home: SettingsScreen()));

  // Find sampling rate dropdown
  final dropdown = find.byKey(Key('sampling_rate_dropdown'));
  await tester.tap(dropdown);
  await tester.pumpAndSettle();

  // Select 250 Hz
  await tester.tap(find.text('250 Hz').last);
  await tester.pumpAndSettle();

  // Verify preference saved
  final prefs = await SharedPreferences.getInstance();
  expect(prefs.getInt('sampling_rate'), equals(250));
});
```

### 3.3 Testing Tools

- **Framework:** `flutter_test` with `WidgetTester`
- **Golden Tests:** Screenshot comparison for visual regression
- **Accessibility:** `SemanticsController` for screen reader testing

### 3.4 Success Criteria

- **Coverage:** All major screens and user flows
- **Execution Time:** <2 minutes
- **Visual Regression:** 100% match with golden images

---

## 4. Integration Testing

### 4.1 Scope

Test interactions between multiple components and external systems.

### 4.2 Test Scenarios

#### End-to-End Signal Flow
```dart
// integration_test/signal_flow_test.dart
testWidgets('full signal acquisition and analysis flow', (tester) async {
  app.main();
  await tester.pumpAndSettle();

  // 1. Connect to mock BLE device
  await tester.tap(find.text('Scan Devices'));
  await tester.pumpAndSettle(Duration(seconds: 2));

  await tester.tap(find.text('Mock Sensor'));
  await tester.pumpAndSettle(Duration(seconds: 1));

  expect(find.text('Connected'), findsOneWidget);

  // 2. Configure settings
  await tester.tap(find.byIcon(Icons.settings));
  await tester.pumpAndSettle();

  await tester.enterText(find.byKey(Key('sampling_rate')), '100');
  await tester.tap(find.text('Save'));
  await tester.pumpAndSettle();

  // 3. Start streaming
  await tester.tap(find.text('Start Streaming'));
  await tester.pump(Duration(seconds: 5)); // Collect 5 seconds of data

  expect(find.byType(SignalPlot), findsOneWidget);

  // 4. Run analysis
  await tester.tap(find.text('Analyze'));
  await tester.pumpAndSettle(Duration(seconds: 10)); // Wait for API

  expect(find.text('Analysis Complete'), findsOneWidget);
  expect(find.byType(Spectrogram), findsOneWidget);
});
```

#### API Integration
```dart
// integration_test/api_test.dart
test('should upload signal and receive analysis results', () async {
  final client = FastMODAClient(baseUrl: 'http://localhost:5000');
  final signal = generateTestSignal(length: 1000, fs: 100);

  // Upload
  final taskId = await client.uploadSignal(
    signal: signal,
    samplingRate: 100.0,
  );

  expect(taskId, isNotEmpty);

  // Poll for results
  AnalysisResult? result;
  for (int i = 0; i < 20; i++) {
    await Future.delayed(Duration(seconds: 1));
    result = await client.getAnalysisResult(taskId);
    if (result.status == 'complete') break;
  }

  expect(result?.status, equals('complete'));
  expect(result?.frequencySummary, isNotEmpty);
  expect(result?.numChangepoints, greaterThan(0));
});
```

#### Bluetooth Integration (with mock device)
```dart
// integration_test/bluetooth_test.dart
testWidgets('BLE connection and data reception', (tester) async {
  // Use mock BLE peripheral (e.g., ESP32 running test firmware)
  final device = await findMockDevice();

  app.main();
  await tester.pumpAndSettle();

  // Connect
  await tester.tap(find.text('Connect'));
  await tester.pumpAndSettle(Duration(seconds: 3));

  // Verify connection
  expect(find.text('Connected'), findsOneWidget);

  // Start streaming
  await tester.tap(find.text('Start'));
  await tester.pump(Duration(seconds: 2));

  // Verify data reception
  final signalData = await getReceivedSignalData();
  expect(signalData.length, greaterThan(100)); // At least 100 samples
  expect(signalData.every((v) => v.isFinite), isTrue);
});
```

### 4.3 Testing Tools

- **Framework:** `integration_test` package
- **API Mock:** `mockito` or real FastMODA instance in Docker
- **BLE Mock:** nRF52 DK or ESP32 running test firmware

### 4.4 Success Criteria

- **Coverage:** All critical user journeys (5-10 scenarios)
- **Execution Time:** <5 minutes per test
- **Pass Rate:** 100% on target devices

---

## 5. Performance Testing

### 5.1 Metrics

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **App Launch Time** | <2s | Time to first frame |
| **BLE Connection Time** | <3s | Scan → connected |
| **Signal Display Latency** | <100ms | BLE RX → UI update |
| **Memory Usage** | <150MB | Continuous streaming |
| **Battery Drain** | <5%/hour | Background streaming |
| **Frame Rate** | >55 FPS | During chart updates |

### 5.2 Test Procedures

#### Cold Start Time
```dart
// performance_test/startup_test.dart
void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('measure cold start time', (tester) async {
    final startTime = DateTime.now();

    app.main();
    await tester.pumpAndSettle();

    final endTime = DateTime.now();
    final duration = endTime.difference(startTime);

    print('Cold start time: ${duration.inMilliseconds}ms');
    expect(duration.inMilliseconds, lessThan(2000));
  });
}
```

#### Memory Profiling
```bash
# Use Flutter DevTools
flutter run --profile
# Open DevTools → Memory tab
# Collect snapshots during:
# - Idle
# - BLE streaming (5 min)
# - Analysis (heavy computation)
# - After GC
```

#### Battery Testing
```
1. Charge device to 100%
2. Disconnect charger
3. Run app with continuous BLE streaming
4. Monitor battery level every 15 minutes for 2 hours
5. Calculate drain rate (%/hour)
```

### 5.3 Tools

- **Flutter DevTools:** Profiling, memory, performance overlay
- **Android Profiler:** CPU, memory, network, battery
- **Xcode Instruments:** Time Profiler, Allocations, Energy Log
- **Custom Benchmarks:** `flutter drive --profile`

---

## 6. Cross-Platform Testing

### 6.1 Device Matrix

#### iOS
| Device | OS Version | Screen Size | Test Priority |
|--------|------------|-------------|---------------|
| iPhone SE (3rd gen) | iOS 16+ | 4.7" | High |
| iPhone 13 | iOS 16+ | 6.1" | High |
| iPhone 15 Pro | iOS 17+ | 6.1" | Medium |
| iPad Air | iPadOS 16+ | 10.9" | Low |

#### Android
| Device | OS Version | Screen Size | Manufacturer | Test Priority |
|--------|------------|-------------|--------------|---------------|
| Pixel 6 | Android 13+ | 6.4" | Google | High |
| Samsung Galaxy S21 | Android 12+ | 6.2" | Samsung | High |
| OnePlus 9 | Android 12+ | 6.55" | OnePlus | Medium |
| Generic Tablet | Android 11+ | 10" | Various | Low |

### 6.2 Testing Approach

- **Physical Devices:** High-priority devices (4-6 devices)
- **Cloud Device Farm:** AWS Device Farm or Firebase Test Lab (20+ devices)
- **Emulators:** Quick smoke tests only (not for BLE testing)

### 6.3 Test Coverage per Platform

```
✓ Installation and first launch
✓ BLE permission handling
✓ Background/foreground transitions
✓ Network connectivity changes
✓ Low battery scenarios
✓ Airplane mode transitions
✓ Screen rotation (portrait/landscape)
✓ Dark mode / Light mode
```

---

## 7. Security & Compliance Testing

### 7.1 Security Tests

#### Data Encryption
```dart
test('should encrypt local database', () async {
  final db = await DatabaseHelper.instance.database;
  final cipher = db.rawQuery('PRAGMA cipher_version');

  expect(cipher, isNotEmpty); // SQLCipher enabled
});
```

#### BLE Pairing Security
```
1. Attempt to connect without pairing → should fail
2. Complete numeric comparison pairing → should succeed
3. Verify encrypted connection (use BLE sniffer)
4. Attempt MITM attack → should be prevented
```

#### API Authentication
```dart
test('should include auth token in API requests', () async {
  final client = FastMODAClient(baseUrl: 'https://api.example.com');
  final request = await client.buildRequest('/analyze');

  expect(request.headers['Authorization'], startsWith('Bearer '));
});
```

### 7.2 Compliance

- **GDPR:** Verify user consent flows, data deletion
- **HIPAA (if medical):** Encrypt data at rest and in transit, audit logs
- **App Store Guidelines:** Privacy policy, permissions justification

---

## 8. Automated Testing Pipeline

### 8.1 CI/CD Integration (GitHub Actions)

```yaml
# .github/workflows/test.yml
name: Test Suite

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: subosito/flutter-action@v2
        with:
          flutter-version: '3.16.0'
      - run: flutter pub get
      - run: flutter test --coverage
      - uses: codecov/codecov-action@v3
        with:
          files: ./coverage/lcov.info

  widget-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: subosito/flutter-action@v2
      - run: flutter pub get
      - run: flutter test test/widgets --reporter expanded

  integration-tests-ios:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v3
      - uses: subosito/flutter-action@v2
      - run: flutter pub get
      - run: |
          flutter drive \
            --driver=test_driver/integration_test.dart \
            --target=integration_test/app_test.dart \
            -d iPhone

  integration-tests-android:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: subosito/flutter-action@v2
      - uses: reactivecircus/android-emulator-runner@v2
        with:
          api-level: 33
          script: |
            flutter drive \
              --driver=test_driver/integration_test.dart \
              --target=integration_test/app_test.dart
```

### 8.2 Test Execution Schedule

- **On every commit:** Unit tests, widget tests
- **On pull request:** Full suite (unit + widget + integration)
- **Nightly:** Full suite + performance tests on device farm
- **Pre-release:** Manual QA + security audit

---

## 9. Test Data & Fixtures

### 9.1 Synthetic Signals

```dart
// test/fixtures/signal_generator.dart
class SignalGenerator {
  static Float64List generateSineWave({
    required double frequency,
    required double samplingRate,
    required int length,
    double amplitude = 1.0,
    double phase = 0.0,
  }) {
    final signal = Float64List(length);
    for (int i = 0; i < length; i++) {
      final t = i / samplingRate;
      signal[i] = amplitude * sin(2 * pi * frequency * t + phase);
    }
    return signal;
  }

  static Float64List generateMultiComponent({
    required List<double> frequencies,
    required List<double> amplitudes,
    required double samplingRate,
    required int length,
  }) {
    final signal = Float64List(length);
    for (int i = 0; i < length; i++) {
      final t = i / samplingRate;
      double value = 0;
      for (int j = 0; j < frequencies.length; j++) {
        value += amplitudes[j] * sin(2 * pi * frequencies[j] * t);
      }
      signal[i] = value;
    }
    return signal;
  }
}
```

### 9.2 Mock API Responses

```dart
// test/fixtures/api_responses.dart
const mockAnalysisResponse = {
  "status": "complete",
  "progress": 100,
  "results": {
    "frequency_summary": [
      {"rank": 1, "frequency": 10.5, "band": "alpha", "duration_pct": 45.2},
      {"rank": 2, "frequency": 20.1, "band": "beta", "duration_pct": 30.8},
    ],
    "num_changepoints": 12,
  }
};
```

---

## 10. Bug Reporting & Tracking

### 10.1 Severity Levels

| Level | Description | SLA |
|-------|-------------|-----|
| **Critical** | App crash, data loss, security breach | Fix within 24h |
| **High** | Major feature broken, BLE connection fails | Fix within 3 days |
| **Medium** | Minor feature issue, UI glitch | Fix within 1 week |
| **Low** | Cosmetic issue, typo | Fix in next release |

### 10.2 Bug Report Template

```markdown
## Bug Description
[Clear description of the issue]

## Steps to Reproduce
1. Launch app
2. Connect to device X
3. Tap "Analyze"
4. Observe error

## Expected Behavior
[What should happen]

## Actual Behavior
[What actually happens]

## Environment
- Device: iPhone 13
- OS: iOS 17.2
- App Version: 1.0.5 (build 23)
- Backend: FastMODA v1.2

## Logs
[Attach relevant logs]

## Screenshots
[If applicable]
```

---

## 11. Success Criteria Summary

| Category | Metric | Target |
|----------|--------|--------|
| **Unit Tests** | Coverage | >85% (critical), >70% (overall) |
| **Widget Tests** | Coverage | All major screens |
| **Integration Tests** | Pass Rate | 100% |
| **Performance** | Cold Start | <2s |
| **Performance** | Memory | <150MB |
| **Performance** | Battery | <5%/hour |
| **Cross-Platform** | Device Coverage | 10+ devices |
| **CI/CD** | Build Time | <15 minutes |

---

## 12. Timeline

| Phase | Duration | Tests Implemented |
|-------|----------|-------------------|
| **Phase 1 (Weeks 1-4)** | 4 weeks | Unit tests (BLE, models) |
| **Phase 2 (Weeks 5-8)** | 4 weeks | Widget tests, API integration tests |
| **Phase 3 (Weeks 9-12)** | 4 weeks | E2E tests, performance tests |
| **Phase 4 (Weeks 13-16)** | 4 weeks | Cross-device testing, regression suite |

---

**Document Version:** 1.0
**Last Updated:** 2026-03-04
**Maintained by:** MODA QA Team
