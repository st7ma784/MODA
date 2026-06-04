# Microphone Input Source Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the MODA app capture signal from the phone's built-in or USB-C microphone as a selectable alternative to the BLE sensor, resampling mic audio into the existing analysis pipeline.

**Architecture:** A new `AudioCaptureService` (mirroring `BleService`) wraps the `record` package's PCM16 stream, converts bytes to normalized doubles, and resamples from the hardware capture rate (16 kHz) down to the configured pipeline `sampleRate` using a pure, testable resampler. `SignalService` gains an active-source switch so exactly one source feeds `addSamples` at a time. A `SegmentedButton` on the Dashboard drives the switch.

**Tech Stack:** Flutter/Dart, `record` ^5.x, `provider`, `flutter_test`.

> **Prerequisite:** A working Flutter toolchain (`flutter doctor` clean) is required to run the `flutter test` / `flutter analyze` steps. Code can be written without it, but verification needs it.

---

### Task 1: Add dependency, permissions, and capture-rate constant

**Files:**
- Modify: `APP/pubspec.yaml`
- Modify: `APP/android/app/src/main/AndroidManifest.xml`
- Modify: `APP/android/app/build.gradle`
- Modify: `APP/ios/Runner/Info.plist`
- Modify: `APP/lib/config/app_config.dart`

- [ ] **Step 1: Add the `record` dependency**

In `APP/pubspec.yaml`, under the existing `# File import` block (after `file_picker`), add:

```yaml
  # Microphone capture (built-in / USB-C mic as a signal source)
  record: ^5.1.2
```

- [ ] **Step 2: Declare the RECORD_AUDIO permission (Android)**

In `APP/android/app/src/main/AndroidManifest.xml`, immediately after the `BLUETOOTH_CONNECT` permission line, add:

```xml
    <!-- ── Microphone: built-in or USB-C mic as a signal source ── -->
    <uses-permission android:name="android.permission.RECORD_AUDIO"/>
```

- [ ] **Step 3: Bump minSdkVersion to 23 (required by `record`)**

In `APP/android/app/build.gradle`, change:

```gradle
        minSdkVersion 21          // flutter_blue_plus minimum; also covers flutter_secure_storage
```

to:

```gradle
        minSdkVersion 23          // record (mic capture) minimum; also covers flutter_blue_plus / flutter_secure_storage
```

- [ ] **Step 4: Add the iOS microphone usage description**

In `APP/ios/Runner/Info.plist`, add the following key just before the final `</dict>` (create the file's key if the project lacks it; otherwise insert alongside existing keys):

```xml
	<key>NSMicrophoneUsageDescription</key>
	<string>MODA uses the microphone to capture signal for analysis.</string>
```

- [ ] **Step 5: Add the capture-rate constant**

In `APP/lib/config/app_config.dart`, after the line `const int kSignalBufferSize = 512;`, add:

```dart

// Hardware sample rate for microphone capture. AudioRecord only supports
// standard rates (8k/16k/44.1k/48k); 16 kHz is universally available and is
// resampled down to the configured pipeline rate. This is the effective
// ceiling for the mic source's sample rate.
const int kAudioCaptureRate = 16000;
```

- [ ] **Step 6: Fetch packages**

Run: `cd APP && flutter pub get`
Expected: resolves with `record` added, no errors.

- [ ] **Step 7: Commit**

```bash
git add APP/pubspec.yaml APP/pubspec.lock APP/android/app/src/main/AndroidManifest.xml APP/android/app/build.gradle APP/ios/Runner/Info.plist APP/lib/config/app_config.dart
git commit -m "feat(audio): add record dependency, mic permission, capture-rate constant"
```

---

### Task 2: Pure PCM-conversion + resampler utility (TDD)

A pure-Dart unit, independent of the `record` plugin, so it is fully unit-testable.

**Files:**
- Create: `APP/lib/utils/audio_resampler.dart`
- Test: `APP/test/utils/audio_resampler_test.dart`

- [ ] **Step 1: Write the failing test**

Create `APP/test/utils/audio_resampler_test.dart`:

```dart
import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:moda_mobile/utils/audio_resampler.dart';

void main() {
  group('pcm16ToDoubles', () {
    test('decodes little-endian signed PCM16 into normalized doubles', () {
      // 0x0000 = 0, 0x7FFF = +32767, 0x8000 = -32768
      final bytes = Uint8List.fromList([0x00, 0x00, 0xFF, 0x7F, 0x00, 0x80]);
      final out = pcm16ToDoubles(bytes);
      expect(out.length, 3);
      expect(out[0], 0.0);
      expect(out[1], closeTo(32767 / 32768, 1e-9));
      expect(out[2], -1.0);
    });

    test('ignores a trailing odd byte', () {
      final bytes = Uint8List.fromList([0x00, 0x00, 0x11]);
      expect(pcm16ToDoubles(bytes).length, 1);
    });
  });

  group('AudioResampler', () {
    test('16 kHz -> 256 Hz yields 256 averaged samples for 16000 inputs', () {
      final r = AudioResampler(inputRate: 16000, outputRate: 256);
      final out = r.process(List<double>.filled(16000, 1.0));
      expect(out.length, 256);
      expect(out.every((v) => (v - 1.0).abs() < 1e-9), isTrue);
    });

    test('preserves output count across chunk boundaries', () {
      final r = AudioResampler(inputRate: 16000, outputRate: 256);
      final a = r.process(List<double>.filled(8000, 1.0));
      final b = r.process(List<double>.filled(8000, 1.0));
      expect(a.length + b.length, 256);
    });

    test('passes samples through when output rate >= input rate', () {
      final r = AudioResampler(inputRate: 16000, outputRate: 16000);
      final out = r.process([1.0, 2.0, 3.0]);
      expect(out, [1.0, 2.0, 3.0]);
    });
  });
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd APP && flutter test test/utils/audio_resampler_test.dart`
Expected: FAIL — `Target of URI doesn't exist: 'package:moda_mobile/utils/audio_resampler.dart'`.

- [ ] **Step 3: Write the implementation**

Create `APP/lib/utils/audio_resampler.dart`:

```dart
import 'dart:typed_data';

/// Decode little-endian signed 16-bit PCM bytes into normalized doubles
/// in the range −1.0…1.0. A trailing odd byte (incomplete sample) is ignored.
List<double> pcm16ToDoubles(Uint8List bytes) {
  final n = bytes.length ~/ 2;
  final out = List<double>.filled(n, 0.0);
  final bd = ByteData.view(bytes.buffer, bytes.offsetInBytes, n * 2);
  for (var i = 0; i < n; i++) {
    out[i] = bd.getInt16(i * 2, Endian.little) / 32768.0;
  }
  return out;
}

/// Streaming decimator that averages blocks of input samples down to a lower
/// output rate. Carries a fractional budget across [process] calls so no
/// samples are lost at chunk boundaries. When [outputRate] >= [inputRate] the
/// input is passed through unchanged (no upsampling).
class AudioResampler {
  final double _ratio; // input samples consumed per output sample
  double _budget;
  double _acc = 0.0;
  int _count = 0;

  AudioResampler({required double inputRate, required double outputRate})
      : _ratio = outputRate >= inputRate ? 1.0 : inputRate / outputRate,
        _budget = outputRate >= inputRate ? 1.0 : inputRate / outputRate;

  List<double> process(List<double> input) {
    final out = <double>[];
    for (final s in input) {
      _acc += s;
      _count++;
      _budget -= 1.0;
      if (_budget <= 0.0) {
        out.add(_acc / _count);
        _acc = 0.0;
        _count = 0;
        _budget += _ratio;
      }
    }
    return out;
  }
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd APP && flutter test test/utils/audio_resampler_test.dart`
Expected: PASS (all 5 tests).

- [ ] **Step 5: Commit**

```bash
git add APP/lib/utils/audio_resampler.dart APP/test/utils/audio_resampler_test.dart
git commit -m "feat(audio): PCM16 decode + streaming resampler utility"
```

---

### Task 3: AudioCaptureService wrapping `record`

Wraps the plugin and the Task 2 utility. The plugin can't run under unit tests, so tests cover only the non-plugin surface (defaults, target-rate setter, dispose). The `start()`/`stop()` paths are verified manually on-device in Task 8.

**Files:**
- Create: `APP/lib/services/audio_capture_service.dart`
- Test: `APP/test/services/audio_capture_service_test.dart`

- [ ] **Step 1: Write the failing test**

Create `APP/test/services/audio_capture_service_test.dart`:

```dart
import 'package:flutter_test/flutter_test.dart';
import 'package:moda_mobile/services/audio_capture_service.dart';

void main() {
  group('AudioCaptureService', () {
    test('is not capturing on construction', () {
      expect(AudioCaptureService().isCapturing, isFalse);
    });

    test('targetSampleRate defaults to 256 and is updatable', () {
      final svc = AudioCaptureService();
      expect(svc.targetSampleRate, 256.0);
      svc.targetSampleRate = 100.0;
      expect(svc.targetSampleRate, 100.0);
    });

    test('exposes a broadcast sampleStream', () {
      final svc = AudioCaptureService();
      expect(svc.sampleStream.isBroadcast, isTrue);
    });

    test('dispose does not throw', () {
      expect(() => AudioCaptureService().dispose(), returnsNormally);
    });
  });
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd APP && flutter test test/services/audio_capture_service_test.dart`
Expected: FAIL — URI for `audio_capture_service.dart` doesn't exist.

- [ ] **Step 3: Write the implementation**

Create `APP/lib/services/audio_capture_service.dart`:

```dart
import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:record/record.dart';
import '../config/app_config.dart';
import '../utils/audio_resampler.dart';

/// Captures mono PCM16 audio from the OS default input (built-in mic, or a
/// USB-C mic when one is plugged in) at [kAudioCaptureRate], resamples it down
/// to [targetSampleRate], and emits normalized doubles on [sampleStream].
/// Mirrors the BleService surface so SignalService can treat the two
/// interchangeably.
class AudioCaptureService extends ChangeNotifier {
  final AudioRecorder _recorder = AudioRecorder();

  final _sampleController = StreamController<List<double>>.broadcast();
  final _errorController = StreamController<String>.broadcast();

  StreamSubscription<Uint8List>? _audioSub;
  AudioResampler? _resampler;
  bool _capturing = false;
  double _targetSampleRate = kDefaultSampleRate;

  /// Resampled signal samples (−1.0…1.0) at [targetSampleRate].
  Stream<List<double>> get sampleStream => _sampleController.stream;

  /// Snackbar-friendly errors. Broadcast; subscribe once.
  Stream<String> get errors => _errorController.stream;

  bool get isCapturing => _capturing;

  double get targetSampleRate => _targetSampleRate;
  set targetSampleRate(double fs) {
    _targetSampleRate = fs;
    if (_capturing) {
      _resampler =
          AudioResampler(inputRate: kAudioCaptureRate.toDouble(), outputRate: fs);
    }
    notifyListeners();
  }

  /// Requests mic permission and starts streaming. Returns false (and emits a
  /// friendly error) if permission is denied or capture fails to start.
  Future<bool> start() async {
    if (_capturing) return true;
    try {
      if (!await _recorder.hasPermission()) {
        _errorController.add(
          'Microphone permission denied. '
          'Go to Settings → Apps → MODA → Permissions to enable it.',
        );
        return false;
      }
      _resampler = AudioResampler(
        inputRate: kAudioCaptureRate.toDouble(),
        outputRate: _targetSampleRate,
      );
      final stream = await _recorder.startStream(
        const RecordConfig(
          encoder: AudioEncoder.pcm16bits,
          sampleRate: kAudioCaptureRate,
          numChannels: 1,
        ),
      );
      _audioSub = stream.listen(_onBytes, onError: (e) {
        _errorController.add('Microphone error: ${_friendly(e)}');
      });
      _capturing = true;
      notifyListeners();
      return true;
    } catch (e) {
      _errorController.add('Could not start microphone: ${_friendly(e)}');
      return false;
    }
  }

  Future<void> stop() async {
    await _audioSub?.cancel();
    _audioSub = null;
    try {
      await _recorder.stop();
    } catch (_) {}
    _resampler = null;
    _capturing = false;
    notifyListeners();
  }

  void _onBytes(Uint8List bytes) {
    if (bytes.isEmpty) return;
    final samples = pcm16ToDoubles(bytes);
    final resampled = _resampler?.process(samples) ?? samples;
    if (resampled.isNotEmpty) _sampleController.add(resampled);
  }

  static String _friendly(Object e) => e.toString().split('\n').first;

  @override
  void dispose() {
    _audioSub?.cancel();
    _recorder.dispose();
    _sampleController.close();
    _errorController.close();
    super.dispose();
  }
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd APP && flutter test test/services/audio_capture_service_test.dart`
Expected: PASS (4 tests). Note: `AudioRecorder()` constructs without touching the platform channel, so the no-capture tests run fine under the VM.

- [ ] **Step 5: Commit**

```bash
git add APP/lib/services/audio_capture_service.dart APP/test/services/audio_capture_service_test.dart
git commit -m "feat(audio): AudioCaptureService streaming mic into the pipeline"
```

---

### Task 4: Source switching in SignalService (TDD)

**Files:**
- Modify: `APP/lib/services/signal_service.dart`
- Test: `APP/test/services/signal_service_test.dart` (extend)

- [ ] **Step 1: Write the failing test**

Append this group inside `main()` in `APP/test/services/signal_service_test.dart` (before the closing `}`):

```dart
  group('SignalService — input source switching', () {
    test('defaults to bluetooth source', () {
      expect(SignalService().activeSource, InputSource.bluetooth);
    });

    test('only the active source feeds the buffer', () async {
      final svc = SignalService();
      final ble = StreamController<List<double>>.broadcast();
      final mic = StreamController<List<double>>.broadcast();
      svc.bindBleStream(ble.stream);
      svc.bindAudioStream(mic.stream);

      ble.add([1.0, 2.0]);
      mic.add([9.0]); // ignored while bluetooth is active
      await Future<void>.delayed(Duration.zero);
      expect(svc.recentSamples, [1.0, 2.0]);

      svc.setInputSource(InputSource.microphone);
      expect(svc.recentSamples, isEmpty); // buffer cleared on switch

      mic.add([7.0, 8.0]);
      ble.add([5.0]); // now ignored
      await Future<void>.delayed(Duration.zero);
      expect(svc.recentSamples, [7.0, 8.0]);

      await ble.close();
      await mic.close();
    });

    test('setInputSource notifies listeners and is a no-op when unchanged',
        () {
      final svc = SignalService();
      var fired = 0;
      svc.addListener(() => fired++);
      svc.setInputSource(InputSource.bluetooth); // unchanged → no notify
      expect(fired, 0);
      svc.setInputSource(InputSource.microphone);
      expect(fired, 1);
    });
  });
```

Add `import 'dart:async';` at the top of the test file if not already present.

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd APP && flutter test test/services/signal_service_test.dart`
Expected: FAIL — `InputSource` / `activeSource` / `bindAudioStream` / `setInputSource` undefined.

- [ ] **Step 3: Implement the source switch**

In `APP/lib/services/signal_service.dart`:

(a) Add the enum just below the existing `enum ServerStatus { ... }` line (line 9):

```dart
enum InputSource { bluetooth, microphone }
```

(b) Replace the field declaration `StreamSubscription<List<double>>? _bleSub;` (around line 93) with:

```dart
  Stream<List<double>>? _bleStream;
  Stream<List<double>>? _audioStream;
  StreamSubscription<List<double>>? _inputSub;
  InputSource _activeSource = InputSource.bluetooth;
```

(c) Add a getter next to the other public getters (e.g. after `List<double> get recentSamples => _recentSamplesSnapshot;`):

```dart
  InputSource get activeSource => _activeSource;
```

(d) Replace the existing `bindBleStream` method:

```dart
  void bindBleStream(Stream<List<double>> stream) {
    _bleSub?.cancel();
    _bleSub = stream.listen(addSamples);
  }
```

with:

```dart
  void bindBleStream(Stream<List<double>> stream) {
    _bleStream = stream;
    if (_activeSource == InputSource.bluetooth) _resubscribe();
  }

  void bindAudioStream(Stream<List<double>> stream) {
    _audioStream = stream;
    if (_activeSource == InputSource.microphone) _resubscribe();
  }

  /// Switch which bound source feeds the buffer. Clears the buffer so the live
  /// view and the next submission don't splice two different signals.
  void setInputSource(InputSource source) {
    if (source == _activeSource) return;
    _activeSource = source;
    _clearBuffer();
    _resubscribe();
    notifyListeners();
  }

  void _resubscribe() {
    _inputSub?.cancel();
    final stream =
        _activeSource == InputSource.bluetooth ? _bleStream : _audioStream;
    _inputSub = stream?.listen(addSamples);
  }

  void _clearBuffer() {
    _head = 0;
    _total = 0;
    _recentSamplesSnapshot = const [];
    _npyCache.clear();
    _changepoints = [];
    _changepointsCache = null;
    _lastChangepointTotal = 0;
    notifyListeners();
  }
```

(e) In `dispose()`, replace `_bleSub?.cancel();` with `_inputSub?.cancel();`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd APP && flutter test test/services/signal_service_test.dart`
Expected: PASS (existing tests + 3 new).

- [ ] **Step 5: Commit**

```bash
git add APP/lib/services/signal_service.dart APP/test/services/signal_service_test.dart
git commit -m "feat(signal): switchable BLE/mic input source"
```

---

### Task 5: Register provider and wire streams in main + home

**Files:**
- Modify: `APP/lib/main.dart`
- Modify: `APP/lib/screens/home.dart`

- [ ] **Step 1: Register the provider**

In `APP/lib/main.dart`, add the import after the `ble_service.dart` import:

```dart
import 'services/audio_capture_service.dart';
```

and add this provider inside the `providers: [ ... ]` list, after the `SignalService` provider:

```dart
        ChangeNotifierProvider(create: (_) => AudioCaptureService()),
```

- [ ] **Step 2: Wire the audio stream and errors in home**

In `APP/lib/screens/home.dart`, add the import after `ble_service.dart`:

```dart
import '../services/audio_capture_service.dart';
```

Add a field next to the other subscription fields (after `_signalErrorSub`):

```dart
  StreamSubscription<String>? _audioErrorSub;
```

In `_initServices`, after `final signal = context.read<SignalService>();`, add:

```dart
    final audio = context.read<AudioCaptureService>();
```

After the existing `signal.bindBleStream(ble.sampleStream);` line, add:

```dart
    signal.bindAudioStream(audio.sampleStream);
    audio.targetSampleRate = fs;
```

After `_signalErrorSub = signal.errors.listen(_showError);`, add:

```dart
    _audioErrorSub = audio.errors.listen(_showError);
```

In `dispose()`, after `_signalErrorSub?.cancel();`, add:

```dart
    _audioErrorSub?.cancel();
```

- [ ] **Step 3: Verify it compiles**

Run: `cd APP && flutter analyze lib/main.dart lib/screens/home.dart`
Expected: No issues (info-level lints from the existing codebase are acceptable; no errors).

- [ ] **Step 4: Commit**

```bash
git add APP/lib/main.dart APP/lib/screens/home.dart
git commit -m "feat(audio): register AudioCaptureService and bind mic stream"
```

---

### Task 6: Dashboard source selector

**Files:**
- Modify: `APP/lib/screens/dashboard_screen.dart`

- [ ] **Step 1: Add imports and the switch handler**

In `APP/lib/screens/dashboard_screen.dart`, add the import after `signal_service.dart`:

```dart
import '../services/audio_capture_service.dart';
```

Add this top-level function at the end of the file (after the `_MetricCard` class):

```dart
/// Coordinates a source switch: starts/stops the mic and flips SignalService.
/// If the mic fails to start (e.g. permission denied), stays on the current
/// source — the error is surfaced via the AudioCaptureService error stream.
Future<void> _switchSource(BuildContext context, InputSource src) async {
  final signal = context.read<SignalService>();
  final audio = context.read<AudioCaptureService>();
  if (src == signal.activeSource) return;
  if (src == InputSource.microphone) {
    if (!await audio.start()) return;
    signal.setInputSource(InputSource.microphone);
  } else {
    await audio.stop();
    signal.setInputSource(InputSource.bluetooth);
  }
}
```

- [ ] **Step 2: Add the SegmentedButton to the Live Signal card**

In the `build` method, inside the Live Signal `Card`'s `Column` children, replace the `Row(...)` that holds the `'Live Signal'` text and pulse dot — specifically insert the selector after that `Row` and before `const SizedBox(height: 8)`. Change:

```dart
                  Row(
                    children: [
                      Text('Live Signal',
                          style: theme.textTheme.labelLarge
                              ?.copyWith(color: theme.colorScheme.primary)),
                      const Spacer(),
                      if (ble.isStreaming)
                        _PulseDot(color: theme.colorScheme.secondary),
                    ],
                  ),
                  const SizedBox(height: 8),
```

to:

```dart
                  Row(
                    children: [
                      Text('Live Signal',
                          style: theme.textTheme.labelLarge
                              ?.copyWith(color: theme.colorScheme.primary)),
                      const Spacer(),
                      if (ble.isStreaming || signal.activeSource == InputSource.microphone)
                        _PulseDot(color: theme.colorScheme.secondary),
                    ],
                  ),
                  const SizedBox(height: 8),
                  SegmentedButton<InputSource>(
                    showSelectedIcon: false,
                    style: const ButtonStyle(
                      visualDensity: VisualDensity.compact,
                    ),
                    segments: const [
                      ButtonSegment(
                        value: InputSource.bluetooth,
                        label: Text('Bluetooth'),
                        icon: Icon(Icons.bluetooth, size: 16),
                      ),
                      ButtonSegment(
                        value: InputSource.microphone,
                        label: Text('Mic'),
                        icon: Icon(Icons.mic, size: 16),
                      ),
                    ],
                    selected: {signal.activeSource},
                    onSelectionChanged: (sel) =>
                        _switchSource(context, sel.first),
                  ),
                  const SizedBox(height: 8),
```

- [ ] **Step 3: Verify it compiles**

Run: `cd APP && flutter analyze lib/screens/dashboard_screen.dart`
Expected: No errors.

- [ ] **Step 4: Commit**

```bash
git add APP/lib/screens/dashboard_screen.dart
git commit -m "feat(audio): Dashboard Bluetooth/Mic source selector"
```

---

### Task 7: Settings caption + keep mic resample target in sync

**Files:**
- Modify: `APP/lib/screens/settings_screen.dart`

- [ ] **Step 1: Sync the audio target rate when the sample rate is saved**

In `APP/lib/screens/settings_screen.dart`, add the import after `signal_service.dart`:

```dart
import '../services/audio_capture_service.dart';
```

In `_saveSampleRate`, change:

```dart
    await context.read<AppSettings>().setSampleRate(fs);
    if (mounted) context.read<SignalService>().sampleRate = fs;
```

to:

```dart
    await context.read<AppSettings>().setSampleRate(fs);
    if (mounted) {
      context.read<SignalService>().sampleRate = fs;
      context.read<AudioCaptureService>().targetSampleRate = fs;
    }
```

- [ ] **Step 2: Add the caption under the Sampling Rate field**

In the Analysis `Card`, after the `TextField` for `_fsController` and before the `const SizedBox(height: 8)` that precedes its Save button, add:

```dart
                  const SizedBox(height: 6),
                  const Text(
                    'Also the target rate microphone audio is resampled to '
                    '(captured at 16 kHz, decimated to this rate).',
                    style: TextStyle(fontSize: 11, color: Colors.white38),
                  ),
```

- [ ] **Step 3: Verify it compiles**

Run: `cd APP && flutter analyze lib/screens/settings_screen.dart`
Expected: No errors.

- [ ] **Step 4: Commit**

```bash
git add APP/lib/screens/settings_screen.dart
git commit -m "feat(audio): sync mic resample target with sample-rate setting + caption"
```

---

### Task 8: Full verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `cd APP && flutter test`
Expected: All tests pass (existing + audio_resampler + audio_capture_service + signal_service source-switch).

- [ ] **Step 2: Static analysis**

Run: `cd APP && flutter analyze`
Expected: No new errors introduced by these changes.

- [ ] **Step 3: Manual on-device check (requires a phone)**

Build and run on a connected Android phone: `cd APP && flutter run --release`. Then:
1. Open the Dashboard, tap **Mic** in the source selector. Grant the microphone permission prompt.
2. Confirm the LIVE dot appears and the Live Signal chart moves in response to sound (tap/speak near the phone).
3. Confirm Band Powers / Dominant Freq update.
4. Plug in a USB-C mic and repeat — capture should follow the USB mic automatically.
5. Tap **Bluetooth** to switch back; confirm the chart clears and mic capture stops.

- [ ] **Step 4: Final commit (if any cleanup was needed)**

```bash
git add -A && git commit -m "test(audio): verify mic input source end-to-end" || echo "nothing to commit"
```

---

## Self-Review Notes

- **Spec coverage:** AudioCaptureService (Task 3), resampling (Task 2), source switch in SignalService (Task 4), provider wiring (Task 5), Dashboard selector (Task 6), Settings caption + rate sync (Task 7), permissions/minSdk/pubspec/iOS plist (Task 1), tests (Tasks 2–4, 8). All spec sections mapped.
- **Type consistency:** `InputSource { bluetooth, microphone }`, `setInputSource`, `bindAudioStream`, `activeSource`, `targetSampleRate`, `pcm16ToDoubles`, `AudioResampler.process`, `kAudioCaptureRate` are used identically across tasks.
- **USB-C:** handled by OS default-input routing — no code path of its own, per design.
- **Out of scope:** device picker, simultaneous capture, stereo, release signing.
