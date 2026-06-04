# Microphone Input Source — Design

**Date:** 2026-06-04
**Status:** Approved
**Component:** Flutter app (`APP/`)

## Goal

Let the MODA app capture signal from the phone's **microphone** — built-in or a
USB-C wired mic — as an alternative to the existing Bluetooth (BLE) sensor source.
Primary motivation: easier testing without BLE hardware. MODA is a general
signal-analysis tool, so the capture rate must stay configurable rather than
hard-coded to EEG assumptions.

## Background

Today the only signal source is `BleService.sampleStream`, wired into
`SignalService.bindBleStream` → `SignalService.addSamples(List<double>)`, which is
the single ingestion point feeding the 512-sample ring buffer, DFT, EEG band
powers, changepoints, and server submissions. The pipeline `sampleRate` is already
user-configurable (Settings → `flutter_secure_storage`) and drives both the local
DFT and FastMODA submissions.

USB-C note: on Android, plugging in a USB-C mic makes it the **default audio input
automatically**. "Built-in mic" and "USB-C mic" are therefore the same capture
code path — the OS routes to whichever is connected. No device-picker UI is needed.

Hardware constraint: Android `AudioRecord` (under the `record` package) only
captures at hardware sample rates (8k/16k/44.1k/48k). The configured pipeline rate
(e.g. 256 Hz) is **not** a valid capture rate, so mic audio is captured at a
supported rate and resampled down to the pipeline rate.

## Architecture

Mirror the existing `BleService` shape so the two sources are interchangeable.

### New: `lib/services/audio_capture_service.dart`

`ChangeNotifier`, same public surface idiom as `BleService`:

- Wraps the `record` package (v5+) `startStream()` → `Stream<Uint8List>` of **mono
  PCM16** at a fixed hardware capture rate `kAudioCaptureRate` (default **16000 Hz**;
  universally supported).
- Converts PCM16 little-endian bytes → normalized doubles in −1.0…1.0.
- **Resamples** from the capture rate down to the configured pipeline `sampleRate`
  using fractional block-averaging (handles non-integer ratios; carries a
  fractional accumulator across chunks so no samples are dropped at chunk
  boundaries).
- Public API:
  - `Future<bool> start()` — requests `Permission.microphone`, starts the stream;
    emits a friendly message on the error stream and returns `false` on
    denial/failure.
  - `Future<void> stop()` — stops capture, releases the mic.
  - `bool get isCapturing`
  - `Stream<List<double>> get sampleStream` — broadcast, resampled doubles.
  - `Stream<String> get errors` — broadcast, snackbar-friendly (same pattern as
    `BleService.errors`).
  - `set targetSampleRate(double)` — the resample target; kept in sync with the
    pipeline `sampleRate`.

### Changed: `lib/services/signal_service.dart`

Generalize the single active subscription so exactly one source feeds the buffer:

- Add `enum InputSource { bluetooth, microphone }` and `InputSource get activeSource`
  (default `bluetooth`).
- Keep `bindBleStream`; add `bindAudioStream(Stream<List<double>>)`.
- Rename internal `_bleSub` → `_inputSub`. Hold references to both bound streams;
  subscribe only to the active one.
- `void setInputSource(InputSource source)` — cancels the current subscription,
  clears the ring buffer (`_head`/`_total`/snapshot reset) so the chart doesn't
  splice two different signals, subscribes to the new source, `notifyListeners()`.

### Changed: `lib/screens/home.dart`

- Register `AudioCaptureService` in the `MultiProvider`.
- In `_initServices`: bind both streams (`signal.bindBleStream(ble.sampleStream)`,
  `signal.bindAudioStream(audio.sampleStream)`), keep `sampleRate` propagated to
  both `SignalService` and `AudioCaptureService.targetSampleRate`.
- Forward `audio.errors` to the existing snackbar handler.
- On source switch (from the Dashboard selector): call `audio.start()` when
  switching to microphone (revert the selector to Bluetooth if it returns false),
  `audio.stop()` when switching away.

### Changed: `lib/screens/dashboard_screen.dart`

- A compact `SegmentedButton<InputSource>` ("🎙 Mic | Bluetooth") near the top,
  reflecting and driving `signal.activeSource` via the source-switch handler.

### Changed: `lib/screens/settings_screen.dart`

- Add a one-line caption under the existing "Sampling Rate" field noting it is also
  the microphone resample target. No new field.

### Permissions / config

- `android/app/src/main/AndroidManifest.xml`: add
  `<uses-permission android:name="android.permission.RECORD_AUDIO"/>`.
- `android/app/build.gradle`: bump `minSdkVersion` 21 → 23 (required by `record`;
  drops only Android 5.x).
- iOS `ios/Runner/Info.plist`: add `NSMicrophoneUsageDescription`.
- `pubspec.yaml`: add `record: ^5.x`.

## Data flow

```
Microphone (built-in or USB-C, OS default input)
  → record.startStream()  : Stream<Uint8List> PCM16 @ 16 kHz
  → AudioCaptureService    : bytes → doubles(−1..1) → resample to pipeline rate
  → sampleStream           : Stream<List<double>>
  → SignalService.addSamples (when activeSource == microphone)
  → ring buffer → DFT / bands / changepoints / server submission
```

BLE path is unchanged; the two are mutually exclusive via `activeSource`.

## Error handling

- Mic permission denied or `start()` failure → friendly snackbar via the existing
  error-stream → `home.dart` `_showError` path; the Dashboard selector reverts to
  Bluetooth.
- Switching sources clears the buffer so stale samples from the previous source
  don't corrupt the live view or the next analysis submission.

## Testing

`test/services/audio_capture_service_test.dart`:
- PCM16 little-endian bytes → normalized doubles (sign, scale by 32768).
- Resampler: 16 kHz → 256 Hz produces the expected output sample count and
  block-averaged values; fractional accumulator preserves count across chunk
  boundaries.

`test/services/signal_service_test.dart` (extend existing):
- `setInputSource` swaps subscriptions, clears the buffer, and only the active
  source feeds `addSamples`.

## Out of scope (YAGNI)

- Explicit input-device picker / routing (system default covers built-in + USB-C).
- Simultaneous mic + BLE capture as separate channels.
- Stereo / multi-channel audio capture.
- Release keystore signing (separate concern; debug-signed APK is fine for testing).
