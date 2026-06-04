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
      _resampler = AudioResampler(
          inputRate: kAudioCaptureRate.toDouble(), outputRate: fs);
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
