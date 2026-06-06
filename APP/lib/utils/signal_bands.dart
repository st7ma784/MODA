import 'package:flutter/material.dart';
import '../services/signal_service.dart' show SignalType;

/// Metadata for one frequency band, with separate Hz boundaries for each mode.
class BandDef {
  /// Key used in SignalService.bandPowers map.
  final String key;
  final String eegName;   // e.g. 'Delta'
  final String genName;   // e.g. 'VLF'
  final double eegLow, eegHigh;
  final double genLow, genHigh;
  final Color color;

  const BandDef({
    required this.key,
    required this.eegName,
    required this.genName,
    required this.eegLow,
    required this.eegHigh,
    required this.genLow,
    required this.genHigh,
    required this.color,
  });

  String label(SignalType type) => type == SignalType.eeg ? eegName : genName;
  double low(SignalType type)   => type == SignalType.eeg ? eegLow  : genLow;
  double high(SignalType type)  => type == SignalType.eeg ? eegHigh : genHigh;

  String hz(SignalType type) {
    final lo = low(type);
    final hi = high(type);
    final loStr = lo % 1 == 0 ? lo.toInt().toString() : lo.toStringAsFixed(1);
    return '$loStr–${hi.toInt()} Hz';
  }
}

/// EEG: classic Greek-letter bands.
/// Generic: equal-octave bands scaled to a 256 Hz (Nyquist 128 Hz) signal.
const List<BandDef> kBands = [
  BandDef(
    key: 'delta', eegName: 'Delta', genName: 'VLF',
    eegLow: 0.5, eegHigh: 4.0,
    genLow: 0.5, genHigh: 4.0,
    color: Colors.purple,
  ),
  BandDef(
    key: 'theta', eegName: 'Theta', genName: 'LF',
    eegLow: 4.0, eegHigh: 8.0,
    genLow: 4.0, genHigh: 16.0,
    color: Colors.blue,
  ),
  BandDef(
    key: 'alpha', eegName: 'Alpha', genName: 'MF',
    eegLow: 8.0, eegHigh: 12.0,
    genLow: 16.0, genHigh: 32.0,
    color: Colors.teal,
  ),
  BandDef(
    key: 'beta', eegName: 'Beta', genName: 'HF',
    eegLow: 12.0, eegHigh: 30.0,
    genLow: 32.0, genHigh: 64.0,
    color: Colors.orange,
  ),
  BandDef(
    key: 'gamma', eegName: 'Gamma', genName: 'VHF',
    eegLow: 30.0, eegHigh: 100.0,
    genLow: 64.0, genHigh: 128.0,
    color: Colors.red,
  ),
];

/// Returns the band whose range contains [freq] for the given [type], or null.
BandDef? bandForFreq(double freq, SignalType type) {
  for (final b in kBands) {
    if (freq >= b.low(type) && freq < b.high(type)) return b;
  }
  return null;
}
