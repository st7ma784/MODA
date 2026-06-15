import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:sqflite/sqflite.dart';
import 'package:path/path.dart';

class AnalysisRecord {
  final int? id;
  final String taskId;
  final String analysisType;
  final DateTime timestamp;
  final double samplingRate;
  final int signalLength;
  final Map<String, dynamic> scalars;
  final List<dynamic>? frequencySummary;
  final Map<String, dynamic>? surrogateStats;
  final bool gpuUsed;
  final String? deviceId;
  final String? serverRecordingId;
  final bool uploaded;

  const AnalysisRecord({
    this.id,
    required this.taskId,
    required this.analysisType,
    required this.timestamp,
    required this.samplingRate,
    required this.signalLength,
    required this.scalars,
    this.frequencySummary,
    this.surrogateStats,
    required this.gpuUsed,
    this.deviceId,
    this.serverRecordingId,
    this.uploaded = false,
  });

  factory AnalysisRecord.fromResult(
    String taskId,
    String analysisType,
    Map<String, dynamic> status,
    double samplingRate,
    int signalLength,
  ) {
    final results = status['results'] as Map<String, dynamic>? ?? {};
    final scalars = Map<String, dynamic>.fromEntries(
      results.entries.where((e) {
        final v = e.value;
        return (v is num || v is bool) ||
            (v is String && v.length < 120 && !v.startsWith('{') && !v.startsWith('['));
      }),
    );
    return AnalysisRecord(
      taskId: taskId,
      analysisType: analysisType,
      timestamp: DateTime.now(),
      samplingRate: samplingRate,
      signalLength: signalLength,
      scalars: scalars,
      frequencySummary: results['frequency_summary'] as List?,
      surrogateStats: results['surrogate_stats'] as Map<String, dynamic>?,
      gpuUsed: results['gpu_used'] == true,
    );
  }

  Map<String, dynamic> toMap() => {
        'task_id': taskId,
        'analysis_type': analysisType,
        'timestamp': timestamp.millisecondsSinceEpoch,
        'sampling_rate': samplingRate,
        'signal_length': signalLength,
        'scalars': jsonEncode(scalars),
        'frequency_summary': frequencySummary != null
            ? jsonEncode(frequencySummary)
            : null,
        'surrogate_stats': surrogateStats != null
            ? jsonEncode(surrogateStats)
            : null,
        'gpu_used': gpuUsed ? 1 : 0,
        'device_id': deviceId,
        'server_recording_id': serverRecordingId,
        'uploaded': uploaded ? 1 : 0,
      };

  factory AnalysisRecord.fromMap(Map<String, dynamic> m) => AnalysisRecord(
        id: m['id'] as int?,
        taskId: m['task_id'] as String,
        analysisType: m['analysis_type'] as String,
        timestamp: DateTime.fromMillisecondsSinceEpoch(m['timestamp'] as int),
        samplingRate: (m['sampling_rate'] as num).toDouble(),
        signalLength: m['signal_length'] as int,
        scalars: m['scalars'] != null
            ? Map<String, dynamic>.from(jsonDecode(m['scalars'] as String))
            : {},
        frequencySummary: m['frequency_summary'] != null
            ? List.from(jsonDecode(m['frequency_summary'] as String))
            : null,
        surrogateStats: m['surrogate_stats'] != null
            ? Map<String, dynamic>.from(jsonDecode(m['surrogate_stats'] as String))
            : null,
        gpuUsed: (m['gpu_used'] as int) == 1,
        deviceId: m['device_id'] as String?,
        serverRecordingId: m['server_recording_id'] as String?,
        uploaded: (m['uploaded'] as int?) == 1,
      );

  AnalysisRecord copyWith({
    String? deviceId,
    String? serverRecordingId,
    bool? uploaded,
  }) => AnalysisRecord(
        id: id,
        taskId: taskId,
        analysisType: analysisType,
        timestamp: timestamp,
        samplingRate: samplingRate,
        signalLength: signalLength,
        scalars: scalars,
        frequencySummary: frequencySummary,
        surrogateStats: surrogateStats,
        gpuUsed: gpuUsed,
        deviceId: deviceId ?? this.deviceId,
        serverRecordingId: serverRecordingId ?? this.serverRecordingId,
        uploaded: uploaded ?? this.uploaded,
      );
}

class AnalysisHistoryService extends ChangeNotifier {
  Database? _db;
  List<AnalysisRecord> _records = [];

  List<AnalysisRecord> get records => List.unmodifiable(_records);

  Future<void> init() async {
    final dbPath = await getDatabasesPath();
    _db = await openDatabase(
      join(dbPath, 'moda_history.db'),
      version: 2,
      onCreate: (db, _) async {
        await db.execute('''
          CREATE TABLE analysis_results (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id         TEXT NOT NULL,
            analysis_type   TEXT NOT NULL,
            timestamp       INTEGER NOT NULL,
            sampling_rate   REAL,
            signal_length   INTEGER,
            scalars         TEXT,
            frequency_summary TEXT,
            surrogate_stats TEXT,
            gpu_used        INTEGER DEFAULT 0,
            device_id       TEXT,
            server_recording_id TEXT,
            uploaded        INTEGER DEFAULT 0
          )
        ''');
      },
      onUpgrade: (db, oldVersion, newVersion) async {
        if (oldVersion < 2) {
          await db.execute('ALTER TABLE analysis_results ADD COLUMN device_id TEXT');
          await db.execute('ALTER TABLE analysis_results ADD COLUMN server_recording_id TEXT');
          await db.execute('ALTER TABLE analysis_results ADD COLUMN uploaded INTEGER DEFAULT 0');
        }
      },
    );
    await _reload();
  }

  Future<void> save(AnalysisRecord record) async {
    if (_db == null) return;
    await _db!.insert('analysis_results', record.toMap());
    await _reload();
  }

  /// Records that a local analysis result has been uploaded to the FastMODA
  /// server as [serverRecordingId] for [deviceId].
  Future<void> markUploaded(int id, String deviceId, String serverRecordingId) async {
    if (_db == null) return;
    await _db!.update(
      'analysis_results',
      {'device_id': deviceId, 'server_recording_id': serverRecordingId, 'uploaded': 1},
      where: 'id = ?',
      whereArgs: [id],
    );
    await _reload();
  }

  Future<void> delete(int id) async {
    if (_db == null) return;
    await _db!.delete('analysis_results', where: 'id = ?', whereArgs: [id]);
    await _reload();
  }

  Future<void> clearAll() async {
    if (_db == null) return;
    await _db!.delete('analysis_results');
    await _reload();
  }

  Future<void> _reload() async {
    if (_db == null) return;
    final rows = await _db!.query('analysis_results',
        orderBy: 'timestamp DESC', limit: 100);
    _records = rows.map(AnalysisRecord.fromMap).toList();
    notifyListeners();
  }
}
