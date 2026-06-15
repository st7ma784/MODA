import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'screens/home.dart';
import 'services/ble_service.dart';
import 'services/audio_capture_service.dart';
import 'services/fastmoda_client.dart';
import 'services/signal_service.dart';
import 'services/analysis_history_service.dart';
import 'services/app_settings.dart';
import 'theme/app_theme.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final history = AnalysisHistoryService();
  await history.init();
  runApp(ModaApp(history: history));
}

class ModaApp extends StatelessWidget {
  final AnalysisHistoryService history;
  const ModaApp({super.key, required this.history});

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => BleService()),
        ChangeNotifierProvider(create: (_) => SignalService()),
        ChangeNotifierProvider(create: (_) => AudioCaptureService()),
        Provider(create: (_) => FastModaClient()),
        Provider(create: (_) => AppSettings()),
        ChangeNotifierProvider<AnalysisHistoryService>.value(value: history),
      ],
      child: MaterialApp(
        title: 'MODA',
        theme: AppTheme.theme,
        home: const HomeScreen(),
        debugShowCheckedModeBanner: false,
      ),
    );
  }
}
