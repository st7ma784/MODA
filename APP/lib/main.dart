import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'screens/home.dart';
import 'services/ble_service.dart';
import 'services/audio_capture_service.dart';
import 'services/fastmoda_client.dart';
import 'services/signal_service.dart';
import 'services/analysis_history_service.dart';
import 'services/app_settings.dart';

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
        theme: _buildTheme(),
        home: const HomeScreen(),
        debugShowCheckedModeBanner: false,
      ),
    );
  }

  ThemeData _buildTheme() {
    const primary = Color(0xFF00BCD4);
    const secondary = Color(0xFF1DE9B6);
    const surface = Color(0xFF1E1E2E);
    const background = Color(0xFF12121F);

    return ThemeData(
      colorScheme: const ColorScheme.dark(
        primary: primary,
        secondary: secondary,
        surface: surface,
        onPrimary: Colors.black,
        onSecondary: Colors.black,
      ),
      scaffoldBackgroundColor: background,
      useMaterial3: true,
      cardTheme: const CardThemeData(
        color: surface,
        elevation: 0,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.all(Radius.circular(12)),
          side: BorderSide(color: Colors.white10),
        ),
      ),
      navigationBarTheme: const NavigationBarThemeData(
        backgroundColor: surface,
        indicatorColor: Color(0x3300BCD4),
      ),
      appBarTheme: const AppBarTheme(
        backgroundColor: background,
        foregroundColor: Colors.white,
        surfaceTintColor: Colors.transparent,
      ),
      tabBarTheme: const TabBarThemeData(
        labelColor: primary,
        unselectedLabelColor: Colors.white38,
        indicatorColor: primary,
      ),
    );
  }
}
