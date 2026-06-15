import 'dart:async';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'dashboard_screen.dart';
import 'ble_screen.dart';
import 'analysis_screen.dart';
import 'settings_screen.dart';
import '../services/ble_service.dart';
import '../services/audio_capture_service.dart';
import '../services/fastmoda_client.dart';
import '../services/signal_service.dart';
import '../services/app_settings.dart';
import '../theme/app_theme.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  int _selectedIndex = 0;

  // Cached references so dispose() never calls context.read on a dead context.
  BleService? _ble;
  VoidCallback? _bleListener;
  StreamSubscription<String>? _bleErrorSub;
  StreamSubscription<String>? _signalErrorSub;
  StreamSubscription<String>? _audioErrorSub;

  static const _screens = <Widget>[
    DashboardScreen(),
    BleScreen(),
    AnalysisScreen(),
    SettingsScreen(),
  ];

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) => _initServices());
  }

  Future<void> _initServices() async {
    if (!mounted) return;

    final settings = context.read<AppSettings>();
    final client = context.read<FastModaClient>();
    final ble = context.read<BleService>();
    final signal = context.read<SignalService>();
    final audio = context.read<AudioCaptureService>();

    // Cache ble so dispose() can remove the listener without context.
    _ble = ble;

    final url = await settings.getServerUrl();
    if (!mounted) return;
    client.setBaseUrl(url);

    final fs = await settings.getSampleRate();
    if (!mounted) return;
    signal.sampleRate = fs;

    signal.bindBleStream(ble.sampleStream);
    signal.bindAudioStream(audio.sampleStream);
    audio.targetSampleRate = fs;
    signal.bindClient(client);

    // Forward errors from all services to snackbars.
    _bleErrorSub = ble.errors.listen(_showError);
    _signalErrorSub = signal.errors.listen(_showError);
    _audioErrorSub = audio.errors.listen(_showError);

    // When a MODA device connects and reports its sample rate, propagate it.
    _bleListener = () {
      final cfg = ble.signalConfig;
      if (cfg != null && cfg.samplingRate > 0) {
        signal.sampleRate = cfg.samplingRate.toDouble();
      }
    };
    ble.addListener(_bleListener!);
  }

  void _showError(String message) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: AppTheme.danger,
        duration: const Duration(seconds: 5),
        action: SnackBarAction(
          label: 'Dismiss',
          textColor: Colors.white70,
          onPressed: () =>
              ScaffoldMessenger.of(context).hideCurrentSnackBar(),
        ),
      ),
    );
  }

  @override
  void dispose() {
    // Use cached reference — context is deactivated during dispose.
    if (_bleListener != null) _ble?.removeListener(_bleListener!);
    _bleErrorSub?.cancel();
    _signalErrorSub?.cancel();
    _audioErrorSub?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: IndexedStack(index: _selectedIndex, children: _screens),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _selectedIndex,
        onDestinationSelected: (i) => setState(() => _selectedIndex = i),
        destinations: const [
          NavigationDestination(
            icon: Icon(Icons.dashboard_outlined),
            selectedIcon: Icon(Icons.dashboard),
            label: 'Dashboard',
          ),
          NavigationDestination(
            icon: Icon(Icons.bluetooth_outlined),
            selectedIcon: Icon(Icons.bluetooth),
            label: 'Devices',
          ),
          NavigationDestination(
            icon: Icon(Icons.analytics_outlined),
            selectedIcon: Icon(Icons.analytics),
            label: 'Analysis',
          ),
          NavigationDestination(
            icon: Icon(Icons.settings_outlined),
            selectedIcon: Icon(Icons.settings),
            label: 'Settings',
          ),
        ],
      ),
    );
  }
}
