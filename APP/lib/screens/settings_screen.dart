import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../config/app_config.dart';
import '../services/analysis_history_service.dart';
import '../services/fastmoda_client.dart';
import '../services/signal_service.dart';
import '../services/app_settings.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  final _urlController = TextEditingController();
  final _uuidController = TextEditingController();
  final _fsController = TextEditingController();
  bool _apiKeyVisible = false;
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    _loadSettings();
  }

  Future<void> _loadSettings() async {
    final settings = context.read<AppSettings>();
    final url = await settings.getServerUrl();
    final uuid = await settings.getBleCharUuid();
    final fs = await settings.getSampleRate();
    if (mounted) {
      setState(() {
        _urlController.text = url;
        _uuidController.text = uuid;
        _fsController.text = fs.toStringAsFixed(0);
        _loading = false;
      });
    }
  }

  @override
  void dispose() {
    _urlController.dispose();
    _uuidController.dispose();
    _fsController.dispose();
    super.dispose();
  }

  Future<void> _saveServer() async {
    final url = _urlController.text.trim();
    if (url.isEmpty) return;
    final settings = context.read<AppSettings>();
    final client = context.read<FastModaClient>();
    final signal = context.read<SignalService>();
    await settings.setServerUrl(url);
    client.setBaseUrl(url);
    signal.bindClient(client);
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Server URL saved')),
      );
    }
  }

  Future<void> _saveBleUuid() async {
    final uuid = _uuidController.text.trim();
    await context.read<AppSettings>().setBleCharUuid(uuid);
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('BLE characteristic UUID saved')),
      );
    }
  }

  Future<void> _saveSampleRate() async {
    final fs = double.tryParse(_fsController.text.trim());
    if (fs == null || fs <= 0) return;
    await context.read<AppSettings>().setSampleRate(fs);
    if (mounted) context.read<SignalService>().sampleRate = fs;
  }

  @override
  Widget build(BuildContext context) {
    if (_loading) {
      return const Scaffold(
          body: Center(child: CircularProgressIndicator()));
    }
    return Scaffold(
      appBar: AppBar(
        title: const Text('Settings'),
        backgroundColor: Colors.transparent,
        elevation: 0,
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          _SectionLabel('Server'),
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text('FastMODA URL',
                      style: TextStyle(fontSize: 13, color: Colors.white54)),
                  const SizedBox(height: 8),
                  TextField(
                    controller: _urlController,
                    decoration: const InputDecoration(
                      hintText: 'https://moda.example.com',
                      border: OutlineInputBorder(),
                      isDense: true,
                      prefixIcon: Icon(Icons.link, size: 18),
                    ),
                    keyboardType: TextInputType.url,
                    onSubmitted: (_) => _saveServer(),
                  ),
                  const SizedBox(height: 8),
                  Align(
                    alignment: Alignment.centerRight,
                    child: FilledButton.tonal(
                      onPressed: _saveServer,
                      child: const Text('Save & Connect'),
                    ),
                  ),
                  const SizedBox(height: 12),
                  Row(
                    children: [
                      const Text('API Key',
                          style:
                              TextStyle(fontSize: 13, color: Colors.white54)),
                      const SizedBox(width: 8),
                      InkWell(
                        onTap: () =>
                            setState(() => _apiKeyVisible = !_apiKeyVisible),
                        child: Icon(
                          _apiKeyVisible
                              ? Icons.visibility_off
                              : Icons.visibility,
                          size: 16,
                          color: Colors.white38,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 4),
                  Text(
                    _apiKeyVisible ? kFastModaApiKey : '•' * 28,
                    style: const TextStyle(
                        fontFamily: 'monospace',
                        fontSize: 11,
                        color: Colors.white38),
                  ),
                  const SizedBox(height: 4),
                  const Text(
                    'Compiled-in — must match the Helm deployment.',
                    style: TextStyle(fontSize: 11, color: Colors.white24),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 16),
          _SectionLabel('Bluetooth'),
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text('Data Characteristic UUID',
                      style: TextStyle(fontSize: 13, color: Colors.white54)),
                  const SizedBox(height: 4),
                  const Text(
                    'Paste the UUID of the notify characteristic your device uses for signal data. Leave blank to select manually in the Devices tab.',
                    style: TextStyle(fontSize: 11, color: Colors.white38),
                  ),
                  const SizedBox(height: 8),
                  TextField(
                    controller: _uuidController,
                    decoration: const InputDecoration(
                      hintText: '0000xxxx-0000-1000-8000-00805f9b34fb',
                      border: OutlineInputBorder(),
                      isDense: true,
                      prefixIcon:
                          Icon(Icons.bluetooth_searching, size: 18),
                    ),
                    keyboardType: TextInputType.text,
                  ),
                  const SizedBox(height: 8),
                  Align(
                    alignment: Alignment.centerRight,
                    child: FilledButton.tonal(
                      onPressed: _saveBleUuid,
                      child: const Text('Save UUID'),
                    ),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 16),
          _SectionLabel('Analysis'),
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text('Sampling Rate (Hz)',
                      style: TextStyle(fontSize: 13, color: Colors.white54)),
                  const SizedBox(height: 8),
                  TextField(
                    controller: _fsController,
                    decoration: const InputDecoration(
                      hintText: '256',
                      border: OutlineInputBorder(),
                      isDense: true,
                      suffixText: 'Hz',
                    ),
                    keyboardType: TextInputType.number,
                    onSubmitted: (_) => _saveSampleRate(),
                  ),
                  const SizedBox(height: 8),
                  Align(
                    alignment: Alignment.centerRight,
                    child: FilledButton.tonal(
                      onPressed: _saveSampleRate,
                      child: const Text('Save'),
                    ),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 16),
          _SectionLabel('Storage'),
          Card(
            child: ListTile(
              leading: const Icon(Icons.delete_outline, color: Colors.red),
              title: const Text('Clear all data',
                  style: TextStyle(color: Colors.red)),
              onTap: () => _confirmClear(context),
            ),
          ),
          const SizedBox(height: 16),
          _SectionLabel('About'),
          const Card(
            child: Column(
              children: [
                ListTile(
                  title: Text('Version'),
                  trailing:
                      Text('1.0.0', style: TextStyle(color: Colors.white54)),
                ),
                Divider(height: 1),
                ListTile(
                  title: Text('API Target'),
                  trailing: Text('FastMODA v1',
                      style: TextStyle(color: Colors.white54)),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  void _confirmClear(BuildContext context) {
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Clear all data?'),
        content: const Text(
            'This permanently deletes all recorded signals and analysis results.'),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(ctx),
              child: const Text('Cancel')),
          TextButton(
              onPressed: () {
                Navigator.pop(ctx);
                context.read<AnalysisHistoryService>().clearAll();
              },
              child: const Text('Delete',
                  style: TextStyle(color: Colors.red))),
        ],
      ),
    );
  }
}

class _SectionLabel extends StatelessWidget {
  final String text;
  const _SectionLabel(this.text);

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(left: 4, bottom: 8),
      child: Text(
        text.toUpperCase(),
        style: TextStyle(
          fontSize: 11,
          fontWeight: FontWeight.w700,
          color: Theme.of(context).colorScheme.primary,
          letterSpacing: 1.4,
        ),
      ),
    );
  }
}
