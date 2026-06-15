import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import '../services/ble_service.dart';
import '../services/app_settings.dart';
import '../config/app_config.dart';

class BleScreen extends StatelessWidget {
  const BleScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final ble = context.watch<BleService>();

    return Scaffold(
      appBar: AppBar(
        title: const Text('Devices'),
        backgroundColor: Colors.transparent,
        elevation: 0,
        actions: [
          if (ble.isConnected && ble.isModaDevice && !ble.isStreaming)
            IconButton(
              tooltip: 'Resume streaming',
              icon: const Icon(Icons.play_arrow),
              onPressed: () => context.read<BleService>().startStreaming(),
            ),
          IconButton(
            tooltip: ble.isScanning ? 'Stop scan' : 'Scan',
            icon: Icon(ble.isScanning ? Icons.stop : Icons.refresh),
            onPressed: ble.isScanning
                ? () => context.read<BleService>().stopScan()
                : () => context.read<BleService>().startScan(),
          ),
        ],
      ),
      body: Column(
        children: [
          // BT adapter state banner — shown whenever Bluetooth is not on
          StreamBuilder<BluetoothAdapterState>(
            stream: context.read<BleService>().adapterState,
            builder: (context, snap) {
              final state = snap.data;
              if (state == null || state == BluetoothAdapterState.on) {
                return const SizedBox.shrink();
              }
              return _BtOffBanner(state: state);
            },
          ),
          if (ble.isConnected) _ConnectedBanner(ble: ble),
          if (ble.isScanning) const LinearProgressIndicator(minHeight: 2),
          Expanded(
            child: ble.isConnected
                ? (ble.isModaDevice
                    ? _ModaDevicePanel(ble: ble)
                    : _GenericPanel(ble: ble))
                : (ble.scanResults.isEmpty
                    ? _EmptyState(isScanning: ble.isScanning)
                    : _ScanResultsList(ble: ble)),
          ),
        ],
      ),
    );
  }
}

// ── BT off banner ────────────────────────────────────────────────────────────

class _BtOffBanner extends StatelessWidget {
  final BluetoothAdapterState state;
  const _BtOffBanner({required this.state});

  @override
  Widget build(BuildContext context) {
    final isOff = state == BluetoothAdapterState.off;
    return Container(
      color: Colors.orange.withValues(alpha: 0.15),
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
      child: Row(
        children: [
          const Icon(Icons.bluetooth_disabled, color: Colors.orange, size: 18),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              isOff
                  ? 'Bluetooth is off — please enable it to scan for devices.'
                  : 'Bluetooth is unavailable on this device.',
              style: const TextStyle(color: Colors.orange, fontSize: 13),
            ),
          ),
        ],
      ),
    );
  }
}

// ── Connected banner ──────────────────────────────────────────────────────────

class _ConnectedBanner extends StatelessWidget {
  final BleService ble;
  const _ConnectedBanner({required this.ble});

  @override
  Widget build(BuildContext context) {
    final name = ble.connectedDevice!.platformName.isNotEmpty
        ? ble.connectedDevice!.platformName
        : ble.connectedDevice!.remoteId.toString();
    return Container(
      color: Colors.green.withValues(alpha: 0.1),
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
      child: Row(
        children: [
          const Icon(Icons.check_circle, color: Colors.green, size: 18),
          const SizedBox(width: 8),
          Expanded(
              child: Text('Connected: $name',
                  style: const TextStyle(color: Colors.green))),
          if (ble.isStreaming)
            Padding(
              padding: const EdgeInsets.only(right: 8),
              child: Text('● LIVE',
                  style: TextStyle(
                      fontSize: 10,
                      color: Theme.of(context).colorScheme.secondary,
                      fontWeight: FontWeight.w700)),
            ),
          TextButton(
            onPressed: () => context.read<BleService>().disconnect(),
            child:
                const Text('Disconnect', style: TextStyle(color: Colors.red)),
          ),
        ],
      ),
    );
  }
}

// ── MODA device panel ─────────────────────────────────────────────────────────

class _ModaDevicePanel extends StatelessWidget {
  final BleService ble;
  const _ModaDevicePanel({required this.ble});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final info = ble.deviceInfo;
    final cfg = ble.signalConfig;
    final status = ble.deviceStatus;

    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        // Device info card
        if (info != null) ...[
          Text('Device', style: theme.textTheme.labelLarge),
          const SizedBox(height: 8),
          Card(
            child: Padding(
              padding: const EdgeInsets.all(12),
              child: Column(
                children: [
                  _InfoRow(
                      icon: Icons.sensors,
                      label: info.deviceName,
                      value: info.firmwareVersion.isNotEmpty
                          ? 'fw ${info.firmwareVersion}'
                          : ''),
                  if (info.batteryLevel >= 0)
                    _InfoRow(
                        icon: _batteryIcon(info.batteryLevel),
                        label: 'Battery',
                        value: '${info.batteryLevel}%',
                        valueColor: info.batteryLevel < 20
                            ? Colors.red
                            : Colors.green),
                  _InfoRow(
                      icon: Icons.cable,
                      label: 'Max channels',
                      value: '${info.maxChannels}'),
                  _InfoRow(
                      icon: Icons.speed,
                      label: 'Max sample rate',
                      value: '${info.maxSamplingRate} Hz'),
                ],
              ),
            ),
          ),
          const SizedBox(height: 16),
        ],

        // Signal config card
        if (cfg != null) ...[
          Text('Signal Config', style: theme.textTheme.labelLarge),
          const SizedBox(height: 8),
          Card(
            child: Padding(
              padding: const EdgeInsets.all(12),
              child: Column(
                children: [
                  _InfoRow(
                      icon: Icons.graphic_eq,
                      label: 'Sampling rate',
                      value: '${cfg.samplingRate} Hz'),
                  _InfoRow(
                      icon: Icons.layers,
                      label: 'Channels',
                      value: '${cfg.numChannels}'),
                  _InfoRow(
                      icon: Icons.data_array,
                      label: 'Format',
                      value: cfg.dataFormat == 0 ? 'Int16 LE' : 'Float32 LE'),
                  if (cfg.filterEnabled)
                    _InfoRow(
                        icon: Icons.filter_alt,
                        label: 'Band-pass filter',
                        value:
                            '${cfg.filterCutoffLow.toStringAsFixed(1)}–${cfg.filterCutoffHigh.toStringAsFixed(0)} Hz'),
                ],
              ),
            ),
          ),
          const SizedBox(height: 16),
        ],

        // Status card
        Text('Status', style: theme.textTheme.labelLarge),
        const SizedBox(height: 8),
        Card(
          child: Padding(
            padding: const EdgeInsets.all(12),
            child: status == null
                ? const Center(
                    child: Padding(
                      padding: EdgeInsets.all(8),
                      child: Text('Awaiting status…',
                          style: TextStyle(color: Colors.white38)),
                    ),
                  )
                : Column(
                    children: [
                      _InfoRow(
                          icon: status.isStreaming
                              ? Icons.radio_button_checked
                              : Icons.radio_button_unchecked,
                          label: 'State',
                          value: status.stateLabel,
                          valueColor: status.isStreaming
                              ? Theme.of(context).colorScheme.secondary
                              : status.hasError
                                  ? Colors.red
                                  : null),
                      _InfoRow(
                          icon: Icons.signal_cellular_alt,
                          label: 'Signal quality',
                          value: '${status.signalQuality}%'),
                      _InfoRow(
                          icon: Icons.warning_amber,
                          label: 'Packets lost',
                          value: '${status.packetsLost + ble.packetsLost}',
                          valueColor: (status.packetsLost + ble.packetsLost) > 10
                              ? Colors.orange
                              : null),
                      if (status.temperature > 0)
                        _InfoRow(
                            icon: Icons.thermostat,
                            label: 'Temperature',
                            value:
                                '${status.temperature.toStringAsFixed(1)} °C'),
                    ],
                  ),
          ),
        ),

        // Stream toggle
        const SizedBox(height: 16),
        ble.isStreaming
            ? OutlinedButton.icon(
                onPressed: () => context.read<BleService>().stopStreaming(),
                icon: const Icon(Icons.stop, color: Colors.red),
                label:
                    const Text('Stop Streaming', style: TextStyle(color: Colors.red)),
              )
            : FilledButton.icon(
                onPressed: () => context.read<BleService>().startStreaming(),
                icon: const Icon(Icons.play_arrow),
                label: const Text('Start Streaming'),
              ),
      ],
    );
  }

  IconData _batteryIcon(int pct) {
    if (pct > 75) return Icons.battery_full;
    if (pct > 50) return Icons.battery_5_bar;
    if (pct > 25) return Icons.battery_3_bar;
    if (pct > 10) return Icons.battery_2_bar;
    return Icons.battery_alert;
  }
}

class _InfoRow extends StatelessWidget {
  final IconData icon;
  final String label;
  final String value;
  final Color? valueColor;

  const _InfoRow(
      {required this.icon,
      required this.label,
      required this.value,
      this.valueColor});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 5),
      child: Row(
        children: [
          Icon(icon, size: 16, color: Colors.white38),
          const SizedBox(width: 10),
          Expanded(
              child: Text(label,
                  style: const TextStyle(fontSize: 13, color: Colors.white70))),
          Text(value,
              style: TextStyle(
                  fontSize: 13,
                  fontWeight: FontWeight.w600,
                  color: valueColor ?? Colors.white)),
        ],
      ),
    );
  }
}

// ── Generic panel (non-MODA) ──────────────────────────────────────────────────

class _GenericPanel extends StatefulWidget {
  final BleService ble;
  const _GenericPanel({required this.ble});

  @override
  State<_GenericPanel> createState() => _GenericPanelState();
}

class _GenericPanelState extends State<_GenericPanel> {
  BleDataFormat _format = BleDataFormat.int16LE;

  @override
  Widget build(BuildContext context) {
    final ble = widget.ble;
    final theme = Theme.of(context);
    final chars = ble.characteristics;

    if (chars.isEmpty) {
      return const Center(
        child: Text('No notify characteristics found',
            style: TextStyle(color: Colors.white38)),
      );
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 12, 16, 4),
          child: Row(
            children: [
              Text('Notify Characteristics', style: theme.textTheme.labelLarge),
              const Spacer(),
              const Text('Format: ',
                  style: TextStyle(fontSize: 12, color: Colors.white54)),
              DropdownButton<BleDataFormat>(
                value: _format,
                isDense: true,
                underline: const SizedBox(),
                items: const [
                  DropdownMenuItem(
                      value: BleDataFormat.int16LE,
                      child: Text('Int16 LE', style: TextStyle(fontSize: 12))),
                  DropdownMenuItem(
                      value: BleDataFormat.float32LE,
                      child:
                          Text('Float32 LE', style: TextStyle(fontSize: 12))),
                ],
                onChanged: (v) => setState(() => _format = v!),
              ),
            ],
          ),
        ),
        Expanded(
          child: ListView.builder(
            padding: const EdgeInsets.fromLTRB(16, 4, 16, 16),
            itemCount: chars.length,
            itemBuilder: (context, i) {
              final char = chars[i];
              final isActive = ble.activeCharacteristic?.uuid == char.uuid;
              final uuid = char.uuid.toString();
              final props = [
                if (char.properties.notify) 'notify',
                if (char.properties.indicate) 'indicate',
              ].join(', ');

              return Card(
                margin: const EdgeInsets.only(bottom: 8),
                child: ListTile(
                  leading: Icon(
                    isActive
                        ? Icons.radio_button_checked
                        : Icons.radio_button_unchecked,
                    color: isActive
                        ? Theme.of(context).colorScheme.secondary
                        : Colors.white38,
                  ),
                  title: Text(
                    uuid.length > 8
                        ? '…${uuid.substring(uuid.length - 8)}'
                        : uuid,
                    style: const TextStyle(
                        fontFamily: 'monospace', fontSize: 13),
                  ),
                  subtitle: Text(props,
                      style: const TextStyle(
                          fontSize: 11, color: Colors.white38)),
                  trailing: isActive
                      ? TextButton(
                          onPressed: () =>
                              context.read<BleService>().stopGenericStreaming(),
                          child: const Text('Stop',
                              style: TextStyle(color: Colors.red)),
                        )
                      : TextButton(
                          onPressed: () async {
                            final bleService = context.read<BleService>();
                            final appSettings = context.read<AppSettings>();
                            await bleService.subscribeToCharacteristic(char,
                                format: _format);
                            await appSettings.setBleCharUuid(uuid);
                          },
                          child: const Text('Stream'),
                        ),
                  onLongPress: () => ScaffoldMessenger.of(context).showSnackBar(
                    SnackBar(
                        content: Text(uuid),
                        duration: const Duration(seconds: 3)),
                  ),
                ),
              );
            },
          ),
        ),
      ],
    );
  }
}

// ── Empty state ───────────────────────────────────────────────────────────────

class _EmptyState extends StatelessWidget {
  final bool isScanning;
  const _EmptyState({required this.isScanning});

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            isScanning ? Icons.bluetooth_searching : Icons.bluetooth_disabled,
            size: 64,
            color: Colors.white24,
          ),
          const SizedBox(height: 16),
          Text(
            isScanning ? 'Scanning for devices…' : 'No devices found',
            style: const TextStyle(color: Colors.white54),
          ),
          if (!isScanning) ...[
            const SizedBox(height: 16),
            ElevatedButton.icon(
              onPressed: () => context.read<BleService>().startScan(),
              icon: const Icon(Icons.search),
              label: const Text('Scan for Devices'),
            ),
          ],
        ],
      ),
    );
  }
}

// ── Scan results list ─────────────────────────────────────────────────────────

class _ScanResultsList extends StatelessWidget {
  final BleService ble;
  const _ScanResultsList({required this.ble});

  @override
  Widget build(BuildContext context) {
    return ListView.builder(
      padding: const EdgeInsets.all(16),
      itemCount: ble.scanResults.length,
      itemBuilder: (context, i) => _DeviceTile(result: ble.scanResults[i], ble: ble),
    );
  }
}

class _DeviceTile extends StatelessWidget {
  final ScanResult result;
  final BleService ble;
  const _DeviceTile({required this.result, required this.ble});

  @override
  Widget build(BuildContext context) {
    final isConnected =
        ble.connectedDevice?.remoteId == result.device.remoteId;
    final name = result.device.platformName.isNotEmpty
        ? result.device.platformName
        : 'Unknown Device';
    // Highlight MODA devices in the scan list
    final isModa = result.advertisementData.serviceUuids
        .any((u) => u.toString().toLowerCase() == kModaServiceUuid.toLowerCase());

    return Card(
      margin: const EdgeInsets.only(bottom: 8),
      child: ListTile(
        leading: Icon(
          isModa ? Icons.sensors : Icons.bluetooth,
          color: isConnected
              ? Colors.green
              : isModa
                  ? Theme.of(context).colorScheme.primary
                  : Colors.white54,
        ),
        title: Row(
          children: [
            Text(name),
            if (isModa) ...[
              const SizedBox(width: 6),
              Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 5, vertical: 1),
                decoration: BoxDecoration(
                  color: Theme.of(context).colorScheme.primary.withValues(alpha: 0.2),
                  borderRadius: BorderRadius.circular(4),
                ),
                child: Text('MODA',
                    style: TextStyle(
                        fontSize: 9,
                        color: Theme.of(context).colorScheme.primary,
                        fontWeight: FontWeight.w700)),
              ),
            ],
          ],
        ),
        subtitle: Text(result.device.remoteId.toString(),
            style: const TextStyle(fontSize: 11)),
        trailing: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text('${result.rssi} dBm',
                style: const TextStyle(fontSize: 12, color: Colors.white38)),
            const SizedBox(width: 4),
            isConnected
                ? TextButton(
                    onPressed: () => context.read<BleService>().disconnect(),
                    child: const Text('Disconnect',
                        style: TextStyle(color: Colors.red)),
                  )
                : TextButton(
                    onPressed: () =>
                        context.read<BleService>().connect(result.device),
                    child: const Text('Connect'),
                  ),
          ],
        ),
      ),
    );
  }
}
