import 'package:flutter/material.dart';

import '../theme/app_theme.dart';

/// Where a feature's computation happens.
enum ProcessingLocation { device, server }

/// Small pill indicating whether a feature runs locally on this device
/// (no network call) or is sent to the FastMODA server for processing.
class ProcessingBadge extends StatelessWidget {
  final ProcessingLocation location;
  const ProcessingBadge({super.key, required this.location});

  @override
  Widget build(BuildContext context) {
    final isDevice = location == ProcessingLocation.device;
    final color = isDevice ? AppTheme.success : AppTheme.secondary;
    return Tooltip(
      message: isDevice
          ? 'Computed on this device — no data leaves the app'
          : 'Sent to the FastMODA server for processing',
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
        decoration: BoxDecoration(
          color: color.withValues(alpha: 0.15),
          borderRadius: BorderRadius.circular(6),
          border: Border.all(color: color, width: 0.5),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(isDevice ? Icons.smartphone : Icons.cloud_outlined,
                size: 11, color: color),
            const SizedBox(width: 3),
            Text(
              isDevice ? 'On-device' : 'Server',
              style: TextStyle(
                  fontSize: 9, fontWeight: FontWeight.w600, color: color),
            ),
          ],
        ),
      ),
    );
  }
}
