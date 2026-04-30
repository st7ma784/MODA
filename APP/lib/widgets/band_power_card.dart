import 'package:flutter/material.dart';

class BandPowerCard extends StatelessWidget {
  final String band;
  final String hz;
  final Color color;
  final double? power; // 0.0–1.0

  const BandPowerCard({
    super.key,
    required this.band,
    required this.hz,
    required this.color,
    this.power,
  });

  @override
  Widget build(BuildContext context) {
    final pct = power?.clamp(0.0, 1.0) ?? 0.0;
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(10),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Container(
                  width: 8,
                  height: 8,
                  decoration: BoxDecoration(color: color, shape: BoxShape.circle),
                ),
                const SizedBox(width: 6),
                Text(band,
                    style: const TextStyle(
                        fontSize: 12, fontWeight: FontWeight.w600)),
              ],
            ),
            const SizedBox(height: 2),
            Text(hz,
                style: const TextStyle(fontSize: 10, color: Colors.white38)),
            const SizedBox(height: 6),
            ClipRRect(
              borderRadius: BorderRadius.circular(3),
              child: LinearProgressIndicator(
                value: pct,
                backgroundColor: Colors.white12,
                valueColor: AlwaysStoppedAnimation(color),
                minHeight: 6,
              ),
            ),
            const SizedBox(height: 4),
            Text(
              power != null ? '${(pct * 100).toStringAsFixed(0)}%' : '—',
              style: const TextStyle(fontSize: 11, color: Colors.white54),
            ),
          ],
        ),
      ),
    );
  }
}
