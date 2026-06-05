// Smoke test for a leaf widget. (Replaces the stale Flutter counter template
// that referenced a non-existent `MyApp`/counter UI.) A full-app pump is not
// used here because ModaApp depends on platform plugins — sqflite,
// secure storage — that aren't available under the plain VM test runner.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:moda_mobile/widgets/band_power_card.dart';

void main() {
  testWidgets('BandPowerCard renders its band label and percentage',
      (WidgetTester tester) async {
    await tester.pumpWidget(const MaterialApp(
      home: Scaffold(
        body: BandPowerCard(
          band: 'Alpha',
          hz: '8–12 Hz',
          color: Colors.teal,
          power: 0.5,
        ),
      ),
    ));

    expect(find.text('Alpha'), findsOneWidget);
    expect(find.text('8–12 Hz'), findsOneWidget);
    expect(find.text('50%'), findsOneWidget);
  });
}
