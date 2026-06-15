import 'package:flutter/material.dart';

/// MODA — "Lancaster Heritage" brand palette.
///
/// Dark-surface variant of the palette used on the FastMODA web UI
/// (see FastMODA/static/css/theme.css), keeping the app's existing
/// dark-mode layout while swapping the cyan/teal accent colors for the
/// terracotta/amber/mustard identity derived from the MODA splash screen.
class AppTheme {
  AppTheme._();

  // Brand accents
  static const primary = Color(0xFFC1502E); // Terracotta
  static const primaryDark = Color(0xFFA8412A);
  static const secondary = Color(0xFFE8932E); // Amber
  static const highlight = Color(0xFFF4C95D); // Mustard
  static const support = Color(0xFFC97D74); // Dusty rose

  // Dark surfaces (warm-toned)
  static const background = Color(0xFF221F1D);
  static const surface = Color(0xFF2F2B28);
  static const surfaceAlt = Color(0xFF3A3530);

  // Semantic
  static const success = Color(0xFF7A9D54);
  static const danger = Color(0xFFB3261E);

  // Primary at 20% alpha — used for nav/tab indicators
  static const indicator = Color(0x33C1502E);

  static ThemeData get theme {
    return ThemeData(
      useMaterial3: true,
      colorScheme: const ColorScheme.dark(
        primary: primary,
        secondary: secondary,
        surface: surface,
        error: danger,
        onPrimary: Colors.white,
        onSecondary: Colors.white,
      ),
      scaffoldBackgroundColor: background,
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
        indicatorColor: indicator,
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
