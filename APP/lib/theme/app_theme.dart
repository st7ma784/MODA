import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

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

  static TextTheme _buildTextTheme(TextTheme base) {
    return base.copyWith(
      displayLarge: GoogleFonts.poppins(textStyle: base.displayLarge),
      displayMedium: GoogleFonts.poppins(textStyle: base.displayMedium),
      displaySmall: GoogleFonts.poppins(textStyle: base.displaySmall),
      headlineLarge: GoogleFonts.poppins(textStyle: base.headlineLarge),
      headlineMedium: GoogleFonts.poppins(textStyle: base.headlineMedium),
      headlineSmall: GoogleFonts.poppins(textStyle: base.headlineSmall),
      titleLarge: GoogleFonts.poppins(textStyle: base.titleLarge, fontWeight: FontWeight.w600),
      titleMedium: GoogleFonts.poppins(textStyle: base.titleMedium, fontWeight: FontWeight.w600),
      titleSmall: GoogleFonts.poppins(textStyle: base.titleSmall, fontWeight: FontWeight.w500),
      bodyLarge: GoogleFonts.sourceSans3(textStyle: base.bodyLarge),
      bodyMedium: GoogleFonts.sourceSans3(textStyle: base.bodyMedium),
      bodySmall: GoogleFonts.sourceSans3(textStyle: base.bodySmall),
      labelLarge: GoogleFonts.sourceSans3(textStyle: base.labelLarge, fontWeight: FontWeight.w600),
      labelMedium: GoogleFonts.sourceSans3(textStyle: base.labelMedium),
      labelSmall: GoogleFonts.sourceSans3(textStyle: base.labelSmall, letterSpacing: 0.5),
    );
  }

  static ThemeData get theme {
    final base = ThemeData.dark();
    final textTheme = _buildTextTheme(base.textTheme);

    return ThemeData(
      useMaterial3: true,
      colorScheme: const ColorScheme.dark(
        primary: primary,
        secondary: secondary,
        tertiary: highlight,
        surface: surface,
        error: danger,
        onPrimary: Colors.white,
        onSecondary: Colors.white,
        onTertiary: Colors.black,
      ),
      textTheme: textTheme,
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
      appBarTheme: AppBarTheme(
        backgroundColor: background,
        foregroundColor: Colors.white,
        surfaceTintColor: Colors.transparent,
        titleTextStyle: GoogleFonts.poppins(
          fontSize: 18,
          fontWeight: FontWeight.w600,
          color: Colors.white,
        ),
      ),
      tabBarTheme: const TabBarThemeData(
        labelColor: primary,
        unselectedLabelColor: Colors.white38,
        indicatorColor: primary,
      ),
      inputDecorationTheme: InputDecorationTheme(
        focusedBorder: const OutlineInputBorder(
          borderSide: BorderSide(color: primary, width: 2),
        ),
        enabledBorder: OutlineInputBorder(
          borderSide: BorderSide(color: Colors.white.withValues(alpha: 0.2)),
        ),
        labelStyle: const TextStyle(color: Colors.white54),
        hintStyle: const TextStyle(color: Colors.white24),
      ),
      sliderTheme: const SliderThemeData(
        activeTrackColor: primary,
        thumbColor: primary,
        inactiveTrackColor: Colors.white12,
        overlayColor: indicator,
      ),
      switchTheme: SwitchThemeData(
        thumbColor: WidgetStateProperty.resolveWith((states) =>
            states.contains(WidgetState.selected) ? primary : Colors.white38),
        trackColor: WidgetStateProperty.resolveWith((states) =>
            states.contains(WidgetState.selected) ? indicator : Colors.white12),
      ),
      checkboxTheme: CheckboxThemeData(
        fillColor: WidgetStateProperty.resolveWith((states) =>
            states.contains(WidgetState.selected) ? primary : Colors.transparent),
        checkColor: const WidgetStatePropertyAll(Colors.white),
        side: const BorderSide(color: Colors.white38, width: 1.5),
      ),
      progressIndicatorTheme: const ProgressIndicatorThemeData(
        color: primary,
        linearTrackColor: Colors.white10,
      ),
      chipTheme: ChipThemeData(
        backgroundColor: surface,
        side: const BorderSide(color: Colors.white12),
        labelStyle: GoogleFonts.sourceSans3(fontSize: 11),
      ),
    );
  }
}
