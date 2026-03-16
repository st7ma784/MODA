# MODA Mobile App Development Plan

## Executive Summary

This document outlines the comprehensive plan for developing a mobile application for MODA (Multiscale Oscillatory Dynamics Analysis) that receives signal data via Bluetooth and performs real-time signal analysis locally on the handset, with optional fallback to a self-hosted ARM server for advanced analysis.

**Target Platforms:** iOS and Android
**Architecture:** ARM-native with primary on-device processing + optional self-hosted server fallback
**Data Input:** Bluetooth Low Energy (BLE)
**Processing Hierarchy:** On-device (PRIMARY) → Self-hosted ARM server (FALLBACK) → Cloud (Optional advanced features)
**Timeline:** 12-16 weeks for MVP

---

## 1. Architecture Overview

### 1.1 System Architecture (On-Device First)

```
┌───────────────────────────────────────────────────────────────────┐
│                   MOBILE APP (Flutter - ARM64 Native)             │
├───────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐    │
│  │   BLE Layer  │  │  UI Layer    │  │   Data Manager       │    │
│  │              │  │              │  │                      │    │
│  │ - Discovery  │  │ - Dashboard  │  │ - Local Storage      │    │
│  │ - Pairing    │  │ - Charts     │  │ - SQLite Database    │    │
│  │ - Data RX    │  │ - Settings   │  │ - Caching            │    │
│  └──────────────┘  └──────────────┘  └──────────────────────┘    │
│           │               │                      │                │
│           └───────────────┴──────────────────────┘                │
│                           │                                       │
│      ┌────────────────────┴─────────────────────────────┐        │
│      │                                                  │        │
│      ▼                                                  ▼        │
│  ┌────────────────────────────────────┐   ┌─────────────────────┐│
│  │   ON-DEVICE PROCESSING (PRIMARY)   │   │  Server Check (if   ││
│  │         (ARM64 Native)             │   │   available locally)││
│  │                                    │   └─────────────────────┘│
│  │  • NumPy (ARM-optimized)           │                          │
│  │  • SciPy/Signal processing         │   If device unavailable │
│  │  • Base FFT/iFFT analysis          │   or processing needed: │
│  │  • Lightweight MODWT (few levels)  │   ▼                     │
│  │  • Time-domain changepoints        │   HTTP Client           │
│  │  • Real-time band power tracking   │   (auto-discovery)      │
│  │  • Basic spectral coherence        │                          │
│  │  • Immediate results (<1s-10s)     │                          │
│  └────────────────────────────────────┘                          │
│           ▲                                                        │
│           │ (offline processing)                                  │
│           │ (no network needed)                                   │
│           │                                                        │
└───────────┼────────────────────────────────────────────────────────┘
            │
            │ (signal data)
            │
    ┌───────┴─────────────────────────────────────────────────────┐
    │   OPTIONAL: LOCAL SELF-HOSTED ARM SERVER (FALLBACK)         │
    │   Raspberry Pi 4/5 or Jetson Nano (on home WiFi)            │
    │   ┌──────────────────────────────────────────────────────┐  │
    │   │         FastMODA API (Python/Flask)                 │  │
    │   │         Running ARM64 Docker Container              │  │
    │   │                                                      │  │
    │   │  • Full Spectral Analysis                           │  │
    │   │  • Complete MODWT Wavelet (all decomposition)       │  │
    │   │  • Phase/Magnitude Coherence                        │  │
    │   │  • Bispectrum Analysis                              │  │
    │   │  • Bayesian Inference                               │  │
    │   │  • GPU Acceleration (Jetson Nano w/ CUDA)           │  │
    │   │  • Batch processing of multiple signals             │  │
    │   │  • Historical analysis & archival                   │  │
    │   │                                                      │  │
    │   │  Discovery: mDNS (moda-server.local)                │  │
    │   │  Network: Local WiFi (home/office)                  │  │
    │   │  Deployment: Docker (linux/arm64)                   │  │
    │   └──────────────────────────────────────────────────────┘  │
    └────────────────────────────────────────────────────────────┘
```

### 1.2 Processing Decision Tree

```
Signal Data Received from BLE Device
            │
            ▼
    ┌───────────────────────────────┐
    │ Analysis Type & Signal Length │
    └───────┬───────────────────────┘
            │
        ┌───┴────┬────────────────────┬─────────────────┐
        │        │                    │                 │
        ▼        ▼                    ▼                 ▼
    Quick    Medium              Deep            Batch/Archive
    (<500s)  (500-5000)         (5000+)         Processing
        │        │                    │                 │
        ▼        ▼                    ▼                 ▼
    ┌──────────────────────┐  ┌──────────────────┐  ┌──────────┐
    │  ON-DEVICE          │  │ Server Available?│  │  Server  │
    │  Spectral FFT       │  │   (WiFi check)   │  │  Only    │
    │  Basic Power Bands  │  │                  │  │  Option  │
    │  Instant Results    │  └────┬────────┬────┘  └──────────┘
    │  (Always available) │       │        │
    └──────────────────────┘       │        │
                            YES ▼   ▼ NO
                        ┌──────────────┐
                        │ Use Server   │  Use Limited On-Dev
                        │ Full Analysis│  Cache for later
                        │ Features     │  When server avail
                        └──────────────┘
```

### 1.3 Data Flow

**Primary Path (On-Device - Always Works):**
1. **Bluetooth Sensor** → BLE signal → **Mobile App**
2. **Mobile App** → Signal buffering → **Local Storage (SQLite)**
3. **Mobile App** → ARM-native processing → **Analysis Results**
4. **Mobile App** → Display results → **User Interface**
5. (Optional) **Mobile App** → Export/Archive → **Local file storage**

**Fallback Path (Self-Hosted Server - When Available):**
1. **Mobile App** → Detect local WiFi & check for mDNS broadcast
2. **Mobile App** → (If moda-server.local found) Send data → **Self-Hosted FastMODA API**
3. **FastMODA API** → Full analysis processing → **Results (JSON)**
4. **Mobile App** → Display enhanced results → **User Interface**

**Cloud Path (Optional, Not Recommended for MVP):**
- Only if user explicitly opts in for archival, historical analysis, or research
- Requires explicit user consent and internet connectivity
- Secondary to all on-device and local server options

### 1.4 Feature Parity

| Feature | On-Device | Self-Hosted Server |
|---------|-----------|-------------------|
| Spectral Analysis (FFT) | ✅ | ✅ |
| Power Spectral Density | ✅ | ✅ |
| Band Power Tracking (Delta, Theta, Alpha, Beta, Gamma) | ✅ | ✅ |
| Real-time Changepoint Detection | ✅ | ✅ |
| Time-Frequency Spectrograms | ✅ | ✅ |
| MODWT Wavelet (Limited: 2-3 levels) | ✅ | ✅ Full (all levels) |
| Phase Coherence (Basic) | ✅ | ✅ Full |
| Bispectrum Analysis | ❌ | ✅ |
| Bayesian Inference | ❌ | ✅ |
| Multi-signal Analysis | ❌ | ✅ |
| Batch / Historical Processing | ❌ | ✅ |
| GPU Acceleration | ❌ | ✅ (Jetson only) |

---

## 2. Technology Stack

### 2.1 Mobile App Framework Options

| Option | Pros | Cons | Recommendation |
|--------|------|------|----------------|
| **Flutter** | Single codebase, excellent performance, rich UI, strong BLE support | Larger app size | ⭐ **Recommended** |
| **React Native** | JavaScript ecosystem, large community, good BLE libraries | Performance overhead for heavy computation | Alternative |
| **Native (Swift/Kotlin)** | Best performance, platform-specific features | Duplicate development effort | Not recommended for MVP |

**Selected: Flutter** for cross-platform efficiency and native performance.

### 2.2 Backend

- **FastMODA API** (existing Python/Flask implementation)
- **Deployment Options:**
  - **Cloud:** AWS/GCP/Azure for scalability
  - **Edge Computing:** On-device server for offline operation (see Section 5)
  - **Hybrid:** Cloud with local fallback

### 2.3 Supporting Technologies

- **Bluetooth:** `flutter_blue_plus` (Flutter) or `react-native-ble-plx` (React Native)
- **Charts:** `fl_chart` (Flutter) or `Plotly.js` (web-based)
- **HTTP Client:** `dio` or `http` package
- **Local Storage:** SQLite via `sqflite` or Hive (key-value store)
- **State Management:** Provider, Riverpod, or BLoC pattern

---

## 3. Mobile App Features

### 3.1 Core Features (MVP) - On-Device First

#### 3.1.1 Bluetooth Management
- **Device Discovery:** Scan for nearby BLE devices with signal strength
- **Pairing & Connection:** Secure pairing and automatic reconnection
- **Data Reception:** Continuous real-time streaming of signal data
- **Protocol Support:**
  - Custom MODA protocol (defined in `BLUETOOTH_PROTOCOL.md`)
  - Generic UART/Serial protocols
  - Medical device standards (e.g., BLE Heart Rate Profile)
- **Connection Status:** Real-time feedback on connection quality and packet loss

#### 3.1.2 Signal Acquisition & Storage
- **Real-time Buffering:** Sliding window buffer (10-60 seconds) for live analysis
- **Sampling Rate Detection:** Auto-detect or manual configuration
- **Multi-channel Support:** Up to 6 simultaneous signals (hardware permitting)
- **Quality Indicators:** Signal strength, packet loss, noise level monitoring
- **Local Archival:** Save raw signals to SQLite for later offline processing

#### 3.1.3 On-Device Analysis (PRIMARY - Always Available)

**Quick Signal Analysis (Real-Time, <1-5 seconds):**
- **Basic FFT Power Spectrum:** Frequency domain representation (0-Nyquist)
- **Power Spectral Density (PSD):** Welch's method for robust frequency estimates
- **Band Power Decomposition:**
  - Delta (0.5-4 Hz): Slow oscillations
  - Theta (4-8 Hz): Sleep spindles, focal activity
  - Alpha (8-12 Hz): Resting state
  - Beta (12-30 Hz): Motor/cognitive activity
  - Gamma (30-100 Hz): High-frequency oscill ations
- **Real-Time Dominant Frequency Tracking:** Top 5 frequency components with power
- **Spectral Flatness:** Entropy-based measure of complexity
- **Changepoint Detection:** Lightweight algorithm to detect abrupt signal changes

**Time-Frequency Analysis (On-Device):**
- **MODWT Wavelet (Lite):** 2-3 level decomposition using arm64 optimized libraries
  - Fast computation suitable for handset CPUs
  - Energy scalogram visualization
- **Spectrogram:** Time-frequency representation with Welch windowing

**Advanced On-Device (If Sufficient Resources):**
- **Cross-Correlation:** Between multi-channel signals
- **Basic Coherence:** Limited but functional for on-device
- **Phase Lag Index:** Non-linear phase-based connectivity

#### 3.1.4 Server-Based Analysis (FALLBACK - When Available Locally)

Only available when a local self-hosted FastMODA server is detected on WiFi:

- **Full MODWT Wavelet Decomposition:** All decomposition levels for complete frequency coverage
- **Advanced Coherence:** Magnitude and phase coherence across all frequency bands
- **Phase Coherence Analysis:** Fine-grained phase synchronization measures
- **Bispectrum Analysis:** Higher-order spectral analysis for non-linear interactions
- **Bayesian Inference:** Probabilistic signal source localization and classification
- **Multi-Signal Integration:** Complex analysis across 6+ channels simultaneously
- **Batch Processing:** Analysis of historical data and signal compilations
- **GPU-Accelerated Analysis:** (If Jetson Nano with CUDA available)

#### 3.1.5 Visualization & Interaction

**Time-Domain Presentation:**
- Raw signal plot with changepoint markers overlaid
- Zoom/pan functionality for detailed inspection
- Multi-channel superposition option

**Frequency-Domain:**
- Interactive power spectrum with frequency labels
- Band power overlay with color-coded regions
- Dominant frequency annotation with power values

**Time-Frequency:**
- Interactive spectrogram with hover tooltips
- Intensity scaling adjustable by user
- Frequency-band highlights

**Dashboard & Cards:**
- Current signal metrics (mean frequency, power, signal quality)
- Band power summary with activity indicators
- Streaming data rate and packet loss
- Battery and storage indicators

#### 3.1.6 User Interface Components

**Main Dashboard:**
- Live BLE device status and signal quality
- Current analysis results with key metrics
- Quick-access buttons for analysis modes
- Historical session list

**Analysis View:**
- Tabbed interface:
  - **Live:** Real-time on-device results (always active)
  - **Detailed:** Full on-device spectral breakdown
  - **History:** Previously analyzed sessions
  - **Server** (if available): Enhanced analysis results from self-hosted server

**Settings & Configuration:**
- BLE device pairing management
- Sampling rate and buffer size configuration
- Analysis parameters (frequency ranges, thresholds)
- Server discovery and connection settings
- Export preferences (format, storage location)

**Archive & Export:**
- Session history with metadata (timestamp, device, duration)
- Export formats: CSV, MAT (MATLAB), JSON, PNG (charts)
- Local folder organization by date/device

### 3.2 Advanced Features (Phase 2+)

**Phase 2 (Weeks 13-16):**
- [ ] Enhanced server integration with automatic remote processing
- [ ] Multi-device recording and comparison
- [ ] Real-time alerts for frequency anomalies
- [ ] Custom frequency band definition
- [ ] PDF report generation

**Future Enhancements:**
- Offline on-device neural network (TensorFlow Lite) for signal classification
- Continuous background monitoring with intelligent notifications
- Cloud sync for archived sessions (optional, user-controlled)
- Wearable device integration (Apple Watch, Wear OS)
- Advanced visualization: 3D spectrograms, waterfall plots

---

## 4. Development Roadmap

### Phase 1: Foundation & On-Device Processing (Weeks 1-4)

**Week 1-2: Project Setup & Baseline**
- Set up Flutter project structure with proper directory organization
- Configure iOS and Android build environments and signing
- Set up CI/CD pipelines (GitHub Actions) for automated builds
- Create UI wireframes and mockups (Figma)
- Define custom Bluetooth protocol specification (refer to `BLUETOOTH_PROTOCOL.md`)
- Prepare ARM-native NumPy/SciPy build pipeline (numpy wheels for ARM64)

**Week 3-4: BLE & On-Device Signal Processing**
- Implement BLE discovery, pairing, and connection management
- Create real-time signal RX buffer with data validation
- Implement local SQLite storage for raw signal archival
- Build arm64-optimized signal processing module:
  - FFT using native ARM NEON SIMD via NumPy
  - PSD calculation (Welch's method)
  - Band power decomposition (delta-gamma)
  - Changepoint detection algorithm
- Create unit tests for signal processing (target >80% coverage)
- Integration testing with BLE simulators and real test devices

**Deliverables:**
- Working BLE connection with real signal devices
- Functional on-device FFT + spectrogram
- Real-time band power calculations
- Signal quality metrics display
- Local signal archival to SQLite

### Phase 2: Core On-Device Analysis & Visualization (Weeks 5-8)

**Week 5-6: Advanced On-Device Analysis**
- Implement MODWT wavelet (2-3 level decomposition) using arm64 optimizations
- Add real-time dominant frequency tracking
- Implement spectral flatness and entropy calculations
- Build spectrum stability monitoring
- Write comprehensive tests for all analysis functions
- Performance profiling: ensure <500ms analysis for typical signals

**Week 7-8: Interactive Visualization**
- Build time-domain signal plot with zoom/pan
- Create interactive power spectrum display with band highlighting
- Implement spectrogram visualization (time-frequency)
- Add band power timeline charts
- Create summary metric cards and KPI dashboard
- Implement export functions (CSV, PNG, JSON)

**Deliverables:**
- Complete real-time spectral analysis on handset
- Full interactive visualization suite
- Performance targets met (<500ms processing, <20MB memory)
- Export functionality working

### Phase 3: Server Integration & Fallback (Weeks 9-11)

**Week 9: Self-Hosted Server Client Integration**
- Implement mDNS discovery (`moda-server.local` auto-detection)
- Build HTTP client with connection pooling and retry logic
- Create API contract between app and local FastMODA server
- Implement graceful fallback: use on-device if server unavailable
- Add VPN/local network connectivity detection

**Week 10: Advanced Server Analysis**
- Send signals to server for full MODWT analysis
- Receive and display enhanced results (bispectrum, bayesian, full coherence)
- Implement result caching to reduce repeated server requests
- Add queue system for offline signal processing (analyze when server comes online)
- Build UI tab for server-based results vs on-device results

**Week 11: Multi-Signal & Batch Processing**
- Enable multi-signal analysis on server (if available)
- Implement batch processing: queue multiple sessions for server analysis
- Create batch result management and comparison UI
- Add server health monitoring and status display

**Deliverables:**
- App auto-discovers local FastMODA server on WiFi
- Graceful degradation: full features on-device, enhanced features on server
- No forced internet dependency (pure local network fallback)
- Server processing workflow tested and validated

### Phase 4: Polish, Testing & Optimization (Weeks 12-16)

**Week 12: UI/UX Refinement**
- Implement complete dashboard with real-time metrics
- Create intuitive settings panel for analysis configuration
- Build session history browser with filtering/sorting
- Add help screens and in-app tutorials
- Optimize color schemes and accessibility (WCAG compliance)

**Week 13-14: Comprehensive Testing**
- Unit test suite: >85% code coverage
- Integration testing: BLE ↔ analysis ↔ visualization pipelines
- Battery drain testing: target <5% per hour background operation
- Memory profiling: ensure <150MB resident set
- Physical device testing across iOS/Android versions
- Cross-device testing: old phones, new phones, tablets
- Stress testing: rapid signal changes, network interruptions

**Week 15: Performance Optimization & Polish**
- Profile and optimize hot paths in signal processing
- Implement smooth animations and transitions
- Reduce app bundle size (target <30MB)
- Battery life optimization (BLE power management)
- Ensure offline functionality is robust
- Final UI/UX audits and polish

**Week 16: Beta & Release Preparation**
- Internal alpha testing with development team
- Closed beta with 20-30 external users
- Collect feedback and iterate on UI/UX
- Prepare App Store submission materials
- Create user documentation and video tutorials
- Set up customer support infrastructure

**Deliverables:**
- Production-ready app for iOS and Android
- >85% test coverage
- Performance benchmarks documented
- User documentation and support materials

---

## 5. Cross-Architecture Compilation Strategy

### 5.1 Mobile App Compilation

#### Flutter Build Targets
```bash
# iOS (requires macOS with Xcode)
flutter build ios --release
flutter build ipa --release

# Android
flutter build apk --release --target-platform android-arm64
flutter build apk --release --target-platform android-arm
flutter build apk --release --target-platform android-x64

# Universal APK (includes all architectures)
flutter build apk --release --split-per-abi

# App Bundle (recommended for Play Store)
flutter build appbundle --release
```

#### Supported Architectures

| Platform | Architecture | Notes |
|----------|--------------|-------|
| **iOS** | ARM64 (arm64) | iPhone 5s and later |
| **iOS** | ARM64e | iPhone XS and later (optional) |
| **Android** | ARM64-v8a | 64-bit ARM (most modern devices) |
| **Android** | ARMv7 (armeabi-v7a) | 32-bit ARM (legacy support) |
| **Android** | x86_64 | 64-bit Intel (emulators, tablets) |
| **Android** | x86 | 32-bit Intel (legacy emulators) |

#### Build Matrix
```yaml
# .github/workflows/build.yml
matrix:
  platform: [ios, android]
  arch:
    ios: [arm64]
    android: [arm64-v8a, armeabi-v7a, x86_64]
```

### 5.2 Backend Compilation (FastMODA)

#### Docker Multi-Architecture Builds

```bash
# Build for multiple architectures using buildx
docker buildx create --use
docker buildx build --platform linux/amd64,linux/arm64 \
  -t moda/fastmoda:latest \
  --push .

# Supported platforms
- linux/amd64    # x86_64 servers, cloud instances
- linux/arm64    # ARM servers, Apple Silicon, Raspberry Pi 4+
- linux/arm/v7   # Raspberry Pi 3, older ARM devices
```

#### Native Python Builds

For edge deployment on mobile devices or embedded systems:

```bash
# iOS (using Kivy-iOS or BeeWare)
# Note: Limited support, recommended to use API approach

# Android (using Buildozer/python-for-android)
buildozer android debug
# Builds APK with embedded Python runtime
# Architecture: armeabi-v7a, arm64-v8a, x86, x86_64
```

### 5.3 Cross-Compilation Tools

#### For Backend Services

| Tool | Purpose | Architectures |
|------|---------|---------------|
| **Docker Buildx** | Multi-arch container builds | amd64, arm64, arm/v7 |
| **QEMU** | Cross-architecture emulation | All major architectures |
| **PyInstaller** | Python to executable | Windows, macOS, Linux |
| **Nuitka** | Python to C compiler | Platform-specific |

#### For Mobile

| Tool | Purpose | Output |
|------|---------|--------|
| **Xcode** | iOS compilation | IPA (ARM64) |
| **Android Studio/Gradle** | Android compilation | APK/AAB (ARM, x86) |
| **Flutter** | Cross-platform builds | iOS & Android binaries |

### 5.4 Build Automation

```yaml
# Example CI/CD for cross-architecture builds
name: Multi-Platform Build

on: [push, pull_request]

jobs:
  build-ios:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v3
      - uses: subosito/flutter-action@v2
      - run: flutter build ios --release --no-codesign

  build-android:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        arch: [arm64-v8a, armeabi-v7a, x86_64]
    steps:
      - uses: actions/checkout@v3
      - uses: subosito/flutter-action@v2
      - run: flutter build apk --release --target-platform android-${{ matrix.arch }}

  build-backend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: docker/setup-buildx-action@v2
      - run: |
          docker buildx build \
            --platform linux/amd64,linux/arm64 \
            --tag fastmoda:${{ github.sha }} \
            --push \
            ./FastMODA
```

---

## 6. Testing Strategy

See `TESTING_STRATEGY.md` for comprehensive testing plan.

**Summary:**
- **Unit Tests:** Flutter/Dart tests for business logic
- **Widget Tests:** UI component testing
- **Integration Tests:** End-to-end flow testing
- **BLE Tests:** Mock and real device testing
- **API Tests:** FastMODA endpoint validation
- **Performance Tests:** Battery, memory, processing time
- **Cross-Device Tests:** Physical device farm (iOS/Android)

**Target Coverage:** 80%+ code coverage

---

## 7. Deployment Architecture

### 7.1 App Deployment Strategy

**Target: Independent Handset Operation**

The mobile app is designed to work completely autonomously on the handset without any external dependencies:

- **No internet required** for core functionality
- **No cloud services required** for data analysis
- **Works offline immediately** after installation
- **All core analysis on-device** running on ARM64 CPU
- **Optional server enhancement** if local FastMODA server available on home WiFi

### 7.2 On-Device Deployment

**Built-in Libraries (ARM-Optimized):**
- NumPy (precompiled wheels for ARM64 from official sources)
- SciPy.signal (for DSP operations)
- Dart-based UI rendering (Flutter)
- SQLite3 (for local storage)
- mDNS client (for service discovery)

**Storage Requirements:**
- App bundle: ~25-35 MB (including all ARM64 code)
- Cache: ~10-50 MB (user's session history, thumbnail caches)
- Signal storage: Unlimited (expansion to external storage if needed)

**Performance Profile:**
- Cold start: <2 seconds
- BLE connection: <3 seconds
- FFT analysis (1024 samples): <100 ms
- MODWT analysis (1024 samples, 2 levels): <500 ms
- Memory footprint: 80-150 MB RSS

### 7.3 Optional Self-Hosted Server Deployment (Local Fallback)

For users who want advanced features or have their own local server hardware:

**Hardware Requirements:**
```
Minimum:
  - Raspberry Pi 4 (4GB RAM+), ARMv8 64-bit
  - 32GB microSD card
  - Local WiFi connectivity

Recommended:
  - Raspberry Pi 5 (8GB RAM), ARMv8 64-bit
  - SSD (256GB+) for history archival
  
Optional (GPU Acceleration):
  - NVIDIA Jetson Nano 2GB (4GB+ recommended)
  - Jetson Orin Nano (latest, >10x faster)
```

**Deployment Model:**
```bash
# User installs FastMODA server on their Raspberry Pi
docker pull luphysics/moda-fastmoda:arm64-latest
docker run -d \
  --name fastmoda \
  -p 5000:5000 \
  -v /home/pi/moda-data:/app/data \
  luphysics/moda-fastmoda:arm64-latest
```

**Automatic Discovery:**
- App sends mDNS query: `_fastmoda._tcp.local`
- Server responds with: `moda-server.local:5000`
- App displays: "Found local analysis server - Enable for advanced features?"
- Connection is automatic once enabled

**Server Capabilities:**
- Full MODWT wavelet decomposition (all levels)
- Phase and magnitude coherence
- Bispectrum analysis
- Bayesian signal inference
- Multi-signal joint analysis
- Batch processing of archived sessions
- GPU acceleration (if Jetson hardware)
- Historical analysis and statistics

**Server Network Model:**
- **Local-first:** Operates entirely on home WiFi (no internet needed)
- **mDNS for discovery:** Works behind any home router, no port forwarding
- **HTTP REST API:** Standard requests, no special ports
- **Optional remote access:** User can expose via VPN if desired (not automatic)

### 7.4 Cloud Deployment (Not Recommended for MVP)

**Cloud Option Explicitly Not Prioritized:**
- No automatic cloud uploads
- No cloud-dependent workflows
- Cloud usage only if user explicitly opts in
- Should be presented as "optional archival" service

**If Cloud is Added Later:**
- Would use existing FastMODA API on AWS/GCP
- Would require explicit user consent per upload
- Would be completely independent of core app functionality
- Could be disabled without breaking app features

### 7.5 Deployment Checklist

**App Release (iOS & Android):**
- [ ] All dependencies pre-compiled and arm64-verified
- [ ] No required internet connectivity
- [ ] mDNS client tested and working
- [ ] BLE works on minimum supported iOS/Android versions
- [ ] Offline analysis fully functional
- [ ] Server discovery (if available) optional and transparent
- [ ] Battery drain benchmarked (<5%/hour background)
- [ ] App sandbox security audit passed
- [ ] Privacy policy: No cloud collection unless user opts in

**Optional Server Setup (Raspberry Pi):**
- [ ] Docker image for linux/arm64 built and tested
- [ ] mDNS broadcast enabled on startup
- [ ] Health check endpoint implemented
- [ ] Documentation: step-by-step setup guide
- [ ] Verification script: confirm server accessibility from phone
- [ ] One-click Docker Compose file provided

---

## 8. Performance Requirements

### 8.1 Mobile App (On-Device Targets)

| Metric | Target | Critical | Notes |
|--------|--------|----------|-------|
| **Cold Start Time** | <2s | <3s | First launch after install |
| **BLE Connection** | <3s | <5s | Discovery + pairing + connection |
| **Signal Latency** | <100ms | <200ms | Bluetooth RX to buffer |
| **FFT Processing (1024 samples)** | <100ms | <200ms | Real-time analysis |
| **MODWT Processing (2-3 level)** | <500ms | <1s | Wavelet decomposition |
| **App Memory Usage** | <150MB | <250MB | During active analysis |
| **Battery Drain (Bluetooth active)** | <8%/hour | <12%/hour | Streaming + analysis |
| **Battery Drain (idle, recording)** | <3%/hour | <5%/hour | Just receiving data |
| **App Bundle Size** | <30MB | <50MB | Download + install size |

**Analysis Latency (Total User-Perceived):**
- Quick analysis (FFT): <500ms from signal arrival to display
- Wavelet analysis: <1.5s from signal arrival to display
- Export to CSV: <2s (local storage write)

### 8.2 Self-Hosted Server (When Available)

| Task | Target | Hardware |
|------|--------|----------|
| **Full MODWT (all levels)** | <10s | Raspberry Pi 4 |
| **Phase Coherence** | <15s | Raspberry Pi 4 |
| **Bispectrum Analysis** | <20s | Raspberry Pi 4 |
| **Bayesian Inference** | <30s | Raspberry Pi 4 |
| **GPU Acceleration** | 2-5x speedup | NVIDIA Jetson Nano |
| **Batch Process (10 signals)** | <5min | Raspberry Pi 4 |
| **Concurrent Clients** | 3-5 | Raspberry Pi 4 |

### 8.3 Bluetooth Performance

- **Packet Delivery Rate:** >99% (with retransmit)
- **Connection Stability:** >95% over 1-hour test
- **Latency (Device → Phone):** <100ms (typical BLE)
- **Data Rate:** Up to 240 byte/sec (typical BLE 4.2)

---

## 9. Cost Estimates

### 9.1 Development Costs

| Phase | Hours | Focus |
|-------|-------|-------|
| Project Setup & Planning | 60h | BLE, on-device architecture |
| On-Device Signal Processing | 180h | ARM-optimized FFT, MODWT, DSP |
| BLE Integration & Buffers | 100h | Real-time data acquisition |
| Visualization & UI | 150h | Interactive charts, dashboard |
| Server Integration (optional fallback) | 80h | mDNS discovery, API client |
| Testing & Optimization | 140h | Battery, memory, CPU profiling |
| Deployment & Documentation | 70h | App Store submission, guides |
| **Total MVP** | **780h** | Fully functional on-device app |

**Optional Server Component** (add if user has hardware):
- Docker image preparation: 40h
- FastMODA API containerization: 30h
- Server documentation: 20h
- **Total Server Setup** | **90h** | Ready-to-deploy ARM64 container

### 9.2 Infrastructure Costs (Monthly)

**App-Only Deployment (Recommended MVP):**
- **No server costs** (all processing on handset)
- **App Store fees:** $99/year (Apple) + $25 one-time (Google)
- **CI/CD:** GitHub Actions - free (includes ARM64 builds)
- **Distribution:** Free (App Store + Google Play)
- **Total:** ~$10/month

**Optional: Self-Hosted Server Setup (One-Time):**
- **Raspberry Pi 4 (4GB):** $60
- **Raspberry Pi 5 (8GB):** $80
- **Power supply:** $15
- **Case + cooling:** $15
- **SD Card (128GB):** $20
- **Total per device:** $110-130

**Optional: Cloud Backend (Not Recommended for MVP):**
- Should only be added if user explicitly requests cloud archival
- Estimated cost: $50-200/month (varies with usage)
- Should be clearly optional and disabled by default

### 9.3 Licensing & Compliance

- **NumPy/SciPy:** Free (BSD license)
- **Flutter:** Free (BSD license)
- **FastMODA:** Free (check existing license)
- **App Store:** $99/year developer account
- **Google Play:** $25 developer account

**Total Startup Cost:** $124 first year, $99/year thereafter

---

## 10. Risk Assessment & Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **On-Device Performance Too Slow** | Medium | High | Pre-benchmark on ARM devices; optimize critical paths; use native FFT libraries |
| **Limited On-Device Features** | Medium | Medium | Self-hosted server fallback for advanced features; clear UI about capabilities |
| **BLE Connectivity Issues** | High | High | Extensive device testing; implement reconnect logic; fallback to queued processing |
| **ARM Library Compatibility** | Medium | High | Use official NumPy ARM64 wheels; test on multiple ARM phones; maintain fallback |
| **Battery Drain** | Medium | Medium | Profile power consumption; optimize BLE parameters; offload when server available |
| **Storage Space Limitations** | Low | Medium | Implement sliding window for recordings; allow external storage expansion |
| **Server Discovery Fails** | Low | Medium | Graceful degradation to on-device; don't break if server unavailable |
| **Cross-Platform Bugs** | Medium | Medium | Automated testing on device farm; real hardware testing not just emulators |
| **App Store Rejection** | Low | High | Strict guideline compliance; pre-submission review; privacy-first design |
| **User Adoption** | Medium | Medium | Free app; no forced cloud requirement; closed beta with real users |
| **Legacy Device Support** | Medium | Low | Test on older phones; graceful degradation for limited hardware |

**Key Mitigation Strategy:**
- On-device processing handles 90%+ of use cases
- Server fallback for advanced features (optional, not required)
- No cloud lock-in or mandatory internet dependency
- Works completely offline immediately after installation

---

## 11. Success Metrics

### 11.1 Technical Metrics (On-Device Performance)

- **BLE connection success rate:** >95%
- **On-device analysis completion rate:** >98%
- **FFT processing latency (1024 samples):** <100ms
- **MODWT processing latency (2 levels):** <500ms
- **App crash rate:** <0.1%
- **Memory stability:** No leaks over 1-hour test
- **Battery drain (active):** <8% per hour
- **Offline functionality:** 100% (no internet required for core features)

### 11.2 Server Integration Metrics (When Available)

- **Server auto-discovery:** >95% success on local network
- **Server processing speed:** Full MODWT <10s on Raspberry Pi 4
- **Graceful fallback:** App continues normally if server unavailable
- **Server uptime:** >99% (when user hosts)

### 11.3 User Satisfaction Metrics

- **App Store Rating:** >4.0 stars (iOS & Android)
- **Session completion rate:** >95% (successful analysis completion)
- **Average session duration:** >5 minutes
- **BLE device connection success (first-time user):** >90%
- **Return user rate (Day 7):** >40%
- **Return user rate (Day 30):** >20%

### 11.4 Adoption Metrics

- **Month 1:** 500+ downloads (closed beta + early adopters)
- **Month 3:** 2,000+ active monthly users
- **Month 6:** 5,000+ active monthly users
- **Server adoption:** 10%+ of users set up local FastMODA instance

---

## 12. Next Steps

### Immediate Actions (Week 1)
1. **Approve architecture** - On-device first with self-hosted server fallback
2. **Assemble team:** 2-3 Flutter/mobile developers, 1 DSP engineer, 1 UI/UX designer, 1 QA
3. **Prepare ARM development environment:**
   - NumPy/SciPy ARM64 wheel sourcing
   - Flutter install with ARM target support
   - BLE library evaluation and selection
4. **Create Bluetooth protocol specification** (see `BLUETOOTH_PROTOCOL.md`)
5. **Design UI/UX mockups** (Figma) - focus on offline-first design

### Week 2-3
6. **Set up Flutter project** with on-device signal processing framework
7. **Initialize CI/CD pipeline** for multi-architecture builds (arm64, armv7, x86)
8. **Begin BLE integration** with test device simulator
9. **Prototype on-device FFT** with performance benchmarking
10. **Create server integration API contract** (for Phase 3)

### Week 4 Milestone
- Working proof-of-concept: Bluetooth data → On-device FFT → Dashboard display
- Performance benchmarks on real ARM devices
- Team consensus on architecture and technology choices

### Important Notes for Product Owners
- **No internet dependency** - app works offline, period
- **No cloud lock-in** - all data stays on device unless user explicitly exports
- **Optional enhancements** - server processing is a nice-to-have, not required
- **Minimal operational cost** - no backend to maintain (user runs their own if desired)
- **Privacy-first** - no user data leaves device without explicit action

---

## 13. Appendices & References

- **A. Bluetooth Protocol:** See `BLUETOOTH_PROTOCOL.md` (custom MODA BLE specification)
- **B. Testing Strategy:** See `TESTING_STRATEGY.md` (comprehensive test plan)
- **C. ARM Build Guide:** Instructions for compiling NumPy/SciPy for ARM64
- **D. FastMODA API:** See `FastMODA/API.md` (server endpoints and data contracts)
- **E. Docker Setup:** FastMODA self-hosted deployment on Raspberry Pi
- **F. Signal Processing:** Technical details of on-device algorithms (FFT, MODWT, changepoint detection)

---

## 14. Architecture Decision Records (ADR)

### ADR-001: On-Device Processing as Primary

**Decision:** Prioritize on-device signal processing on the handset.

**Rationale:**
- No internet dependency (essential for medical/portable use)
- Immediate results (no network latency)
- Privacy by design (data never leaves device)
- Reduced operational costs (no servers to maintain)
- Better user experience (offline-first)

**Alternatives Considered:**
- Cloud-first (rejected: requires internet, privacy concerns, latency)
- Hybrid cloud-default (rejected: no clear advantage over on-device primary)

**Consequences:**
- Limited to algorithms suitable for ARM CPUs
- Requires careful optimization for battery life
- More complex mobile development

### ADR-002: Self-Hosted Server as Fallback

**Decision:** Optional self-hosted FastMODA server on local WiFi for advanced features.

**Rationale:**
- No cloud dependency or signup required
- User runs their own hardware (privacy + control)
- Transparent fallback (app works without it)
- Leverages existing FastMODA codebase
- Good for users with computing infrastructure at home

**Hardware:**
- Minimum: Raspberry Pi 4 (~$60)
- Recommended: Raspberry Pi 5 (~$80)
- Optional GPU: NVIDIA Jetson Nano

**Deployment:** Docker container, auto-discovery via mDNS

### ADR-003: No Cloud Backend for MVP

**Decision:** Do not include cloud services in MVP.

**Rationale:**
- Not necessary for core functionality
- Adds complexity and operational cost
- Privacy concerns (especially for medical data)
- Defers cloud decision to later phases if actually needed

**Future Option:** Can be added if users explicitly request it

---

## 15. Technical Debt & Future Optimizations

### Short Term (Phase 2)
- [ ] Profile on-device memory usage under load
- [ ] Optimize MODWT for faster decomposition
- [ ] Implement adaptive buffer sizing based on available RAM
- [ ] Add proper error handling for low-memory conditions

### Medium Term (Phase 3+)
- [ ] GPU acceleration on devices with Mali/Adreno GPU (via NNAPI)
- [ ] TensorFlow Lite for on-device ML-based classification
- [ ] Advanced caching strategies for repeated analysis
- [ ] Battery optimization: CPU frequency scaling coordination

### Long Term
- [ ] WASM version for web-based analysis
- [ ] Native C++ rewrite of critical DSP paths (if needed)
- [ ] Multi-threaded processing for multi-channel signals
- [ ] Integration with wearable devices and Apple Watch

---

**Document Version:** 2.0 (Revised - On-Device First)
**Last Updated:** 2026-03-05
**Author:** MODA Development Team

**Key Changes (v1.0 → v2.0):**
- Reorganized to prioritize on-device processing as PRIMARY
- Self-hosted server moved to explicit FALLBACK role
- Removed cloud services from MVP scope
- Expanded on-device capability specifications
- Added "no internet dependency" as core requirement
- Updated roadmap to focus on mobile-first development
- Added risk mitigation for on-device CPU limitations
- Clarified cost structure (minimal operational costs)
