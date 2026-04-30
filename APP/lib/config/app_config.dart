// Pre-baked API key for FastMODA server authentication.
// The server reads FASTMODA_API_KEY from its environment (injected via Helm secret).
// Both values must match — rotate by rebuilding the app and redeploying the chart.
const String kFastModaApiKey =
    'moda_8e6695088c2e3114cbb25e3554544f2577cd53c58a3672ac';

// Base URL for the hosted FastMODA instance.
// Users can override this in Settings; this is the compiled-in default.
const String kFastModaDefaultUrl = 'https://moda.example.com';

const Duration kApiTimeout = Duration(seconds: 30);
const Duration kAnalysisReceiveTimeout = Duration(seconds: 120);
const Duration kPollInterval = Duration(milliseconds: 500);

const double kDefaultSampleRate = 256.0;
const int kSpectrumWindowSize = 256;
const int kSignalBufferSize = 512;

// ── MODA-BLE-SP Protocol UUIDs ──────────────────────────────────────────────
// "MODA" in ASCII hex = 4d6f6461; used as the first segment of each UUID.
// These must match the firmware on your MODA sensor hardware.
const String kModaServiceUuid      = '4d6f6461-0000-1000-8000-00805f9b34fb';
const String kModaSignalDataUuid   = '4d6f6461-0001-1000-8000-00805f9b34fb';
const String kModaSignalConfigUuid = '4d6f6461-0002-1000-8000-00805f9b34fb';
const String kModaControlUuid      = '4d6f6461-0003-1000-8000-00805f9b34fb';
const String kModaStatusUuid       = '4d6f6461-0004-1000-8000-00805f9b34fb';
const String kModaDeviceInfoUuid   = '4d6f6461-0005-1000-8000-00805f9b34fb';
