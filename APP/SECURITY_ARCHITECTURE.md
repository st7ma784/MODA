# MODA Mobile App: Security Architecture & Implementation Plan

## Executive Summary

This document defines the comprehensive security architecture for the MODA mobile application including:
- **Authentication & Authorization:** App-device binding with JWT tokens and HMAC verification
- **Data Encryption:** AES-256-GCM for data at rest, TLS 1.3 for transport
- **Secure Storage:** Encrypted SQLite with platform-specific keystores
- **Bluetooth Security:** BLE pairing, encrypted channels, MAC filtering
- **API Protection:** Certificate pinning, device fingerprinting, request signing
- **Server Endpoints:** Multi-layer protection against unauthorized access

**Threat Model:** Prevent unauthorized data access, session hijacking, man-in-the-middle attacks, and device spoofing.

---

## 1. Authentication & Authorization

### 1.1 App-Device Binding Architecture

The mobile app uses a **device-scoped authentication model** where each app installation is uniquely identified and bound to the backend.

```
┌─────────────────────────────────────────────────────────────────┐
│                        MODA Mobile App                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Device Key Storage (Platform Keystore)                  │  │
│  │  ├─ Device GUID (immutable, per install)                │  │
│  │  ├─ Device Private Key (RSA-2048)                       │  │
│  │  ├─ Device Public Certificate                          │  │
│  │  └─ Keystore PIN/Biometric Lock                        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          │                                       │
│                          ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  JWT Token Manager                                       │  │
│  │  ├─ Access Token (short-lived: 15 min)                 │  │
│  │  ├─ Refresh Token (long-lived: 7 days)                │  │
│  │  ├─ Token Signing Key (HMAC-SHA256 local)             │  │
│  │  └─ Token Expiry Tracking                              │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          │                                       │
│                          ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Session Manager                                         │  │
│  │  ├─ Device Fingerprint                                 │  │
│  │  ├─ Challenge-Response Cache                          │  │
│  │  ├─ OAuth State Parameters                            │  │
│  │  └─ Logged-In Status                                  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Initial App Registration Flow

**First-Launch Registration (One-Time):**

```
User installs app
        │
        ▼
┌─────────────────────┐
│ Generate Device Key │  Generate RSA-2048 key pair locally
│ - Private Key       │  Store in Keystore with encryption
│ - Public Cert       │
└─────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│ Create Device GUID              │  UUID v4, unique per install
│ (immutable for lifetime)        │
└─────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────────────┐
│ Send Registration Request to /api/auth/register    │
│ {                                                  │
│   "device_guid": "<UUID>",                        │
│   "device_public_cert": "<base64 PEM>",           │
│   "device_name": "iPhone 14 Pro",                 │
│   "app_version": "1.0.0",                         │
│   "os_version": "17.2",                           │
│   "timestamp": 1234567890,                        │
│   "signature": "<RSA signature of above>"         │
│ }                                                  │
└────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────┐
│ Server Validates & Stores                            │
│ - Verify RSA signature                              │
│ - Check certificate validity                        │
│ - Store device_guid → public_cert mapping           │
│ - Create server-side device record                  │
└──────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────┐
│ Server Responds: OAuth Authorization URL │
│ {                                        │
│   "auth_url": "<redirect to OAuth flow>",│
│   "state": "<CSRF token>",              │
│   "challenge": "<16-byte random>"       │
│ }                                        │
└──────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│ User Authenticates (OAuth 2.0)      │
│ - Via webview or external browser   │
│ - Returns auth_code                 │
└─────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────┐
│ Exchange auth_code for Tokens        │
│ POST /api/auth/token                 │
│ {                                    │
│   "grant_type": "authorization_code",│
│   "code": "<auth_code>",            │
│   "device_guid": "<UUID>",          │
│   "device_signature": "<RSA sig>",  │
│   "code_verifier": "<PKCE>"         │
│ }                                    │
└──────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────┐
│ Server Issues JWT Tokens            │
│ - Access Token (15 min)            │
│ - Refresh Token (7 days)           │
│ - Device binding confirmed         │
└────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│ App Stores Tokens in Keystore       │
│ (Encrypted at rest)                 │
└─────────────────────────────────────┘
```

### 1.3 JWT Token Structure

**Access Token Payload:**
```json
{
  "iss": "moda-auth-server",
  "sub": "device_guid_<UUID>",
  "aud": "moda-api",
  "iat": 1234567890,
  "exp": 1234568890,
  "device_guid": "<UUID>",
  "device_fingerprint": "<SHA256 hash>",
  "token_family": "access",
  "jti": "<unique token ID>",
  "scope": "signals:read signals:write analysis:read"
}
```

**Refresh Token Payload:**
```json
{
  "iss": "moda-auth-server",
  "sub": "device_guid_<UUID>",
  "aud": "moda-api",
  "iat": 1234567890,
  "exp": 1234826890,
  "device_guid": "<UUID>",
  "token_family": "refresh",
  "jti": "<unique token ID>",
  "rotation_count": 0
}
```

**Token Signing:**
- Algorithm: HS256 (HMAC-SHA256) for API validation
- Key: Server-side secret (never sent to app)
- Verification: Server validates signature on every request

### 1.4 Session Management

**Token Refresh Flow:**
```
Access Token Expires (15 minutes)
        │
        ▼
┌──────────────────────────────────┐
│ Check Refresh Token (7 days TTL) │
└──────────────────────────────────┘
        │
    ┌───┴──────┬──────────────┐
    │          │              │
Valid      Expired      Invalid/Revoked
    │          │              │
    ▼          ▼              ▼
  Use      Re-auth         Clear tokens
 Refresh    Required        Log out
 Token                      User
    │
    ▼
┌────────────────────────────────────┐
│ POST /api/auth/refresh-token       │
│ {                                  │
│   "refresh_token": "<JWT>",       │
│   "device_guid": "<UUID>",        │
│   "device_fingerprint": "<hash>" │
│ }                                  │
└────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────┐
│ Server Validates:                        │
│ - Token signature and expiry            │
│ - Device fingerprint matches            │
│ - Token not blacklisted                 │
│ - Token rotation counts OK              │
└──────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│ Issue New Token Pair            │
│ - New Access Token              │
│ - New Refresh Token (rotated)  │
└─────────────────────────────────┘
```

### 1.5 Device Fingerprinting

Prevents token theft and unauthorized device usage:

```
Device Fingerprint = SHA256(
  device_guid +
  model_identifier +     (iPhone 14 Pro, Samsung Galaxy S23, etc.)
  os_version +           (17.2, 14.0, etc.)
  app_version +          (1.0.0)
  build_id +             (immutable)
  hardware_serial_hash   (one-way hash of serial if available)
)
```

**Verification:** Included in every API request header, validated server-side.

---

## 2. Data Storage & Encryption

### 2.1 Storage Architecture

```
┌─────────────────────────────────────────────────────┐
│  MODA Mobile App Storage Hierarchy                  │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌─────────────────────────────────────────────┐  │
│  │ 1. Platform Keystore (Secure Enclave)      │  │
│  │    ├─ Device Private Key (RSA-2048)        │  │
│  │    ├─ Master Encryption Key (AES-256)      │  │
│  │    ├─ JWT Tokens (encrypted)               │  │
│  │    └─ Biometric Authentication              │  │
│  │                                              │  │
│  │    iOS: Keychain                           │  │
│  │    Android: AndroidKeyStore + Encrypted .. │  │
│  │    Protection: Hardware-backed when available
│  └─────────────────────────────────────────────┘  │
│           ▲          ▲          ▲                 │
│           │          │          │                 │
│  ┌─────────────────────────────────────────────┐  │
│  │ 2. Encrypted SQLite Database                │  │
│  │    ├─ Signal Data (streaming archive)       │  │
│  │    ├─ Analysis Results (with metadata)      │  │
│  │    ├─ Session History                       │  │
│  │    ├─ Cached BLE Device Info                │  │
│  │    ├─ User Preferences                      │  │
│  │    └─ Sync Status Tracking                  │  │
│  │                                              │  │
│  │    Encryption: AES-256-GCM                 │  │
│  │    - Database key derived from Master Key  │  │
│  │    - Per-table initialization vectors      │  │
│  │    - Authenticated encryption               │  │
│  └─────────────────────────────────────────────┘  │
│                                                    │
│  ┌─────────────────────────────────────────────┐  │
│  │ 3. Memory Cache (Session Only)              │  │
│  │    ├─ Current Bluetooth Connection          │  │
│  │    ├─ Active Analysis Results               │  │
│  │    ├─ API Response Cache (5-min TTL)        │  │
│  │    ├─ Decrypted Signal Buffers              │  │
│  │    └─ Cryptographic Keys (temporary)        │  │
│  │                                              │  │
│  │    Cleared: On encrypt/decrypt cycle        │  │
│  │    Memory-locked: MLOCK where possible      │  │
│  │    Timeout: Auto-clear after 15 min idle   │  │
│  └─────────────────────────────────────────────┘  │
│                                                    │
│  ┌─────────────────────────────────────────────┐  │
│  │ 4. TMP/Cache Directories (Encrypted)        │  │
│  │    ├─ Exported Files (CSV, MAT, PNG)        │  │
│  │    ├─ Downloaded Analysis Results           │  │
│  │    └─ Temporary Processing Buffers          │  │
│  │                                              │  │
│  │    Auto-delete: On app backgrounding        │  │
│  │    Encryption: App-specific directory       │  │
│  └─────────────────────────────────────────────┘  │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 2.2 Encrypted SQLite Implementation

**Database Setup:**
```sql
-- Initialize encrypted database (iOS - SQLCipher)
PRAGMA key = 'encryption_key_from_keystore';
PRAGMA cipher = 'aes-256-gcm';
PRAGMA integrity_check;

-- Create tables
CREATE TABLE signal_data (
  id INTEGER PRIMARY KEY,
  session_id TEXT NOT NULL,
  timestamp INTEGER NOT NULL,
  signal_data BLOB NOT NULL,  -- AES-256-GCM encrypted
  sample_rate INTEGER,
  channel_count INTEGER,
  device_id TEXT,
  synced BOOLEAN DEFAULT 0,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE analysis_results (
  id INTEGER PRIMARY KEY,
  session_id TEXT NOT NULL,
  analysis_type TEXT NOT NULL,  -- 'fft', 'modwt', 'coherence'
  result_data BLOB NOT NULL,     -- JSON-serialized, AES-256-GCM
  metadata BLOB NOT NULL,        -- AES-256-GCM encrypted
  synced BOOLEAN DEFAULT 0,
  sync_timestamp INTEGER,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE ble_devices (
  id INTEGER PRIMARY KEY,
  device_uuid TEXT UNIQUE NOT NULL,
  device_name TEXT,
  device_address TEXT,
  trusted BOOLEAN DEFAULT 0,
  pairing_key BLOB NOT NULL,  -- AES-256-GCM encrypted
  last_connected INTEGER,
  connection_count INTEGER,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE api_cache (
  id INTEGER PRIMARY KEY,
  endpoint TEXT NOT NULL,
  cache_key TEXT NOT NULL,
  response_data BLOB NOT NULL,  -- AES-256-GCM encrypted
  expires_at INTEGER NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes (can be in plaintext)
CREATE INDEX idx_signal_session ON signal_data(session_id);
CREATE INDEX idx_analysis_session ON analysis_results(session_id);
CREATE INDEX idx_ble_device ON ble_devices(device_uuid);
CREATE INDEX idx_cache_expires ON api_cache(expires_at);
```

### 2.3 Encryption Key Hierarchy

```
┌────────────────────────────────┐
│ Master Encryption Key (MEK)    │
│ - 256-bit random key           │
│ - Stored in Platform Keystore  │
│ - Protected by biometric lock  │
│ - Never transmitted            │
└────────────────────────────────┘
         │
    ┌────┴────┬──────────────┬────────────┐
    │         │              │            │
    ▼         ▼              ▼            ▼
┌────────┐ ┌────────┐ ┌──────────┐ ┌────────┐
│Database│ │Keystore│ │API Cache │ │ Temp  │
│KEK     │ │Tokens  │ │Encryption│ │Files  │
└────────┘ └────────┘ └──────────┘ └────────┘
    │         │          │           │
    ▼         ▼          ▼           ▼
  M-DEK    T-DEK      C-DEK        F-DEK
```

**Key Derivation (PBKDF2-based):**
```
Database_Encryption_Key = PBKDF2(
  password = MEK (256-bit),
  salt = SHA256(device_guid),
  iterations = 100000,
  length = 256 bits,
  hash_algorithm = SHA-256
)

API_Cache_Key = PBKDF2(
  password = MEK,
  salt = SHA256(device_guid + "api_cache"),
  iterations = 100000,
  length = 256 bits,
  hash_algorithm = SHA-256
)
```

### 2.4 Per-Record Encryption

**Signal Data Encryption:**
```
For each signal record:

1. Generate random IV (16 bytes)
2. Encrypt payload with AES-256-GCM:
   ciphertext, tag = AES_GCM_Encrypt(
     key = Database_Encryption_Key,
     nonce = IV,
     plaintext = signal_bytes,
     aad = session_id + timestamp
   )

3. Store in database:
   stored_blob = IV || ciphertext || tag

4. On retrieval:
   IV, ciphertext, tag = parse_blob(stored_blob)
   plaintext = AES_GCM_Decrypt(
     key = Database_Encryption_Key,
     nonce = IV,
     ciphertext = ciphertext,
     tag = tag,
     aad = session_id + timestamp
   )
```

### 2.5 Data Lifecycle & Cleanup

**Retention Policy:**
```
Signal Data:
  - In SQLite: 90 days (configurable)
  - In memory: 5 minutes (active analysis)
  - After export: Deleted immediately unless user saves

Analysis Results:
  - In SQLite: Indefinite (can be synced to server)
  - In memory: 15 minutes after viewing
  - Synced to server: Marked with sync_timestamp

API Cache:
  - TTL: 5 minutes (API response dependent)
  - Auto-cleared: On token refresh
  - Explicit clear: On logout

Keystore Items:
  - Tokens: Deleted on logout or expiry + grace period
  - Keys: Lifetime of app installation
  - Biometric: Cleared if biometric revoked
```

**Secure Deletion:**
```
function secureDelete(data_blob, encryption_key) {
  // Overwrite with random data multiple times
  for i = 0 to 3:
    overwrite_with_random(data_blob)
  
  // Final zero pass
  memset(data_blob, 0, length)
  
  // Deallocate memory
  free(data_blob)
}
```

---

## 3. Bluetooth Security

### 3.1 BLE Pairing & Connection

**Pairing Protocol:**
```
┌─────────────────────────────────────────┐
│ Step 1: Initial Discovery               │
│ - Scan for BLE devices                  │
│ - Display advertised name and signal    │
│ - User selects device to pair           │
└─────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────┐
│ Step 2: Verify Device Identity           │
│ - Check MAC address against whitelist    │
│ - Calculate device fingerprint:          │
│   fingerprint = SHA256(                 │
│     mac_address +                       │
│     advertised_name +                   │
│     device_uuid                         │
│   )                                      │
│ - Present fingerprint for user confirmation
└──────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────┐
│ Step 3: Enable LMP/LE Security Features  │
│ - BLE version 5.0+ with AES-CCM         │
│ - Authenticated & encrypted link        │
│ - Key size: 128-bit minimum             │
│ - Pairing: LE Secure Connections (ECDH)│
└──────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────┐
│ Step 4: Store Pairing Keys               │
│ - Generate pairing key (16 bytes random)│
│ - Encrypt with Master Encryption Key:   │
│   encrypted_pairing_key = AES_GCM(      │
│     key = MEK,                          │
│     nonce = device_uuid,               │
│     plaintext = pairing_key             │
│   )                                      │
│ - Store in SQLite with device record    │
└──────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────┐
│ Step 5: Establish Secure Connection      │
│ - All data encrypted with pairing key   │
│ - MAC filtering enabled                 │
│ - Connection timeout: 30 seconds        │
└──────────────────────────────────────────┘
```

### 3.2 BLE Characteristic Security

**Data Transmission:**
```
Bluetooth Characteristics:

1. Signal Data Characteristic (UUID: 12345678-...)
   ├─ Permission: Encrypted Read
   ├─ Notification: Enabled (encrypted)
   ├─ Data format: Raw bytes (PCM audio/analog)
   └─ Update rate: 100 Hz - 1000 Hz (device dependent)

2. Configuration Characteristic (UUID: 87654321-...)
   ├─ Permission: Encrypted Read/Write
   ├─ Data: Sample rate, channel config
   └─ Protocol version handshake

3. Status Characteristic (UUID: aabbccdd-...)
   ├─ Permission: Encrypted Read
   ├─ Data: Battery level, connection quality
   └─ Notification: Enabled

App-side handling:
  - All reads/writes use encrypted links only
  - Verify pairing before any operations
  - Monitor RSSI (signal strength) for tampering
  - Drop connection if RSSI drops >20dB suddenly
```

### 3.3 Trusted Device Whitelist

**MAC Address Filtering:**
```sql
CREATE TABLE ble_whitelist (
  id INTEGER PRIMARY KEY,
  device_uuid TEXT NOT NULL,
  device_mac_address TEXT NOT NULL,
  device_name TEXT,
  device_fingerprint TEXT NOT NULL,
  trusted BOOLEAN DEFAULT 1,
  added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  last_verified INTEGER
);

-- On connection attempt:
SELECT * FROM ble_whitelist 
WHERE device_mac_address = incoming_address 
  AND trusted = 1 
  AND device_fingerprint = calculated_fingerprint;

-- If not found: Reject connection
-- If found: Proceed with pairing
-- If fingerprint mismatch: Alert user, reject
```

### 3.4 Connection Quality Monitoring

```
Every 1000ms during active connection:
┌──────────────────────────────────┐
│ Check Connection Health:         │
│ - Measure RSSI (signal strength) │
│ - Track packet loss rate         │
│ - Monitor latency               │
│ - Check MTU size                │
└──────────────────────────────────┘
        │
    ┌───┴────────────────────┐
    │                        │
  Healthy             Degraded/Failed
    │                        │
    ▼                        ▼
 Continue         ┌──────────────────┐
 Operation        │ Attempt Reconnect│
                  │ - Exponential    │
                  │   backoff        │
                  │ - Max 3 retries  │
                  │ - Then alert user│
                  └──────────────────┘
```

---

## 4. API Security & Endpoint Protection

### 4.1 Request Authentication

**Every API request includes:**

```
Header: Authorization
Value: Bearer <JWT_ACCESS_TOKEN>

Header: X-Device-Signature
Value: Base64(HMAC-SHA256(
  request_body + 
  endpoint + 
  request_timestamp + 
  device_guid,
  shared_secret_derived_from_device_private_key
))

Header: X-Device-Fingerprint
Value: SHA256(
  device_guid + 
  model + 
  os_version +
  app_version +
  build_id
)

Header: X-Request-Timestamp
Value: Unix timestamp (within 5-minute window)

Header: X-Request-Nonce
Value: Random 16-byte hex string (checked for uniqueness)
```

### 4.2 Certificate Pinning

**Implementation:**
```
// In app configuration, pin FastMODA server certificate:

pinned_certificates = [
  "sha256/AAAAAAAAAAAAAAAAAAAAAA==",  // Leaf cert
  "sha256/BBBBBBBBBBBBBBBBBBBBBB==",  // Intermediate CA
  "sha256/CCCCCCCCCCCCCCCCCCCCCC=="   // Root CA
]

For self-hosted server:
  - User provides server's certificate during setup
  - Pin certificate in app
  - Accept only requests from pinned certificate
  - Reject if certificate changes (alert user)
  - Provide certificate update mechanism
```

**TLS Configuration:**
```
tls_version: 1.3 (minimum)
cipher_suites:
  - TLS_AES_256_GCM_SHA384
  - TLS_CHACHA20_POLY1305_SHA256
  - TLS_AES_128_GCM_SHA256

certificate_validation:
  - Chain validation: Full path to root
  - Hostname verification: Exact match
  - Revocation check: OCSP stapling
  - Pinning: Compare against stored fingerprints
  - Certificate expiry: Reject if expired
```

### 4.3 Request Signing & Verification

**Message Authentication:**
```
Client sends signed request:
{
  "method": "POST",
  "path": "/api/v1/signals/upload",
  "timestamp": 1234567890,
  "nonce": "rnd16hexchars",
  "device_guid": "uuid-1234",
  "body": {
    "session_id": "session-123",
    "signal_data": "base64_encoded_blob"
  }
}

Signature = HMAC-SHA256(
  key = HKDF(
    ikm = device_private_key,
    salt = device_guid,
    info = "request_signing_key",
    length = 32
  ),
  message = method + path + timestamp + nonce + json(body)
)

Header: X-Request-Signature: base64(Signature)
```

**Server Validation:**
```
For each request:
1. Verify JWT token signature
2. Extract device_guid from token
3. Lookup device's public certificate
4. Verify X-Request-Signature with device's public key
5. Verify X-Device-Fingeprint matches stored value
6. Check timestamp is within ±5 minutes
7. Check nonce not seen before (prevent replay)
8. Verify endpoint matches requested resource
9. If all pass: Process request
   If any fail: Return 401/403 Unauthorized
```

### 4.4 Server Endpoint Protection

**Endpoint Security Model:**

```
┌────────────────────────────────────────────────┐
│ Public Endpoints (No Authentication)           │
├────────────────────────────────────────────────┤
│ GET  /api/v1/health                           │
│ POST /api/v1/auth/register                    │
│ POST /api/v1/auth/token                       │
│ POST /api/v1/auth/oauth/callback              │
│ GET  /api/v1/oauth/authorize                  │
└────────────────────────────────────────────────┘

┌────────────────────────────────────────────────┐
│ Authenticated Endpoints (JWT + Signature)     │
├────────────────────────────────────────────────┤
│ POST   /api/v1/signals/upload                │
│ GET    /api/v1/signals/<session_id>          │
│ POST   /api/v1/analysis/submit               │
│ GET    /api/v1/analysis/results              │
│ POST   /api/v1/sync/status                   │
│ DELETE /api/v1/data/purge                    │
└────────────────────────────────────────────────┘

┌────────────────────────────────────────────────┐
│ Admin Endpoints (Special Permissions)         │
├────────────────────────────────────────────────┤
│ POST   /api/v1/admin/devices/revoke          │
│ GET    /api/v1/admin/audit/logs              │
│ POST   /api/v1/admin/certificates/update     │
│ DELETE /api/v1/admin/data/gdpr-delete        │
└────────────────────────────────────────────────┘
```

**Rate Limiting per Device:**
```
Endpoint-specific limits:

/api/v1/signals/upload:
  - 1000 requests per hour per device
  - Burst: 50 requests per minute
  - Penalty: Exponential backoff on excess

/api/v1/analysis/submit:
  - 100 requests per hour per device
  - Burst: 10 requests per minute

/api/v1/auth/token (refresh):
  - 100 requests per hour per device
  - Attempt to refresh too often = suspicious

/api/v1/* (general):
  - 2000 requests per hour per device
  - Suspension for 1 hour if exceeded
```

### 4.5 Response Security

**Server Responses:**
```
All responses include:

Header: Content-Type: application/json
Header: X-Content-Type-Options: nosniff
Header: X-Frame-Options: DENY
Header: X-XSS-Protection: 1; mode=block
Header: Strict-Transport-Security: max-age=31536000

Body (encrypted for sensitive data):
{
  "status": "success",
  "timestamp": 1234567890,
  "request_id": "<UUID>",
  "data": {
    // If sensitive: AES-256-GCM encrypted
    "encrypted": true,
    "ciphertext": "base64_encrypted_data",
    "nonce": "base64_iv",
    "tag": "base64_auth_tag"
  },
  "meta": {
    "api_version": "1.0",
    "server_time": 1234567890
  }
}

Client decryption:
  - Use APP's Master Encryption Key
  - Extract nonce, ciphertext, tag
  - Decrypt with AES-256-GCM
  - Verify authentication tag
  - Use decrypted data
```

---

## 5. Bluetooth Device Handling

### 5.1 Device Pairing & Trust Flow

**User Experience:**
```
User taps "Add Device"
        │
        ▼
┌──────────────────────────────┐
│ Scan for BLE devices         │
│ Show: Name, Signal Strength  │
└──────────────────────────────┘
        │
        ▼
User selects device
        │
        ▼
┌──────────────────────────────────────┐
│ Show Device Fingerprint              │
│ "Verify this matches your device:"   │
│ ABC1 2345 6789 DEF0                  │
│ [Trust] [Cancel]                     │
└──────────────────────────────────────┘
        │
        ▼
User confirms
        │
        ▼
┌────────────────────────────────┐
│ BLE Pairing Handshake         │
│ - Establish encrypted link    │
│ - Exchange pairing keys       │
└────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────┐
│ Store Device Locally               │
│ - Device UUID (BLE identifier)     │
│ - Device MAC (physical address)    │
│ - Device Fingerprint (hash)        │
│ - Encrypted Pairing Key            │
│ - Trust Level: HIGH                │
└────────────────────────────────────┘
        │
        ▼
┌─────────────────────┐
│ Ready for streaming │
└─────────────────────┘
```

### 5.2 Device Verification on Re-connection

**Each Connection Attempt:**
```
App detects BLE device
        │
        ▼
┌──────────────────────────────┐
│ Check Whitelist              │
│ - MAC address match?         │
│ - UUID match?               │
│ - Fingerprint match?        │
└──────────────────────────────┘
        │
    ┌───┴──────────┬──────────┐
  Match         No Match    Mismatch
    │              │           │
    ▼              ▼           ▼
Connect      Require      ⚠ Alert User
             Re-pair      "Device
                          fingerprint
                          changed!"
                          [Trust] [Block]
```

### 5.3 Session Isolation

**Per-Session Device Binding:**
```sql
CREATE TABLE sessions (
  id INTEGER PRIMARY KEY,
  session_uuid TEXT UNIQUE NOT NULL,
  ble_device_id TEXT NOT NULL,  -- Must match throughout session
  ble_device_mac TEXT NOT NULL,
  ble_device_fingerprint TEXT NOT NULL,
  started_at INTEGER NOT NULL,
  ended_at INTEGER,
  signal_count INTEGER DEFAULT 0,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (ble_device_id) REFERENCES ble_devices(device_uuid)
);

-- Constraint: Device cannot change during session
-- If device disconnects: Optional reconnect with same device only
-- If different device connects: Reject connection, ask user to start new session
```

### 5.4 Multi-Device Support

**Multiple Devices Allowed, Serial Connection:**
```
┌──────────────────────────────────────┐
│ Device 1 Connected                   │
│ Streaming: Active                    │
│ Session: Recording                   │
└──────────────────────────────────────┘

┌──────────────────────────────────────┐
│ Try to connect Device 2               │
├──────────────────────────────────────┤
│ ✗ Cannot connect                     │
│ Device 1 is currently connected.     │
│                                      │
│ Options:                             │
│ [Disconnect Device 1 & Connect Dev2] │
│ [Keep Device 1 Connected]            │
│ [Cancel]                             │
└──────────────────────────────────────┘
```

---

## 6. Secure Communication with Self-Hosted Server

### 6.1 mDNS Discovery & Service Binding

**Server Discovery Flow:**
```
User enables "Connect to Local Server"
        │
        ▼
┌────────────────────────────────┐
│ Query mDNS for:                │
│ _fastmoda._tcp.local           │
│ (Raspberry Pi advertising)     │
└────────────────────────────────┘
        │
    ┌───┴──────────┬────────────┐
  Found        Not Found      Multiple
   │              │              │
   ▼              ▼              ▼
Return         Show       Select one
Server      "Server not   from list
IP:Port     found"         │
                           ▼
                      Return chosen
                      Server IP:Port
    │
    ▼
┌──────────────────────────────────┐
│ Retrieve Server Certificate      │
│ POST <server>/api/certificate    │
│ (unencrypted, TLS only)         │
└──────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────┐
│ Verify Certificate               │
│ - Check validity dates           │
│ - Verify hostname matches        │
│ - Ask user: Trust this cert?     │
│ [Trust Device] [Cancel]          │
└──────────────────────────────────┘
    │
    ▼
┌──────────────────────────────┐
│ Pin Certificate              │
│ Store fingerprint locally    │
│ Reject future certs with     │
│ different fingerprint        │
└──────────────────────────────┘
    │
    ▼
┌─────────────────────────────═──┐
│ Server marked as "Local"        │
│ Future requests go to:          │
│ https://<server>:5000/api/...  │
└────────────────────────────────┘
```

### 6.2 Self-Hosted Server API Authentication

**Different authentication for self-hosted vs cloud:**

```
CLOUD SERVER (if applicable in future):
- Standard JWT + device signature
- Requires user account
- HTTPS with public CA certificates

SELF-HOSTED SERVER (on home WiFi):
- Local-only authentication
- Device-to-server binding
- Self-signed certificate (pinned)
- Simpler credentials (optional pre-shared key)
```

**Self-Hosted Connection Sequence:**
```
Device → Self-Hosted Server
        │
        ▼
┌─────────────────────────────────────┐
│ POST /api/v1/auth/local-connect     │
│ {                                   │
│   "device_guid": "<UUID>",         │
│   "device_name": "iPhone 14 Pro",   │
│   "timestamp": 1234567890,          │
│   "signature": "<HMAC-SHA256>"      │
│ }                                   │
│                                     │
│ (No TLS client cert needed)        │
│ (Server is trusted via pinned cert)│
└─────────────────────────────────────┘
        │
        ▼
Server validates:
  - Device GUID is known
  - Signature is valid
  - Timestamp within 5 minutes
        │
        ▼
┌─────────────────────────────────┐
│ Server responds:                │
│ {                               │
│   "session_token": "<JWT>",    │
│   "expires_in": 3600,          │
│   "capabilities": [            │
│     "full_modwt",              │
│     "coherence",               │
│     "batch_processing"         │
│   ]                            │
│ }                              │
└─────────────────────────────────┘
```

### 6.3 Encrypted Data Transfer to Server

**Upload Flow (On-Device → Server):**
```
┌─────────────────────────────────────────┐
│ Signal Ready for Analysis/Upload        │
│ - Buffered locally                      │
│ - Analyzed on-device                    │
│ - User selects "Upload to Server"       │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│ Serialization                           │
│ - Convert signal to MAT/HDF5 format    │
│ - Include metadata (sample rate, etc)  │
│ - Compress with zstd                   │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│ Encryption (Additional Layer)           │
│ - Generate ephemeral AES-256 key       │
│ - Encrypt compressed data               │
│ - Generate HMAC for integrity check    │
│ - Encrypt AES key with server's public │
│   key (RSA-2048)                       │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│ Payload Structure                       │
│ {                                       │
│   "device_guid": "<UUID>",             │
│   "session_id": "<session-123>",       │
│   "timestamp": 1234567890,              │
│   "encrypted_aes_key": "base64(...)",  │
│   "encrypted_data": "base64(...)",     │
│   "hmac": "hex(...)",                  │
│   "compression": "zstd"                │
│ }                                       │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│ HTTPS POST /api/v1/signals/upload      │
│ - TLS 1.3 encryption (data in transit) │
│ - Device signature in headers          │
│ - Request body is JSON (encrypted)     │
│ - Certificate pinning active           │
└─────────────────────────────────────────┘
        │
        ▼
Server processes:
  1. Validate JWT token
  2. Verify request signature
  3. Decrypt AES key with server's private key
  4. Decrypt data with AES key
  5. Verify HMAC
  6. Decompress
  7. Store securely
  8. Respond with success
```

**Download Flow (Server → Device):**
```
Device requests: GET /api/v1/analysis/<analysis_id>
  │
  ▼
Server retrieves analysis results
  │
  ▼
Server encrypts response (same as above):
  - AES-256 encrypt
  - RSA encrypt AES key with device's public key
  - HTTPS TLS encryption (double layer)
  │
  ▼
Device receives encrypted response
  │
  ▼
Device decrypts:
  1. Use TLS decryption (transport layer)
  2. Extract encrypted AES key (encrypted with device's private key)
  3. Use device's private key to decrypt AES key
  4. Use AES key to decrypt data
  5. Verify HMAC
  6. Use data
```

### 6.4 Offline Queue & Sync

**When Server Unavailable:**
```
User submits signal for deep analysis
              │
              ▼
  ┌─────────────────────┐
  │ Check server status │
  │ (mDNS query)       │
  └─────────────────────┘
         │
    ┌────┴─────┐
    │           │
Available   Unavailable
    │           │
    ▼           ▼
  Upload    Queue for
  Now       Later
            │
            ▼
  ┌──────────────────────────┐
  │ Store in SQLite:         │
  │ - session_id             │
  │ - signal_data (encrypted)│
  │ - synced = 0             │
  │ - queued_at = timestamp  │
  └──────────────────────────┘
            │
   (background sync on WiFi)
            │
            ▼
  ┌──────────────────────────┐
  │ Periodic Check (30min)   │
  │ - Server available?      │
  │ - User connected to WiFi?│
  └──────────────────────────┘
            │
        Available
            │
            ▼
  ┌──────────────────────────┐
  │ Upload Queued Data       │
  │ - Encrypt with session   │
  │   key                    │
  │ - POST to server         │
  │ - Mark synced_at         │
  └──────────────────────────┘
```

---

## 7. Compliance & Best Practices

### 7.1 GDPR & Data Privacy

**User Data Protection:**
- All sensitive data encrypted at rest
- Clear consent for data collection
- User can delete all local data: Settings → Data → "Delete All Data"
- GDPR right to be forgotten: DELETE /api/v1/user/data
- No tracking or analytics (unless explicitly opted-in)
- Privacy policy clearly states what data is collected

**Data Minimization:**
- Only collect signal data necessary for analysis
- No location tracking
- No user identification (UUID-based, not name)
- No behavioral analytics
- Optional: Device telemetry (battery, crash reports)

### 7.2 HIPAA (If Medical Use)

**If analyzing medical signals (ECG, EEG, etc):**
- Enable HIPAA-compliant encryption
- Audit logging of all data access
- Business Associate Agreement (BAA) with server provider
- Data retention policies (user-defined)
- Incident reporting procedures
- Access controls (biometric lock required)

### 7.3 Secure Coding Practices

**Development Standards:**
- Code review for security-critical paths
- Static analysis (SonarQube, Lint)
- Dependency scanning (OWASP Dependency-Check)
- Secrets scanning (prevent API keys in code)
- No hardcoded credentials
- Input validation on all user input
- Output encoding to prevent injection attacks

**Testing:**
- Penetration testing (quarterly)
- OWASP Top 10 vulnerability checks
- Cryptographic algorithm validation
- Key rotation testing
- Token expiry and refresh testing

### 7.4 Security Updates & Patching

**Update Strategy:**
```
Vulnerability Discovered
        │
        ▼
┌──────────────────────┐
│ Assess Severity      │
│ (CVSS Score)         │
└──────────────────────┘
        │
    ┌───┴───┬──────┐
Critical High   Low
    │      │       │
    ▼      ▼       ▼
  <24h   <1wk   <1mo

Deploy patch → Code Review → Testing → Build → Submit
```

---

## 8. Implementation Checklist

### Phase 1: Foundation (Weeks 1-2)
- [ ] Design authentication system (JWT + device binding)
- [ ] Set up Platform Keystore integration (iOS/Android)
- [ ] Implement RSA-2048 key generation and storage
- [ ] Create encrypted SQLite schema
- [ ] Design API request/response encryption

### Phase 2: Core Security (Weeks 3-4)
- [ ] Implement BLE pairing with encrypted links
- [ ] Build device fingerprinting system
- [ ] Implement certificate pinning
- [ ] Create request signing mechanism (HMAC)
- [ ] Build token refresh flow

### Phase 3: Server Integration (Weeks 5-6)
- [ ] Implement mDNS discovery for self-hosted server
- [ ] Create self-signed certificate handling
- [ ] Build encrypted data upload/download
- [ ] Test with actual Raspberry Pi server
- [ ] Implement offline queue and sync

### Phase 4: Testing & Hardening (Weeks 7-8)
- [ ] Security penetration testing
- [ ] Rate limiting & DDoS mitigation testing
- [ ] Key rotation testing
- [ ] Token revocation testing
- [ ] Certificate pinning validation
- [ ] Encryption edge case testing

### Phase 5: Documentation & Deployment (Week 9)
- [ ] Create security documentation
- [ ] Create deployment guide for self-hosted server
- [ ] Document certificate update procedure
- [ ] Create incident response procedures
- [ ] Prepare security audit logs

---

## 9. Threat Model & Mitigations

| Threat | Attack Vector | Mitigation |
|--------|---------------|-----------|
| **Token Theft** | Network interception, device compromise | Device binding, short expiry, refresh rotation, biometric lock |
| **Man-in-Middle** | Unencrypted connection, fake cert | TLS 1.3, certificate pinning, HTTPS only |
| **Unauthorized Device** | Fake Bluetooth device, stolen device | BLE pairing, device fingerprinting, whitelist |
| **Data Breach** | Unencrypted database, memory dumps | AES-256-GCM encryption, secure deletion, memory locking |
| **Endpoint Exposure** | API reverse engineering, endpoint discovery | Request signing, rate limiting, device authentication |
| **Replay Attack** | Captured requests replayed | Nonce validation, timestamp checking, token rotation |
| **Privilege Escalation** | Fake admin endpoints, JWT tampering | Request signing, server-side verification, scope validation |
| **Bluetooth Spoofing** | Device spoofing attacks | MAC address filtering, fingerprint verification, pairing confirmation |

---

## 10. Security Contact & Disclosure

**Report Security Issues to:**
- Email: security@moda-project.org
- GPG Key: (to be generated)
- Response Time: 24-48 hours for critical issues

**Responsible Disclosure Policy:**
- 90-day grace period before public disclosure
- Regular security updates and patches
- Transparency in breach notifications

---

**Document Version:** 1.0
**Last Updated:** 2026-03-05
**Author:** MODA Security Architecture Team
**Classification:** Confidential - Internal Review