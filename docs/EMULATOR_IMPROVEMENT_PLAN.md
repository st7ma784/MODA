# Emulator Architecture Improvement Plan

## Current State

**Problem:** The emulator manually calls the FastMODA backend via Python, creating a bottleneck and preventing realistic testing of the Flutter app's actual signal acquisition and analysis workflow.

**Current Flow:**
```
Browser → emulator.py (Flask)
              ├─ Generates synthetic signals
              ├─ Computes metrics locally
              └─ Forwards to FastMODA
```

## Desired Architecture

**Goal:** Load the actual Flutter app UI and let it interact directly with a containerized FastMODA instance.

**New Flow:**
```
Browser → Flutter Web App (served by emulator or separate container)
              │
              └─→ FastMODA Container (port 5000)
                   ├─ /health
                   ├─ /analyze
                   ├─ /analyze_modwt
                   ├─ /analyze_coherence
                   ├─ /analyze_bispectrum
                   └─ /status/<task_id>
```

---

## Phase 1: Containerized Test Environment

### 1.1 Docker Compose Enhancement

**File:** `docker-compose.yml`

Add/enhance services:
```yaml
services:
  fastmoda-api:
    build:
      context: ./FastMODA
      dockerfile: Dockerfile
    environment:
      - FASTMODA_API_KEY=moda_8e6695088c2e3114cbb25e3554544f2577cd53c58a3672ac
      - FLASK_ENV=development
    ports:
      - "5000:5000"
    volumes:
      - ./FastMODA:/app
    networks:
      - moda-net
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 5s
      timeout: 3s
      retries: 3

  flutter-emulator:
    build:
      context: ./APP
      dockerfile: Dockerfile.web  # NEW
      args:
        FASTMODA_URL: http://fastmoda-api:5000
    environment:
      - FASTMODA_BASE_URL=http://fastmoda-api:5000
    ports:
      - "8080:80"  # or 8000 for Flutter development server
    depends_on:
      fastmoda-api:
        condition: service_healthy
    networks:
      - moda-net

  moda-signal-mock-server:  # OPTIONAL: Replace browser-based signal generation
    image: python:3.11
    volumes:
      - ./emulator_refactored.py:/app/server.py
      - ./APP:/app/flutter
    working_dir: /app
    command: python server.py
    ports:
      - "8081:8081"
    networks:
      - moda-net
    depends_on:
      - fastmoda-api

networks:
  moda-net:
    driver: bridge

volumes:
  moda-dev-data:
  fastmoda-cache:
```

### 1.2 Rationale

- **fastmoda-api**: Standard FastMODA service, unchanged
- **flutter-emulator**: Builds Flutter web app with configurable FastMODA URL
- **moda-signal-mock-server** (optional): Provides a REST endpoint for injecting mock signals without blocking the main app
- **networks**: All containers communicate over internal Docker network

---

## Phase 2: Flutter Web Build Integration

### 2.1 Create Dockerfile for Flutter Web

**File:** `APP/Dockerfile.web`

```dockerfile
# Stage 1: Build Flutter app
FROM ghcr.io/cirruslabs/flutter:latest AS builder

ARG FASTMODA_URL=http://localhost:5000

WORKDIR /app
COPY . .

# Override default URL if passed as build arg
RUN sed -i "s|kFastModaDefaultUrl = '[^']*'|kFastModaDefaultUrl = '${FASTMODA_URL}'|" \
    lib/config/app_config.dart

RUN flutter clean && \
    flutter pub get && \
    flutter build web --release --no-tree-shake-icons

# Stage 2: Serve with nginx
FROM nginx:alpine

COPY --from=builder /app/build/web /usr/share/nginx/html

# Enable CORS for FastMODA API calls
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

### 2.2 Nginx Configuration for CORS

**File:** `APP/nginx.conf`

```nginx
server {
    listen 80;
    root /usr/share/nginx/html;
    index index.html;

    # Fallback to index.html for routing
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Proxy FastMODA API calls if needed (optional CORS workaround)
    location /api/ {
        proxy_pass http://fastmoda-api:5000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # CORS headers for direct calls
    add_header Access-Control-Allow-Origin "*" always;
    add_header Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS" always;
    add_header Access-Control-Allow-Headers "Content-Type, Authorization" always;
}
```

---

## Phase 3: Refactored Python Emulator (Optional Mock Signal Server)

### 3.1 New Purpose: Signal Injection Only

Instead of a browser-based emulator, provide a **server-side signal injection API** that:
- Does NOT generate signals in the browser
- Provides REST endpoints for injecting test data into a queue
- Allows the Flutter app to request signal data (simulating BLE)
- Integrates with the Flutter app's `SignalService`

**File:** `emulator_refactored.py`

```python
#!/usr/bin/env python3
"""
Signal Mock Server for Testing
==============================
Provides HTTP endpoints to:
1. Generate synthetic EEG signals (resting, active, drowsy, sleep, noise)
2. Inject them into a queue accessible to the Flutter app
3. Simulate BLE data streaming without hardware

Usage:
    python emulator_refactored.py --port 8081 --fastmoda http://fastmoda-api:5000
"""

import argparse
import json
import time
import threading
from collections import deque
from flask import Flask, jsonify, request
import numpy as np

class SignalGenerator:
    """Generates synthetic EEG signals in different brain states."""
    
    PRESETS = {
        "resting": dict(alpha=1.0, theta=0.3, beta=0.12, delta=0.10, gamma=0.05, noise=0.20),
        "active":  dict(alpha=0.3, theta=0.2, beta=0.80, delta=0.05, gamma=0.20, noise=0.30),
        "drowsy":  dict(alpha=0.5, theta=0.9, beta=0.05, delta=0.30, gamma=0.02, noise=0.15),
        "sleep":   dict(alpha=0.1, theta=0.3, beta=0.04, delta=1.20, gamma=0.02, noise=0.10),
        "noise":   dict(alpha=0.1, theta=0.1, beta=0.10, delta=0.10, gamma=0.10, noise=1.50),
    }
    
    def __init__(self, fs=256.0):
        self.fs = fs
        self.preset = "resting"
        self.apply_preset(self.preset)
        self.queue = deque(maxlen=1024)
        self.lock = threading.Lock()
        self.streaming = False
    
    def apply_preset(self, name):
        p = self.PRESETS.get(name, self.PRESETS["resting"])
        for k, v in p.items():
            setattr(self, k, v)
    
    def generate(self, duration_sec, start_time=0.0):
        """Generate synthetic signal samples."""
        n_samples = int(duration_sec * self.fs)
        samples = []
        for i in range(n_samples):
            t = start_time + i / self.fs
            s = (
                self.alpha * np.sin(2 * np.pi * 10.0 * t)
                + self.theta * np.sin(2 * np.pi * 6.0 * t)
                + self.beta * np.sin(2 * np.pi * 18.0 * t)
                + self.delta * np.sin(2 * np.pi * 2.0 * t)
                + self.gamma * np.sin(2 * np.pi * 40.0 * t)
                + self.noise * np.random.randn()
            )
            samples.append(float(s))
        return samples

gen = SignalGenerator()

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "signal-mock-server"})

@app.route("/stream/start", methods=["POST"])
def start_stream():
    """Begin streaming signals."""
    gen.streaming = True
    gen.preset = request.json.get("preset", "resting") if request.json else "resting"
    gen.apply_preset(gen.preset)
    return jsonify({"ok": True, "streaming": True})

@app.route("/stream/stop", methods=["POST"])
def stop_stream():
    """Stop streaming signals."""
    gen.streaming = False
    return jsonify({"ok": True, "streaming": False})

@app.route("/stream/chunk", methods=["GET"])
def get_chunk():
    """
    Fetch next signal chunk (simulates BLE streaming).
    
    Query params:
    - duration: seconds of data to return (default 0.1)
    - preset: brain state (default from /stream/start)
    """
    if not gen.streaming:
        return jsonify({"error": "Not streaming"}), 400
    
    duration = float(request.args.get("duration", 0.1))
    chunk = gen.generate(duration)
    
    return jsonify({
        "samples": chunk,
        "sample_rate": gen.fs,
        "preset": gen.preset,
        "count": len(chunk),
    })

@app.route("/preset", methods=["POST"])
def set_preset():
    """Change the signal preset (brain state)."""
    preset = request.json.get("preset", "resting")
    gen.apply_preset(preset)
    gen.preset = preset
    return jsonify({"preset": preset, "ok": True})

@app.route("/settings", methods=["GET", "POST"])
def settings():
    """Get/set server settings."""
    if request.method == "POST":
        if "sample_rate" in request.json:
            gen.fs = float(request.json["sample_rate"])
    return jsonify({
        "sample_rate": gen.fs,
        "streaming": gen.streaming,
        "preset": gen.preset,
    })

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Signal Mock Server")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--fastmoda", default="http://localhost:5000")
    args = parser.parse_args()
    
    app.run(host="0.0.0.0", port=args.port, debug=True)
```

---

## Phase 4: Flutter App Configuration Updates

### 4.1 Environment-Based URL Configuration

**File:** `APP/lib/config/app_config.dart`

Update to support environment-based defaults:

```dart
// Default URL — override at build time with --dart-define
const String kFastModaDefaultUrl = String.fromEnvironment(
  'FASTMODA_URL',
  defaultValue: 'http://localhost:5000',  // Fallback for local dev
);

// ... rest of config
```

### 4.2 Build Command for Web

Add to `APP/README.md` or CI/CD:

```bash
# Build with custom FastMODA URL
flutter build web \
  --dart-define=FASTMODA_URL=http://localhost:5000 \
  --release

# Run dev server with custom URL
flutter run -d web-server \
  --dart-define=FASTMODA_URL=http://localhost:5000 \
  --web-port 8000
```

---

## Phase 5: Integration Testing

### 5.1 Test Harness

**File:** `tests/emulator_integration_test.sh`

```bash
#!/bin/bash
set -e

echo "🐳 Starting containerized emulator environment..."

# Start all services
docker-compose up -d

# Wait for FastMODA to be ready
echo "⏳ Waiting for FastMODA health check..."
timeout 30 bash -c 'until curl -f http://localhost:5000/health; do sleep 1; done'

# Wait for Flutter web app
echo "⏳ Waiting for Flutter web app..."
timeout 30 bash -c 'until curl -f http://localhost:8080; do sleep 1; done'

echo "✅ Emulator environment ready!"
echo "📱 Flutter App: http://localhost:8080"
echo "🔬 FastMODA API: http://localhost:5000"
echo "🎯 Mock Signal Server: http://localhost:8081"

# Optionally run smoke tests
if [ "$1" = "test" ]; then
    echo "🧪 Running integration tests..."
    python3 tests/emulator_smoke_tests.py
fi
```

### 5.2 Smoke Tests

**File:** `tests/emulator_smoke_tests.py`

```python
#!/usr/bin/env python3
import requests
import time

def test_fastmoda_health():
    """Verify FastMODA API is running."""
    resp = requests.get("http://localhost:5000/health")
    assert resp.status_code == 200
    print("✅ FastMODA health check passed")

def test_flutter_app_loads():
    """Verify Flutter web app serves."""
    resp = requests.get("http://localhost:8080")
    assert resp.status_code == 200
    assert "MODA" in resp.text or "flutter" in resp.text.lower()
    print("✅ Flutter app loads")

def test_signal_mock_server():
    """Verify mock signal server is available."""
    resp = requests.get("http://localhost:8081/health")
    assert resp.status_code == 200
    print("✅ Mock signal server health check passed")

def test_end_to_end_analysis():
    """
    E2E test: Submit signal to FastMODA and verify response.
    """
    import numpy as np
    
    # Generate test signal
    fs = 256.0
    t = np.arange(0, 2, 1/fs)
    signal = np.sin(2 * np.pi * 10 * t)
    signal_bytes = signal.astype(np.float32).tobytes()
    
    # Submit to FastMODA
    files = {"file": ("signal.npy", signal_bytes)}
    data = {"fs": "256.0", "win": "1.0", "pen": "auto"}
    
    resp = requests.post(
        "http://localhost:5000/analyze",
        files=files,
        data=data,
    )
    
    assert resp.status_code == 200
    result = resp.json()
    assert "task_id" in result
    task_id = result["task_id"]
    
    # Poll for completion
    for _ in range(60):  # 30 seconds max
        status_resp = requests.get(f"http://localhost:5000/status/{task_id}")
        status = status_resp.json()
        
        if status.get("status") == "complete":
            print(f"✅ E2E analysis completed: {task_id}")
            return
        
        time.sleep(0.5)
    
    raise TimeoutError(f"Analysis {task_id} did not complete in 30s")

if __name__ == "__main__":
    test_fastmoda_health()
    test_flutter_app_loads()
    test_signal_mock_server()
    test_end_to_end_analysis()
    print("\n✅ All integration tests passed!")
```

---

## Implementation Roadmap

### Week 1: Foundation
- [ ] Enhance `docker-compose.yml` with FastMODA + Flutter web services
- [ ] Create `Dockerfile.web` and `nginx.conf` for Flutter app
- [ ] Update `app_config.dart` with environment-aware defaults
- [ ] Test docker-compose orchestration

### Week 2: Mock Server & Integration
- [ ] Refactor emulator as standalone signal injection service
- [ ] Implement REST endpoints for signal control
- [ ] Create integration test harness
- [ ] Document new emulator flow

### Week 3: E2E Testing & Documentation
- [ ] Run smoke tests in Docker environment
- [ ] Build CI/CD pipeline for automated testing
- [ ] Update README with new emulator startup instructions
- [ ] Create troubleshooting guide

---

## Fallback & Compatibility

**Old emulator.py:**
- Can be preserved as `emulator_legacy.py` for reference
- Useful for isolated browser-only testing if needed

**Migration path:**
- Existing emulator.py works for current workflows
- New docker-compose setup runs in parallel
- Gradual migration as team gets comfortable

---

## Benefits of New Architecture

✅ **Realistic testing**: App talks directly to FastMODA (actual production path)  
✅ **No bottleneck**: No Python middleware forwarding signals  
✅ **Containerized**: One `docker-compose up` to run entire stack  
✅ **Scalable**: Easy to add more test scenarios  
✅ **CI/CD ready**: Automated integration testing  
✅ **Cross-platform**: Works on Linux, macOS, Windows (with Docker)  

---

## Questions & Decisions

1. **BLE Simulation**: Should Flutter app's BLE service talk to mock-server, or should we mock it in code?
   - *Recommendation*: Mock in code initially; add server-based BLE simulation if full integration needed

2. **Authentication**: FastMODA API key handling in containerized environment?
   - *Recommendation*: Pass via `FASTMODA_API_KEY` environment variable to both app (at build) and API (at runtime)

3. **Persistent data**: Where should test signals / analysis history be stored?
   - *Recommendation*: SQLite for app (per spec), temporary container volume for server

4. **Deployment**: Should this become the primary dev environment?
   - *Recommendation*: Yes, but keep option to run Flutter app locally for rapid iteration
