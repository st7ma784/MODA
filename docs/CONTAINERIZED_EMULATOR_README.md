# MODA Containerized Emulator

## Overview

The containerized emulator provides a complete, production-like testing environment for the MODA app without needing physical hardware. It brings together:

- **Flutter Web App** — The actual UI served in a browser
- **FastMODA API** — Signal analysis backend (unchanged)
- **Signal Mock Server** — Synthetic EEG signal generation and control

### Key Improvements Over Legacy Emulator

| Aspect | Legacy (`emulator.py`) | New (Containerized) |
|--------|---|---|
| **Architecture** | Browser + Python middleware | Flutter app → FastMODA (direct) |
| **Signal path** | Browser → emulator → FastMODA | Browser → FastMODA |
| **Setup** | Manual Python installation | Single `docker-compose up` |
| **Testing** | Manual browser testing | Automated smoke tests |
| **Deployment** | Development only | Production-like environment |

---

## Quick Start

### 1. Start the Environment

```bash
cd /home/user/MODA
docker-compose up -d fastmoda-api flutter-emulator moda-signal-mock-server
```

Wait for all services to be healthy (~30 seconds):

```bash
# Monitor startup
docker-compose logs -f
```

### 2. Access the Services

Open these in your browser:

- **Flutter MODA App**: http://localhost:8080
- **FastMODA API**: http://localhost:5000
- **Mock Signal Server** (debugging): http://localhost:8081

### 3. Run Integration Tests

```bash
./tests/emulator_integration_test.sh test
```

Or use the helper script:

```bash
# Start environment and run tests
./tests/emulator_integration_test.sh test

# Just start
./tests/emulator_integration_test.sh up

# Restart
./tests/emulator_integration_test.sh restart

# Tear down
./tests/emulator_integration_test.sh down
```

---

## Services Architecture

### FastMODA API (`fastmoda-api`)

**Image**: Built from `FastMODA/Dockerfile`  
**Port**: 5000  
**Purpose**: Signal analysis backend (unchanged from production)

**Environment**:
```
FLASK_APP=app.py
FLASK_ENV=development
```

**Health check**: `GET http://localhost:5000/health`

**Key endpoints**:
- `POST /analyze` — Submit signal for FFT analysis
- `POST /analyze_modwt` — Wavelet decomposition
- `POST /analyze_coherence` — Multi-channel coherence
- `POST /analyze_bispectrum` — Bispectral analysis
- `GET /status/<task_id>` — Poll analysis results

### Flutter Emulator (`flutter-emulator`)

**Image**: Built from `APP/Dockerfile.web`  
**Port**: 8080  
**Purpose**: Serves the actual Flutter web app UI

**Build arguments**:
```dockerfile
FASTMODA_URL=http://fastmoda-api:5000
```

This URL is baked into the app at build time (configurable at startup).

**Health check**: `GET http://localhost:8080/`

**Features**:
- Single-page app routing (all routes → `index.html`)
- Static asset caching (hashed filenames)
- CORS headers for API calls
- Gzip compression

### Signal Mock Server (`moda-signal-mock-server`)

**Image**: Built from `Dockerfile.signal-mock`  
**Port**: 8081  
**Purpose**: Generate synthetic EEG signals without hardware

**Environment**:
```
FLASK_APP=emulator_refactored.py
FLASK_ENV=development
FASTMODA_URL=http://fastmoda-api:5000
```

**Health check**: `GET http://localhost:8081/health`

**Key endpoints**:
- `POST /stream/start` — Start signal generation
- `POST /stream/stop` — Stop signal generation
- `GET /stream/chunk` — Fetch signal samples
- `POST /preset` — Switch brain state (resting, active, drowsy, sleep, noise)
- `GET /settings` — View/modify server settings

---

## Signal Generation

### Brain State Presets

The mock signal server can generate synthetic EEG in different states:

| Preset | Use Case | Characteristics |
|--------|----------|---|
| **resting** | Default, relaxed state | Strong alpha (10 Hz), moderate noise |
| **active** | Alert, focused | High beta (18 Hz), low alpha |
| **drowsy** | Sleepy, declining consciousness | Strong theta (6 Hz) + alpha |
| **sleep** | Deep sleep | Dominant delta (2 Hz) waves |
| **noise** | Testing robustness | High noise, low coherent signal |

### API Examples

#### Start streaming (resting state)
```bash
curl -X POST http://localhost:8081/stream/start \
  -H "Content-Type: application/json" \
  -d '{"preset": "resting"}'
```

#### Get signal chunk
```bash
curl "http://localhost:8081/stream/chunk?duration=0.5"
```

Response:
```json
{
  "samples": [0.123, -0.456, ...],
  "sample_rate": 256.0,
  "preset": "resting",
  "count": 128,
  "total_samples": 5120
}
```

#### Switch preset
```bash
curl -X POST http://localhost:8081/preset \
  -H "Content-Type: application/json" \
  -d '{"preset": "active"}'
```

#### Stop streaming
```bash
curl -X POST http://localhost:8081/stream/stop
```

---

## Testing

### Smoke Tests

Verify all services are healthy and functional:

```bash
python3 tests/emulator_smoke_tests.py
```

Tests check:
1. ✅ FastMODA API health
2. ✅ Flutter app loads
3. ✅ Mock signal server responds
4. ✅ Signal generation works
5. ✅ Preset switching works
6. ✅ End-to-end analysis completes

### Integration Test Harness

The shell script orchestrates startup, health checks, and test execution:

```bash
./tests/emulator_integration_test.sh [command]

Commands:
  up       — Start environment
  down     — Tear down environment
  test     — Start + run smoke tests
  restart  — Restart all services
```

### Custom Testing

You can write custom tests against the APIs:

**Python example**:
```python
import requests
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

task_id = resp.json()["task_id"]

# Poll for results
while True:
    status = requests.get(f"http://localhost:5000/status/{task_id}").json()
    if status["status"] == "complete":
        print(f"✅ Analysis done!")
        print(status["results"])
        break
    time.sleep(0.5)
```

---

## Configuration

### FastMODA URL (in Flutter app)

The app can point to different FastMODA instances:

**At build time** (Docker build):
```bash
docker build --build-arg FASTMODA_URL=http://custom.host:5000 -t moda-web APP/
```

**At runtime** (Settings screen in app):
Users can change the FastMODA URL via the Settings tab in the app.

**Environment variable** (for development):
```bash
flutter run -d web-server \
  --dart-define=FASTMODA_URL=http://localhost:5000
```

### Sample Rate

The mock signal server defaults to 256 Hz sampling rate. Change it:

```bash
curl -X POST http://localhost:8081/settings \
  -H "Content-Type: application/json" \
  -d '{"sample_rate": 512.0}'
```

---

## Docker Compose Usage

### Start all MODA services (including legacy MATLAB dev)

```bash
docker-compose up -d
```

### Start only emulator stack (recommended for testing)

```bash
docker-compose up -d fastmoda-api flutter-emulator moda-signal-mock-server
```

### View logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f flutter-emulator

# Last N lines
docker-compose logs --tail=100 fastmoda-api
```

### Stop services

```bash
docker-compose down
```

### Rebuild images (after code changes)

```bash
docker-compose build --no-cache flutter-emulator
docker-compose up -d flutter-emulator
```

---

## Troubleshooting

### Services won't start

**Check logs**:
```bash
docker-compose logs fastmoda-api
docker-compose logs flutter-emulator
docker-compose logs moda-signal-mock-server
```

**Common issues**:
- Port already in use: `lsof -i :8080` and kill the process
- Build failed: Try `docker-compose build --no-cache`
- Out of disk space: `docker system prune -a`

### Flutter app can't reach FastMODA

**Check network connectivity**:
```bash
docker-compose exec flutter-emulator curl http://fastmoda-api:5000/health
```

**If inside Docker** (from app logs):
- Use service name: `http://fastmoda-api:5000`
- NOT `http://localhost:5000`

**From browser** (outside Docker):
- Use `http://localhost:5000`

### Signal mock server returns 500 errors

**Check for bugs**:
```bash
docker-compose logs moda-signal-mock-server
```

**Restart the service**:
```bash
docker-compose restart moda-signal-mock-server
```

### Analysis hangs indefinitely

**Check FastMODA health**:
```bash
curl http://localhost:5000/health
docker-compose logs fastmoda-api
```

**Increase timeout** in app Settings or code:
```dart
const Duration kAnalysisReceiveTimeout = Duration(seconds: 180);
```

---

## Development Workflow

### Making Changes to Flutter App

```bash
# 1. Edit source code
vim APP/lib/screens/home.dart

# 2. Rebuild web app
docker-compose build --no-cache flutter-emulator

# 3. Restart container
docker-compose restart flutter-emulator

# 4. Refresh browser: http://localhost:8080
```

### Making Changes to FastMODA

```bash
# 1. Edit source code
vim FastMODA/app.py

# 2. Rebuild and restart (Flask auto-reloads in dev mode)
docker-compose restart fastmoda-api

# 3. Test via API or app
```

### Making Changes to Signal Mock Server

```bash
# 1. Edit source code
vim emulator_refactored.py

# 2. Restart (Flask auto-reloads)
docker-compose restart moda-signal-mock-server

# 3. Test via /health or API calls
```

---

## Migration from Legacy Emulator

The original `emulator.py` is still available as a fallback:

```bash
# Old way (if needed)
python emulator.py --port 8000

# New way (recommended)
./tests/emulator_integration_test.sh test
```

**Advantages of new approach**:
- ✅ Tests the actual app UI
- ✅ No Python dependencies in host environment
- ✅ Closer to production deployment
- ✅ Automated health checks
- ✅ Easier to extend with new services

---

## Advanced: Adding New Services

To add another container (e.g., a BLE simulator):

1. **Create Dockerfile**:
   ```dockerfile
   # services/Dockerfile.ble-sim
   FROM python:3.11-slim
   COPY ble_simulator.py /app/
   CMD ["python", "/app/ble_simulator.py"]
   ```

2. **Add to docker-compose.yml**:
   ```yaml
   ble-simulator:
     build:
       context: ./services
       dockerfile: Dockerfile.ble-sim
     ports:
       - "9000:9000"
     networks:
       - moda-net
   ```

3. **Rebuild and test**:
   ```bash
   docker-compose up -d ble-simulator
   docker-compose logs ble-simulator
   ```

---

## Performance Notes

### Resource Limits

Each service has CPU/memory limits in `docker-compose.yml`:

- **fastmoda-api**: 2 CPU, 2 GB RAM
- **flutter-emulator**: 1 CPU, 512 MB RAM
- **moda-signal-mock-server**: 1 CPU, 512 MB RAM

Adjust for your system:

```yaml
deploy:
  resources:
    limits:
      cpus: '4'
      memory: 4G
```

### Network Performance

Services communicate over internal Docker bridge network (`moda-net`), which is fast and isolated from host network. No performance penalty for inter-service calls.

---

## References

- [Flutter Web Deployment](https://docs.flutter.dev/deployment/web)
- [FastMODA API Spec](../FastMODA/API.md)
- [Docker Compose Reference](https://docs.docker.com/compose/)
- [Nginx Configuration](./docs/fastmoda/DOCKER_GUIDE.md)

---

## Support

For issues or questions:

1. **Check logs**: `docker-compose logs -f [service]`
2. **Run smoke tests**: `python3 tests/emulator_smoke_tests.py`
3. **View plan**: `docs/EMULATOR_IMPROVEMENT_PLAN.md`
4. **Manual API test**: Use `curl` examples above

---

**Last updated**: May 2, 2026
