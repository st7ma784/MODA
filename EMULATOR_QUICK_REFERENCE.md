# MODA Emulator — Quick Reference

## One-Liner Start

```bash
cd /home/user/MODA && docker-compose build fastmoda-api flutter-emulator moda-signal-mock-server && docker-compose up -d fastmoda-api flutter-emulator moda-signal-mock-server && sleep 30 && python3 tests/emulator_smoke_tests.py
```

## Key Commands

### Build & Start
```bash
# Build all three services
docker-compose build fastmoda-api flutter-emulator moda-signal-mock-server

# Start all three
docker-compose up -d fastmoda-api flutter-emulator moda-signal-mock-server

# Start with logs
docker-compose up fastmoda-api flutter-emulator moda-signal-mock-server

# Stop all
docker-compose down
```

### Verify Services
```bash
# Automated tests
python3 tests/emulator_smoke_tests.py

# Or manual checks (3 commands)
curl http://localhost:5000/health      # FastMODA
curl http://localhost:8080/            # Flutter app
curl http://localhost:8081/health      # Mock server
```

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f flutter-emulator
docker-compose logs -f fastmoda-api
docker-compose logs -f moda-signal-mock-server

# Last 50 lines
docker-compose logs --tail=50 fastmoda-api
```

### Access Services
```bash
# Open in browser
http://localhost:8080        # ← Flutter app (main)
http://localhost:5000        # ← FastMODA API (debug)
http://localhost:8081/health # ← Mock server (health)
```

## Signal Control (API)

### Start Streaming
```bash
curl -X POST http://localhost:8081/stream/start \
  -H "Content-Type: application/json" \
  -d '{"preset": "resting"}'
```

### Get Signal Chunk
```bash
curl "http://localhost:8081/stream/chunk?duration=0.5"
```

### Switch Preset
```bash
curl -X POST http://localhost:8081/preset \
  -H "Content-Type: application/json" \
  -d '{"preset": "active"}'  # resting|active|drowsy|sleep|noise
```

### Stop Streaming
```bash
curl -X POST http://localhost:8081/stream/stop
```

## Troubleshooting

### Quick Diagnostic
```bash
# Show all running containers
docker ps

# Rebuild from scratch
docker-compose build --no-cache fastmoda-api flutter-emulator moda-signal-mock-server

# Reset everything
docker-compose down -v
docker-compose up -d fastmoda-api flutter-emulator moda-signal-mock-server

# Check port conflicts
lsof -i :8080
lsof -i :5000
lsof -i :8081
```

### Common Issues

| Error | Fix |
|-------|-----|
| Port already in use | `docker-compose down` then retry |
| Service unhealthy | `docker-compose logs [service]` |
| Docker not found | Install Docker Desktop or Docker Engine |
| Slow build | Try `docker-compose build --no-cache` |
| App can't reach API | Check service is running: `docker ps` |

## File Locations

```
/home/user/MODA/
├── docker-compose.yml              ← Main configuration
├── emulator_refactored.py           ← Signal server code
├── Dockerfile.signal-mock           ← Signal server image
├── APP/
│   ├── Dockerfile.web              ← Flutter web image
│   └── nginx.conf                  ← Nginx configuration
├── tests/
│   ├── emulator_integration_test.sh ← Orchestration script
│   └── emulator_smoke_tests.py     ← Test suite
└── docs/
    ├── CONTAINERIZED_EMULATOR_README.md    ← Full guide
    ├── EMULATOR_BUILD_GUIDE.md             ← Build info
    ├── EMULATOR_IMPROVEMENT_PLAN.md        ← Architecture
    └── EMULATOR_IMPLEMENTATION_SUMMARY.md  ← This summary
```

## URLs (when running)

| Service | URL |
|---------|-----|
| Flutter App | http://localhost:8080 |
| FastMODA API | http://localhost:5000 |
| Mock Server | http://localhost:8081 |

## Brain State Presets

| Preset | Description |
|--------|---|
| `resting` | Awake, relaxed (strong alpha) |
| `active` | Alert, focused (high beta) |
| `drowsy` | Sleepy (strong theta) |
| `sleep` | Deep sleep (dominant delta) |
| `noise` | High noise (test robustness) |

## Development Tips

### Live Reload (Flutter)
```bash
# For fastest iteration, run locally
cd APP
flutter run -d web-server \
  --dart-define=FASTMODA_URL=http://localhost:5000

# Then run FastMODA only in Docker
docker-compose up -d fastmoda-api moda-signal-mock-server
```

### Rebuild After Code Changes
```bash
# Flutter changes
docker-compose build flutter-emulator
docker-compose restart flutter-emulator

# Signal server changes
docker-compose build moda-signal-mock-server
docker-compose restart moda-signal-mock-server

# FastMODA changes
docker-compose restart fastmoda-api  # Auto-reloads in dev mode
```

## Integration Testing

### Run Full Test Suite
```bash
./tests/emulator_integration_test.sh test
```

### Run Specific Tests
```bash
python3 -c "
from tests.emulator_smoke_tests import *
test_fastmoda_health()
test_flutter_app_loads()
test_signal_mock_server_health()
test_signal_generation()
test_preset_switching()
test_end_to_end_analysis()
"
```

## Environment Variables

### Build-time (Docker)
```bash
docker-compose build \
  --build-arg FASTMODA_URL=http://api.example.com \
  flutter-emulator
```

### Runtime (App Settings)
Set in Flutter app Settings tab.

### Development (Flutter CLI)
```bash
flutter run -d web-server \
  --dart-define=FASTMODA_URL=http://localhost:5000
```

## Useful Links

- 📖 Full Docs: `docs/CONTAINERIZED_EMULATOR_README.md`
- 🔨 Build Guide: `docs/EMULATOR_BUILD_GUIDE.md`
- 📋 Architecture: `docs/EMULATOR_IMPROVEMENT_PLAN.md`
- 📊 Summary: `docs/EMULATOR_IMPLEMENTATION_SUMMARY.md`

## Status Check

```bash
# Show what's running
echo "=== Containers ===" && docker ps
echo "=== Health ===" && curl -s http://localhost:8080 > /dev/null && echo "✅ App" || echo "❌ App"
echo "=== Networks ===" && docker network ls | grep moda
```

---

**Quick Start**: `docker-compose up -d fastmoda-api flutter-emulator moda-signal-mock-server`

**Test It**: `python3 tests/emulator_smoke_tests.py`

**Open**: http://localhost:8080

**Done!** ✨
