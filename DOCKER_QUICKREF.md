# MODA Docker Quick Reference

**Quick Start:**
```bash
bash docker_quickstart.sh        # Interactive menu
bash docker_quickstart.sh check  # Verify Docker installed
bash docker_quickstart.sh dev    # Build dev image & run
bash docker_quickstart.sh test   # Build test image & run tests
bash docker_quickstart.sh compose# Full stack with Docker Compose
```

---

## Common Workflows

### Development (GUI on Linux)

```bash
# 1. Build development image
docker build -t moda-dev:latest --target matlab-dev .

# 2. Allow X11 access
xhost +local:docker

# 3. Run with display forwarding
docker run -it \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v ~/.Xauthority:/home/matlab/.Xauthority:ro \
  moda-dev:latest \
  matlab

# 4. Inside container, run MODA
>> addpath(genpath('/app'))
>> MODA

# 5. Cleanup X11 access
xhost -local:docker
```

### Testing (Headless - No Display Required)

```bash
# 1. Build test image
docker build -t moda-test:latest --target moda-test .

# 2. Run tests
docker run --rm \
  -v $(pwd)/test_results:/tmp/test_results \
  moda-test:latest \
  matlab -batch "runtests tests/; exit(0);"

# 3. View results
cat test_results/summary.txt
```

### Full Stack (Docker Compose)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f moda-test
docker-compose logs -f fastmoda-api

# Run one-off command
docker-compose run --rm moda-test matlab -batch "runtests; exit(0);"

# Stop everything
docker-compose down
```

### Interactive Shell

```bash
# Get shell access
docker run -it \
  -v $(pwd):/app \
  moda-dev:latest \
  /bin/bash

# Inside container
$ cd /app
$ matlab -r "MODA; exit;"
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **License not found** | `docker run -e MLM_LICENSE_FILE=hostname@port moda-dev` |
| **X11 connection refused** | `xhost +local:docker` before running container |
| **Out of memory** | `docker run -m 8g --memory-swap 8g moda-dev` |
| **Module not found** | Set MATLABPATH: `-e MATLABPATH=/app:/app/allguis` |
| **Build fails** | Check: `docker build -v --progress=plain` for verbose output |

---

## Image Management

```bash
# List images
docker images | grep moda

# View image history
docker history moda-dev:latest

# Remove image
docker rmi moda-dev:latest

# Inspect image
docker inspect moda-dev:latest
```

---

## Container Management

```bash
# List running containers
docker ps | grep moda

# View logs
docker logs -f <container_id>

# Execute command in running container
docker exec -it <container_id> matlab

# Stop container
docker stop <container_id>

# Remove container
docker rm <container_id>
```

---

## Key Files

| File | Purpose |
|------|---------|
| `Dockerfile` | Multi-stage build for dev/test/prod |
| `docker-compose.yml` | Orchestrate multiple containers |
| `docker_quickstart.sh` | Interactive setup script |
| `.dockerignore` | Exclude files from build context |
| `docs/DOCKER_SETUP_GUIDE.md` | Comprehensive guide |

---

## Environment Variables

```bash
# MATLAB Configuration
-e MATLABPATH=/app:/app/allguis  # Add paths to MATLAB search path
-e MATLAB_LOG_DIR=/tmp/logs      # Log directory

# Display (GUI only)
-e DISPLAY=$DISPLAY              # X11 display

# License (if needed)
-e MLM_LICENSE_FILE=host@port    # Network license server
```

---

## Performance Tips

- **Cache builds:** `docker build --cache-from moda-dev:latest ...`
- **Multi-stage:** Only needed files in production image
- **Volume mounts:** Use `:ro` for read-only test runs  
- **Resource limits:** `-m 8g --memory-swap 8g` for large computations

---

## GitHub Actions Integration

See `.github/workflows/test-moda.yml` for automated testing on every push.

Run locally to simulate CI:
```bash
docker build -t moda-test:latest --target moda-test .
docker run --rm moda-test:latest bash tests/test_integration.sh
```

---

## Next Steps

1. ✅ **Read**: `docs/DOCKER_SETUP_GUIDE.md` (comprehensive)
2. ✅ **Build**: `bash docker_quickstart.sh dev` (dev image)
3. ✅ **Test**: `bash docker_quickstart.sh test` (run tests)
4. ✅ **Deploy**: `docker-compose up -d` (full stack)

---

## Resources

- [MATLAB Docker GitHub](https://github.com/mathworks-ref-arch/matlab-dockerfile)
- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Spec](https://github.com/compose-spec/compose-spec)
