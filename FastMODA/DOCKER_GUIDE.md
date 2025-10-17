# FastMODA Docker Deployment Guide

## Overview

FastMODA supports multiple deployment modes:
- **CPU-only**: Standard deployment for systems without GPU
- **GPU-accelerated**: High-performance deployment using NVIDIA CUDA
- **Development**: Hot-reload environment for development

## Prerequisites

### For CPU Deployment
- Docker Engine 20.10+
- Docker Compose 2.0+
- 4GB RAM minimum
- 10GB disk space

### For GPU Deployment
- NVIDIA GPU with CUDA support
- NVIDIA Docker runtime (`nvidia-docker2`)
- CUDA 11.8+ compatible GPU
- 8GB GPU memory recommended

## Quick Start

### 1. CPU Deployment (Default)

```bash
# Clone the repository
cd /data/MODA/FastMODA

# Build and run CPU version
docker-compose up -d fastmoda-cpu

# Access the application
open http://localhost:5000
```

### 2. GPU Deployment

```bash
# Install NVIDIA Docker runtime (if not installed)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Test NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Build and run GPU version
docker-compose --profile gpu up -d fastmoda-gpu

# Access the application
open http://localhost:5001
```

### 3. Development Mode

```bash
# Run with hot-reload
docker-compose --profile dev up fastmoda-dev

# Application auto-reloads on code changes
# Access at http://localhost:5000
```

## Configuration

### Environment Variables

Create a `.env` file in the FastMODA directory:

```env
# GPU Settings
USE_GPU=auto                    # auto|true|false
CUDA_VISIBLE_DEVICES=0          # GPU device ID

# Flask Settings
FLASK_ENV=production            # production|development
FLASK_DEBUG=0                   # 0|1

# Performance
MAX_UPLOAD_SIZE=50              # MB
WORKER_PROCESSES=4              # Number of worker processes

# Security (for production)
SECRET_KEY=your-secret-key-here
ALLOWED_HOSTS=localhost,yourdomain.com
```

### Docker Compose Profiles

- **Default**: CPU-only deployment
- **gpu**: GPU-accelerated deployment
- **dev**: Development mode
- **production**: Production with Nginx reverse proxy

## Building Images

### Build CPU Image

```bash
docker build -t fastmoda:cpu \
  --target base \
  -f Dockerfile .
```

### Build GPU Image

```bash
docker build -t fastmoda:gpu \
  --target gpu \
  --build-arg CUDA_VERSION=11.8 \
  -f Dockerfile .
```

### Multi-architecture Build

```bash
# Build for multiple platforms
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t fastmoda:latest \
  --push .
```

## Performance Optimization

### GPU Memory Management

```yaml
# In docker-compose.yml
services:
  fastmoda-gpu:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
        limits:
          memory: 8G
          cpus: '4'
```

### Scaling with Multiple GPUs

```yaml
# Use all available GPUs
environment:
  - CUDA_VISIBLE_DEVICES=0,1,2,3
  
# Or specific GPUs
environment:
  - CUDA_VISIBLE_DEVICES=0,2
```

## Monitoring

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f fastmoda-gpu

# Last 100 lines
docker-compose logs --tail=100 fastmoda-cpu
```

### GPU Monitoring

```bash
# Inside container
docker exec -it fastmoda-gpu nvidia-smi

# Continuous monitoring
docker exec -it fastmoda-gpu watch -n 1 nvidia-smi
```

### Health Checks

```bash
# Check service health
docker-compose ps

# Manual health check
curl http://localhost:5000/api/gpu-info
```

## Production Deployment

### With Nginx Reverse Proxy

1. Create `nginx.conf`:

```nginx
events {
    worker_connections 1024;
}

http {
    upstream fastmoda {
        server fastmoda-cpu:5000;
    }
    
    server {
        listen 80;
        server_name yourdomain.com;
        
        client_max_body_size 50M;
        
        location / {
            proxy_pass http://fastmoda;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
```

2. Deploy with Nginx:

```bash
docker-compose --profile production up -d
```

### SSL/TLS Configuration

```nginx
server {
    listen 443 ssl http2;
    server_name yourdomain.com;
    
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    
    # ... rest of config
}
```

## CI/CD Integration

### GitHub Actions

The included `.github/workflows/ci-cd.yml` provides:

1. **Automated Testing**: Runs on every push/PR
2. **Docker Image Building**: Multi-arch builds
3. **GPU Testing**: On self-hosted runners
4. **Automated Deployment**: On releases

#### Setup Self-Hosted GPU Runner

```bash
# On your GPU server
mkdir actions-runner && cd actions-runner
curl -o actions-runner-linux-x64-2.311.0.tar.gz -L \
  https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-linux-x64-2.311.0.tar.gz
tar xzf ./actions-runner-linux-x64-2.311.0.tar.gz

# Configure
./config.sh --url https://github.com/OWNER/REPO --token YOUR_TOKEN --labels gpu

# Install as service
sudo ./svc.sh install
sudo ./svc.sh start
```

## Troubleshooting

### GPU Not Detected

```bash
# Check NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Check container GPU access
docker exec fastmoda-gpu python -c "import torch; print(torch.cuda.is_available())"

# Verify CUDA version
docker exec fastmoda-gpu nvcc --version
```

### Out of Memory Errors

```python
# In app_gpu.py, add batch processing
def process_large_signal(x, fs, batch_size=10000):
    results = []
    for i in range(0, len(x), batch_size):
        batch = x[i:i+batch_size]
        result = sliding_fft_gpu(batch, fs)
        results.append(result)
        torch.cuda.empty_cache()  # Clear GPU memory
    return combine_results(results)
```

### Container Networking Issues

```bash
# Check network
docker network inspect fastmoda-network

# Recreate network
docker-compose down
docker network prune
docker-compose up -d
```

## Benchmarking

### GPU vs CPU Performance

```bash
# Inside container
docker exec fastmoda-gpu python -c "
from fastmoda.gpu_utils import benchmark_gpu_vs_cpu
import json
result = benchmark_gpu_vs_cpu(signal_length=100000, num_runs=10)
print(json.dumps(result, indent=2))
"
```

Expected speedup: 5-20x depending on GPU model.

## Updating

### Pull Latest Images

```bash
docker-compose pull
docker-compose up -d
```

### Rebuild After Code Changes

```bash
docker-compose build --no-cache
docker-compose up -d
```

## Data Persistence

### Volume Mapping

```yaml
volumes:
  - ./data:/app/data              # Input data
  - ./uploads:/app/uploads        # Uploaded files
  - ./results:/app/results        # Analysis results
  - ./models:/app/models          # Saved models
```

### Backup

```bash
# Backup volumes
docker run --rm \
  -v fastmoda_uploads:/data \
  -v $(pwd):/backup \
  alpine tar czf /backup/uploads-backup.tar.gz -C /data .

# Restore
docker run --rm \
  -v fastmoda_uploads:/data \
  -v $(pwd):/backup \
  alpine tar xzf /backup/uploads-backup.tar.gz -C /data
```

## Security Considerations

1. **Use secrets for production**:
   ```bash
   docker secret create db_password ./db_password.txt
   ```

2. **Limit container privileges**:
   ```yaml
   security_opt:
     - no-new-privileges:true
   read_only: true
   ```

3. **Network isolation**:
   ```yaml
   networks:
     frontend:
     backend:
       internal: true
   ```

## Support

For issues and questions:
- GitHub Issues: https://github.com/OWNER/MODA/issues
- Documentation: `/data/MODA/docs/`
- GPU Guide: This file
