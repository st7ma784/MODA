#!/bin/bash
# FastMODA Quick Setup Script

set -e

echo "======================================"
echo "   FastMODA Setup & Deployment"
echo "======================================"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check GPU availability
check_gpu() {
    if command_exists nvidia-smi; then
        echo "✓ NVIDIA GPU detected:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
        return 0
    else
        echo "✗ No NVIDIA GPU detected"
        return 1
    fi
}

# Function to check Docker
check_docker() {
    if command_exists docker; then
        echo "✓ Docker installed: $(docker --version)"
        return 0
    else
        echo "✗ Docker not installed"
        return 1
    fi
}

# Function to check nvidia-docker
check_nvidia_docker() {
    if docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi &>/dev/null; then
        echo "✓ NVIDIA Docker runtime available"
        return 0
    else
        echo "✗ NVIDIA Docker runtime not available"
        return 1
    fi
}

# Main menu
show_menu() {
    echo ""
    echo "Select deployment option:"
    echo "1) CPU-only (Local Python)"
    echo "2) GPU-accelerated (Local Python)"
    echo "3) Docker CPU"
    echo "4) Docker GPU"
    echo "5) Docker Development Mode"
    echo "6) Run Tests"
    echo "7) Benchmark GPU vs CPU"
    echo "8) Exit"
    echo ""
    read -p "Enter choice [1-8]: " choice
    
    case $choice in
        1) setup_cpu_local ;;
        2) setup_gpu_local ;;
        3) setup_docker_cpu ;;
        4) setup_docker_gpu ;;
        5) setup_docker_dev ;;
        6) run_tests ;;
        7) run_benchmark ;;
        8) exit 0 ;;
        *) echo "Invalid option"; show_menu ;;
    esac
}

# Setup CPU (local)
setup_cpu_local() {
    echo ""
    echo "Setting up CPU version (local Python)..."
    
    pip install -r requirements.txt
    
    echo ""
    echo "✓ Setup complete!"
    echo ""
    echo "To run the application:"
    echo "  python app.py"
    echo ""
    echo "Then open: http://localhost:5000"
    
    read -p "Start now? [y/N]: " start
    if [[ $start =~ ^[Yy]$ ]]; then
        python app.py
    fi
}

# Setup GPU (local)
setup_gpu_local() {
    echo ""
    echo "Setting up GPU version (local Python)..."
    
    if ! check_gpu; then
        echo "Warning: No GPU detected. Continuing with CPU fallback."
    fi
    
    pip install -r requirements.txt
    pip install -r requirements-gpu.txt --index-url https://download.pytorch.org/whl/cu118
    
    echo ""
    echo "✓ Setup complete!"
    echo ""
    echo "Testing GPU availability..."
    python -c "from fastmoda.gpu_utils import get_gpu_info; import json; print(json.dumps(get_gpu_info(), indent=2))"
    
    echo ""
    echo "To run the application:"
    echo "  python app_gpu.py"
    echo ""
    echo "Then open: http://localhost:5000"
    
    read -p "Start now? [y/N]: " start
    if [[ $start =~ ^[Yy]$ ]]; then
        export USE_GPU=auto
        python app_gpu.py
    fi
}

# Setup Docker CPU
setup_docker_cpu() {
    echo ""
    echo "Setting up Docker (CPU)..."
    
    if ! check_docker; then
        echo "Please install Docker first: https://docs.docker.com/get-docker/"
        return 1
    fi
    
    echo "Building Docker image..."
    docker-compose build fastmoda-cpu
    
    echo "Starting container..."
    docker-compose up -d fastmoda-cpu
    
    echo ""
    echo "✓ Container started!"
    echo ""
    echo "Access the application at: http://localhost:5000"
    echo ""
    echo "To view logs: docker-compose logs -f fastmoda-cpu"
    echo "To stop: docker-compose down"
}

# Setup Docker GPU
setup_docker_gpu() {
    echo ""
    echo "Setting up Docker (GPU)..."
    
    if ! check_docker; then
        echo "Please install Docker first: https://docs.docker.com/get-docker/"
        return 1
    fi
    
    if ! check_gpu; then
        echo "No GPU detected. Please use CPU version instead."
        return 1
    fi
    
    if ! check_nvidia_docker; then
        echo ""
        echo "NVIDIA Docker runtime not available."
        echo "Install with:"
        echo ""
        echo "  distribution=\$(. /etc/os-release;echo \$ID\$VERSION_ID)"
        echo "  curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -"
        echo "  curl -s -L https://nvidia.github.io/nvidia-docker/\$distribution/nvidia-docker.list | \\"
        echo "    sudo tee /etc/apt/sources.list.d/nvidia-docker.list"
        echo "  sudo apt-get update && sudo apt-get install -y nvidia-docker2"
        echo "  sudo systemctl restart docker"
        return 1
    fi
    
    echo "Building Docker GPU image..."
    docker-compose --profile gpu build fastmoda-gpu
    
    echo "Starting GPU container..."
    docker-compose --profile gpu up -d fastmoda-gpu
    
    echo ""
    echo "✓ GPU container started!"
    echo ""
    echo "Access the application at: http://localhost:5001"
    echo ""
    echo "To check GPU: docker exec fastmoda-gpu nvidia-smi"
    echo "To view logs: docker-compose logs -f fastmoda-gpu"
    echo "To stop: docker-compose --profile gpu down"
}

# Setup Docker Dev
setup_docker_dev() {
    echo ""
    echo "Starting development environment..."
    
    if ! check_docker; then
        echo "Please install Docker first: https://docs.docker.com/get-docker/"
        return 1
    fi
    
    docker-compose --profile dev up fastmoda-dev
}

# Run tests
run_tests() {
    echo ""
    echo "Running tests..."
    
    if [ -f test_features.py ]; then
        python test_features.py
    else
        echo "Test file not found"
    fi
    
    echo ""
    read -p "Press Enter to continue..."
    show_menu
}

# Run benchmark
run_benchmark() {
    echo ""
    echo "Running GPU vs CPU benchmark..."
    
    python -c "
from fastmoda.gpu_utils import benchmark_gpu_vs_cpu, is_gpu_available
import json

if not is_gpu_available():
    print('Warning: GPU not available. Running CPU-only benchmark.')

try:
    result = benchmark_gpu_vs_cpu(signal_length=100000, num_runs=5)
    print(json.dumps(result, indent=2))
    
    if 'speedup' in result:
        print(f\"\nSpeedup: {result['speedup']:.1f}x\")
except Exception as e:
    print(f'Error: {e}')
"
    
    echo ""
    read -p "Press Enter to continue..."
    show_menu
}

# Check system
echo "Checking system..."
echo ""

check_docker
check_gpu

if check_gpu; then
    check_nvidia_docker
fi

# Show menu
show_menu
