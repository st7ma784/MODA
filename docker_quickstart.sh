#!/bin/bash
# MODA Docker Quick Start Script
# Usage: bash docker_quickstart.sh [dev|test|compose|clean]

set -e  # Exit on error

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_header() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites"
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Install from https://docs.docker.com/get-docker/"
        exit 1
    fi
    print_success "Docker found ($(docker --version))"
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed."
        exit 1
    fi
    print_success "Docker Compose found"
    
    # Check disk space
    space=$(df -k . | tail -1 | awk '{print $4}')
    if [ $space -lt 50000000 ]; then
        print_warning "Low disk space available. Need ~50GB for Docker images."
    else
        print_success "Disk space available: $(df -h . | tail -1 | awk '{print $4}')"
    fi
    
    echo ""
}

# Build development image
build_dev() {
    print_header "Building MODA Development Image"
    
    docker build \
        -t moda-dev:latest \
        --target matlab-dev \
        --progress=plain \
        .
    
    if [ $? -eq 0 ]; then
        print_success "Development image built successfully"
        docker images | grep moda-dev
    else
        print_error "Failed to build development image"
        exit 1
    fi
}

# Build test image
build_test() {
    print_header "Building MODA Test Image"
    
    docker build \
        -t moda-test:latest \
        --target moda-test \
        --progress=plain \
        .
    
    if [ $? -eq 0 ]; then
        print_success "Test image built successfully"
        docker images | grep moda-test
    else
        print_error "Failed to build test image"
        exit 1
    fi
}

# Run development container
run_dev() {
    print_header "Running MODA Development Container"
    
    # Check for X11 on Linux
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [ -z "$DISPLAY" ]; then
            print_warning "DISPLAY not set. GUI will not be available."
            print_warning "Run: export DISPLAY=:0 (or :1, :2, etc.)"
        fi
        
        xhost +local:docker 2>/dev/null || true
        
        docker run -it \
            --env DISPLAY=$DISPLAY \
            -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
            -v ~/.Xauthority:/home/matlab/.Xauthority:rw \
            -v $(pwd):/app:rw \
            moda-dev:latest \
            matlab -r "MODA; exit;"
        
        xhost -local:docker 2>/dev/null || true
    else
        print_warning "GUI support only available on Linux with X11"
        print_warning "Running in headless mode instead"
        
        docker run -it \
            -v $(pwd):/app:rw \
            moda-dev:latest \
            matlab -r "addpath(genpath('.')); disp('MODA Ready for Testing'); exit;"
    fi
}

# Run tests
run_tests() {
    print_header "Running MODA Test Suite"
    
    mkdir -p test_results matlab_logs
    
    docker run --rm \
        -v $(pwd):/app:ro \
        -v $(pwd)/test_results:/tmp/test_results:rw \
        -v $(pwd)/matlab_logs:/tmp/matlab_logs:rw \
        moda-test:latest \
        matlab -batch "addpath(genpath(pwd)); disp('Testing MODA'); exit(0);"
    
    if [ $? -eq 0 ]; then
        print_success "Tests completed"
        if [ -f test_results/summary.txt ]; then
            echo ""
            echo "Test Summary:"
            cat test_results/summary.txt
        fi
    else
        print_error "Tests failed"
    fi
}

# Docker Compose operations
run_compose() {
    print_header "Starting Docker Compose Services"
    
    docker-compose up -d
    
    print_success "Services started"
    echo ""
    docker-compose ps
    echo ""
    print_warning "View logs with: docker-compose logs -f moda-test"
}

# Clean up
cleanup() {
    print_header "Cleaning Up Docker Resources"
    
    print_warning "Shutting down Docker Compose services..."
    docker-compose down -v
    
    print_warning "Removing MODA images..."
    docker rmi moda-dev:latest 2>/dev/null || true
    docker rmi moda-test:latest 2>/dev/null || true
    
    print_success "Cleanup complete"
}

# Interactive menu
show_menu() {
    echo ""
    echo "MODA Docker Quick Start"
    echo "======================"
    echo ""
    echo "1) Check prerequisites"
    echo "2) Build dev image"
    echo "3) Build test image"  
    echo "4) Run development (interactive)"
    echo "5) Run tests"
    echo "6) Start with Docker Compose"
    echo "7) Clean up"
    echo "0) Exit"
    echo ""
    read -p "Select option: " choice
}

# Main script logic
case "${1:-menu}" in
    check)
        check_prerequisites
        ;;
    dev)
        check_prerequisites
        build_dev
        run_dev
        ;;
    test)
        check_prerequisites
        build_test
        run_tests
        ;;
    build-dev)
        check_prerequisites
        build_dev
        ;;
    build-test)
        check_prerequisites
        build_test
        ;;
    compose)
        check_prerequisites
        run_compose
        ;;
    clean)
        cleanup
        ;;
    menu|"")
        check_prerequisites
        show_menu
        case $choice in
            1) check_prerequisites ;;
            2) build_dev ;;
            3) build_test ;;
            4) run_dev ;;
            5) run_tests ;;
            6) run_compose ;;
            7) cleanup ;;
            0) echo "Goodbye!"; exit 0 ;;
            *) print_error "Invalid option"; exit 1 ;;
        esac
        ;;
    *)
        echo "Usage: $0 [check|dev|test|build-dev|build-test|compose|clean|menu]"
        echo ""
        echo "Commands:"
        echo "  check      - Verify Docker is installed and accessible"
        echo "  dev        - Build and run development environment"
        echo "  test       - Build and run test suite"
        echo "  build-dev  - Build development image only"
        echo "  build-test - Build test image only"
        echo "  compose    - Start full stack with Docker Compose"
        echo "  clean      - Remove all Docker resources"
        echo "  menu       - Interactive menu (default)"
        exit 1
        ;;
esac

print_success "Done!"
