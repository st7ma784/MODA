#!/bin/bash

################################################################################
# MODA vs FastMODA Test Suite Quick Start
#
# A menu-driven interface to quickly set up and run the comprehensive test suite
#
# Usage: bash test_suite_quickstart.sh
################################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTS_DIR="$SCRIPT_DIR"
ROOT_DIR="$(dirname "$TESTS_DIR")"

# Global variables
RESULTS_DIR="$TESTS_DIR/results"
DATA_DIR="$TESTS_DIR/test_data"

################################################################################
# Utility functions
################################################################################

print_header() {
    echo -e "\n${BLUE}═══════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════════════${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

print_step() {
    echo -e "\n${YELLOW}→ $1${NC}"
}

check_command() {
    if command -v "$1" &> /dev/null; then
        print_success "$1 is installed"
        return 0
    else
        print_error "$1 is not installed"
        return 1
    fi
}

wait_for_container() {
    local container=$1
    local timeout=60
    local elapsed=0
    
    print_info "Waiting for $container to be healthy..."
    
    while [ $elapsed -lt $timeout ]; do
        if docker ps --filter "name=$container" --filter "health=healthy" -q | grep -q .; then
            print_success "$container is healthy"
            return 0
        fi
        echo -n "."
        sleep 1
        ((elapsed++))
    done
    
    print_error "Timeout waiting for $container"
    return 1
}

################################################################################
# Prerequisites check
################################################################################

check_prerequisites() {
    print_header "Checking Prerequisites"
    
    local all_ok=true
    
    # Check Docker
    if check_command docker; then
        docker --version
    else
        print_error "Docker is required. Install from: https://docs.docker.com/get-docker/"
        all_ok=false
    fi
    
    # Check Docker daemon
    if docker ps > /dev/null 2>&1; then
        print_success "Docker daemon is running"
    else
        print_error "Docker daemon is not running. Start it with: docker start"
        all_ok=false
    fi
    
    # Check Docker Compose
    if check_command docker-compose; then
        docker-compose --version
    else
        print_warning "docker-compose not found (optional). Using 'docker compose' instead"
    fi
    
    # Check Python
    if check_command python3; then
        python3 --version
    else
        print_error "Python 3 is required"
        all_ok=false
    fi
    
    # Check disk space
    available=$(df "$TESTS_DIR" | awk 'NR==2 {print $4}')
    if [ "$available" -gt 5242880 ]; then  # 5GB
        print_success "Sufficient disk space available: $((available / 1048576))GB"
    else
        print_warning "Low disk space: $((available / 1048576))GB (recommended 10GB+)"
    fi
    
    if [ "$all_ok" = true ]; then
        print_success "All prerequisites satisfied"
        return 0
    else
        print_error "Some prerequisites are missing"
        return 1
    fi
}

################################################################################
# Test data setup
################################################################################

prepare_test_data() {
    print_header "Preparing Test Data"
    
    # Create directories
    mkdir -p "$DATA_DIR/signals" "$RESULTS_DIR"
    
    # Check if signals already exist
    if [ -d "$DATA_DIR/signals" ] && [ "$(ls -A "$DATA_DIR/signals")" ]; then
        print_warning "Test signals already exist. Skipping generation..."
        print_info "Signals directory: $DATA_DIR/signals"
        return 0
    fi
    
    print_step "Generating test signals..."
    
    cd "$TESTS_DIR"
    python3 test_comparison_harness.py --mode prepare
    
    if [ $? -eq 0 ]; then
        print_success "Test data prepared successfully"
        return 0
    else
        print_error "Failed to prepare test data"
        return 1
    fi
}

################################################################################
# Build containers
################################################################################

build_containers() {
    print_header "Building Docker Containers"
    
    cd "$TESTS_DIR"
    
    print_step "Building MODA test container..."
    docker build \
        -f "$ROOT_DIR/Dockerfile" \
        --target moda-test \
        -t moda-test:latest \
        "$ROOT_DIR"
    
    if [ $? -ne 0 ]; then
        print_error "Failed to build moda-test image"
        return 1
    fi
    print_success "moda-test image built"
    
    print_step "Building FastMODA container..."
    docker build \
        -f "$ROOT_DIR/FastMODA/Dockerfile" \
        -t fastmoda:latest \
        "$ROOT_DIR/FastMODA"
    
    if [ $? -ne 0 ]; then
        print_warning "Failed to build fastmoda image (optional)"
        print_info "This is OK if you only want to test MODA"
    else
        print_success "fastmoda image built"
    fi
    
    print_step "Building test harness container..."
    docker build \
        -f "$TESTS_DIR/Dockerfile.harness" \
        -t test-harness:latest \
        "$ROOT_DIR"
    
    if [ $? -eq 0 ]; then
        print_success "test-harness image built"
        return 0
    else
        print_error "Failed to build test-harness image"
        return 1
    fi
}

################################################################################
# Start containers
################################################################################

start_containers_detached() {
    print_header "Starting Containers"
    
    cd "$TESTS_DIR"
    
    print_step "Starting moda-matlab-test..."
    docker-compose -f docker-compose.test.yml up -d moda-matlab
    
    if [ $? -ne 0 ]; then
        print_error "Failed to start moda-matlab"
        return 1
    fi
    
    print_step "Starting fastmoda-python-test..."
    docker-compose -f docker-compose.test.yml up -d fastmoda-python
    
    if [ $? -ne 0 ]; then
        print_warning "Failed to start fastmoda-python (optional)"
    fi
    
    # Wait for containers to be healthy
    print_step "Waiting for containers to be healthy..."
    if ! wait_for_container moda-matlab-test; then
        print_error "moda-matlab failed to become healthy"
        print_info "Check logs with: docker logs moda-matlab-test"
        return 1
    fi
    
    print_success "All containers started successfully"
    return 0
}

################################################################################
# Run tests
################################################################################

run_all_tests() {
    print_header "Running Tests"
    
    cd "$TESTS_DIR"
    
    print_step "Running MODA tests..."
    docker exec moda-matlab-test matlab -batch \
        "addpath(genpath('/workspace')); tester = TestAllComponents('/workspace/results'); tester.runAllTests();"
    
    if [ $? -ne 0 ]; then
        print_warning "MODA tests had issues (check logs)"
    else
        print_success "MODA tests completed"
    fi
    
    print_step "Running FastMODA tests..."
    python3 test_comparison_harness.py --mode fastmoda || \
        print_warning "FastMODA tests had issues (optional)"
    
    print_step "Comparing results..."
    python3 test_comparison_harness.py --mode compare
    
    print_step "Generating plots..."
    python3 test_comparison_harness.py --mode plot
    
    print_step "Generating report..."
    python3 test_comparison_harness.py --mode report
    
    print_success "All tests completed"
    return 0
}

run_moda_only() {
    print_header "Running MODA Tests Only"
    
    cd "$TESTS_DIR"
    
    if ! start_containers_detached; then
        return 1
    fi
    
    print_step "Running MATLAB test suite..."
    docker exec moda-matlab-test matlab -batch \
        "addpath(genpath('/workspace')); tester = TestAllComponents('/workspace/results/moda'); tester.runAllTests();"
    
    print_success "MODA tests completed"
    return 0
}

run_fastmoda_only() {
    print_header "Running FastMODA Tests Only"
    
    cd "$TESTS_DIR"
    
    if ! start_containers_detached; then
        return 1
    fi
    
    print_step "Running FastMODA API tests..."
    python3 test_comparison_harness.py --mode fastmoda
    
    print_success "FastMODA tests completed"
    return 0
}

################################################################################
# View results
################################################################################

view_results() {
    print_header "Test Results"
    
    local report_file="$RESULTS_DIR/comparison/comparison_report.txt"
    
    if [ ! -f "$report_file" ]; then
        print_error "Report file not found: $report_file"
        print_info "Run tests first with: bash test_suite_quickstart.sh"
        return 1
    fi
    
    cat "$report_file"
    return 0
}

view_plots() {
    print_header "Opening Results Dashboard"
    
    if ! check_command python3; then
        print_error "Python 3 is required to run dashboard"
        return 1
    fi
    
    cd "$TESTS_DIR"
    print_info "Launching interactive dashboard..."
    print_info "Dashboard will open in your default web browser or as a GUI window"
    
    python3 dashboard_gui.py --results "$RESULTS_DIR"
    
    return 0
}

################################################################################
# Cleanup
################################################################################

cleanup_containers() {
    print_header "Cleanup"
    
    cd "$TESTS_DIR"
    
    print_step "Stopping containers..."
    docker-compose -f docker-compose.test.yml down
    
    if [ $? -eq 0 ]; then
        print_success "Containers stopped"
    else
        print_warning "Some containers may not have stopped cleanly"
    fi
}

cleanup_all() {
    print_header "Complete Cleanup"
    
    cd "$TESTS_DIR"
    
    print_step "Stopping and removing containers..."
    docker-compose -f docker-compose.test.yml down -v
    
    print_step "Removing images..."
    docker rmi moda-test:latest 2>/dev/null || true
    docker rmi fastmoda:latest 2>/dev/null || true
    docker rmi test-harness:latest 2>/dev/null || true
    
    print_step "Removing test results..."
    rm -rf "$RESULTS_DIR"
    
    print_success "Cleanup completed. All test artifacts removed."
}

################################################################################
# Menu system
################################################################################

show_menu() {
    echo ""
    print_info "Select an option:"
    echo ""
    echo "  ${BLUE}Setup & Build${NC}"
    echo "    1) Check prerequisites"
    echo "    2) Prepare test data"
    echo "    3) Build containers"
    echo "    4) Quick setup (all of above)"
    echo ""
    echo "  ${BLUE}Run Tests${NC}"
    echo "    5) Run all tests (full suite)"
    echo "    6) Run MODA tests only"
    echo "    7) Run FastMODA tests only"
    echo ""
    echo "  ${BLUE}View Results${NC}"
    echo "    8) View test report (text)"
    echo "    9) View interactive dashboard (GUI)"
    echo ""
    echo "  ${BLUE}Maintenance${NC}"
    echo "    10) Stop containers"
    echo "    11) Cleanup (remove containers)"
    echo "    12) Full cleanup (remove everything)"
    echo ""
    echo "  ${BLUE}Help & Exit${NC}"
    echo "    13) Show documentation"
    echo "    0) Exit"
    echo ""
}

show_documentation() {
    print_header "Test Suite Documentation"
    
    if [ -f "$TESTS_DIR/README_TEST_SUITE.md" ]; then
        less "$TESTS_DIR/README_TEST_SUITE.md"
    else
        print_error "README_TEST_SUITE.md not found"
        print_info "Documentation available at: $TESTS_DIR/README_TEST_SUITE.md"
    fi
}

main_loop() {
    while true; do
        show_menu
        read -p "Enter option (0-13): " choice
        
        case $choice in
            1) check_prerequisites ;;
            2) prepare_test_data ;;
            3) build_containers ;;
            4)
                check_prerequisites && \
                prepare_test_data && \
                build_containers
                ;;
            5) build_containers && run_all_tests ;;
            6) build_containers && run_moda_only ;;
            7) build_containers && run_fastmoda_only ;;
            8) view_results ;;
            9) view_plots ;;
            10) cleanup_containers ;;
            11) cleanup_containers ;;
            12) cleanup_all ;;
            13) show_documentation ;;
            0)
                print_info "Exiting..."
                exit 0
                ;;
            *)
                print_error "Invalid option. Please enter 0-13."
                ;;
        esac
        
        read -p "Press Enter to continue..."
    done
}

################################################################################
# Entry point
################################################################################

print_header "MODA vs FastMODA Test Suite Quick Start"

print_info "Test Suite Location: $TESTS_DIR"
print_info "Results Directory: $RESULTS_DIR"
print_info "Test Data Directory: $DATA_DIR"

main_loop
