#!/bin/bash
#
# Integration Test Harness for Containerized Emulator
# ===================================================
# 
# Brings up the complete MODA testing environment:
#   - FastMODA API server
#   - Flutter web emulator
#   - Signal mock server
# 
# Then runs smoke tests to verify all components are healthy.
#
# Usage:
#   ./tests/emulator_integration_test.sh              # Start environment
#   ./tests/emulator_integration_test.sh test         # Also run smoke tests
#   ./tests/emulator_integration_test.sh down         # Tear down environment
#
# Environment:
#   DOCKER_COMPOSE: Path to docker-compose.yml (default: ./docker-compose.yml)
#   WAIT_TIMEOUT: Seconds to wait for health checks (default: 60)

set -e

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

DOCKER_COMPOSE="${DOCKER_COMPOSE:-.}/docker-compose.yml"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-60}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ──────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────────────────────────────────────

log_info() {
    echo -e "${BLUE}ℹ️  $*${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $*${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $*${NC}"
}

log_error() {
    echo -e "${RED}❌ $*${NC}"
}

wait_for_url() {
    local url="$1"
    local timeout="$2"
    local elapsed=0

    while [ $elapsed -lt "$timeout" ]; do
        if curl -f --silent --max-time 2 "$url" > /dev/null 2>&1; then
            return 0
        fi
        elapsed=$((elapsed + 2))
        sleep 2
    done

    return 1
}

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

main() {
    local command="${1:-up}"

    case "$command" in
        up)
            cmd_up
            ;;
        down)
            cmd_down
            ;;
        test)
            cmd_up
            cmd_test
            ;;
        restart)
            cmd_down
            cmd_up
            ;;
        *)
            log_error "Unknown command: $command"
            echo ""
            echo "Usage: $0 [up|down|test|restart]"
            echo ""
            echo "Commands:"
            echo "  up       — Start containerized environment (default)"
            echo "  down     — Tear down environment"
            echo "  test     — Start environment and run smoke tests"
            echo "  restart  — Restart all services"
            exit 1
            ;;
    esac
}

cmd_up() {
    log_info "Starting containerized emulator environment..."
    cd "$REPO_ROOT"

    # Start services
    docker-compose -f "$DOCKER_COMPOSE" up -d fastmoda-api flutter-emulator moda-signal-mock-server

    echo ""
    log_info "⏳ Waiting for services to be healthy..."
    echo ""

    # Wait for FastMODA
    log_info "Waiting for FastMODA API (http://localhost:5000/health)..."
    if wait_for_url "http://localhost:5000/health" "$WAIT_TIMEOUT"; then
        log_success "FastMODA API is ready"
    else
        log_error "FastMODA API failed to start within ${WAIT_TIMEOUT}s"
        docker-compose -f "$DOCKER_COMPOSE" logs fastmoda-api
        exit 1
    fi

    # Wait for Flutter web app
    log_info "Waiting for Flutter web app (http://localhost:8080/)..."
    if wait_for_url "http://localhost:8080/" "$WAIT_TIMEOUT"; then
        log_success "Flutter web app is ready"
    else
        log_error "Flutter web app failed to start within ${WAIT_TIMEOUT}s"
        docker-compose -f "$DOCKER_COMPOSE" logs flutter-emulator
        exit 1
    fi

    # Wait for mock signal server
    log_info "Waiting for Mock Signal Server (http://localhost:8081/health)..."
    if wait_for_url "http://localhost:8081/health" "$WAIT_TIMEOUT"; then
        log_success "Mock Signal Server is ready"
    else
        log_error "Mock Signal Server failed to start within ${WAIT_TIMEOUT}s"
        docker-compose -f "$DOCKER_COMPOSE" logs moda-signal-mock-server
        exit 1
    fi

    echo ""
    log_success "✨ Emulator environment is ready!"
    echo ""
    echo "📱 Flutter App:           http://localhost:8080"
    echo "🔬 FastMODA API:          http://localhost:5000"
    echo "🎯 Mock Signal Server:    http://localhost:8081"
    echo ""
    log_info "View logs with: docker-compose logs -f [service-name]"
    echo ""
}

cmd_down() {
    log_info "Stopping containerized emulator environment..."
    cd "$REPO_ROOT"

    docker-compose -f "$DOCKER_COMPOSE" down fastmoda-api flutter-emulator moda-signal-mock-server

    log_success "Environment stopped"
}

cmd_test() {
    echo ""
    log_info "🧪 Running integration smoke tests..."
    echo ""

    # Check if Python test file exists
    if [ ! -f "$SCRIPT_DIR/emulator_smoke_tests.py" ]; then
        log_error "Test file not found: $SCRIPT_DIR/emulator_smoke_tests.py"
        exit 1
    fi

    # Run tests
    python3 "$SCRIPT_DIR/emulator_smoke_tests.py"

    if [ $? -eq 0 ]; then
        log_success "All smoke tests passed! 🎉"
    else
        log_error "Some tests failed"
        exit 1
    fi
}

# Run main
main "$@"
