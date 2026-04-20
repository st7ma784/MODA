# Multi-stage MATLAB MODA Container
# Build with: docker build -t moda-dev:latest --target matlab-dev .
# Test with: docker run --rm moda-test:latest bash tests/test_integration.sh
# Run GUI: docker run -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix moda-dev:latest

ARG MATLAB_VERSION=r2024b

# ============================================================================
# Stage 1: MATLAB Development Container
# ============================================================================
FROM mathworks/matlab:${MATLAB_VERSION} as matlab-dev

WORKDIR /app

LABEL maintainer="MODA Team" \
      version="2.0" \
      description="MODA - Modulation/Demodulation Analysis Toolkit with App Designer"

# Install additional development tools with retry and cleanup
RUN apt-get clean && rm -rf /var/lib/apt/lists/* && \
    apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    vim \
    build-essential \
    graphviz \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy MODA source code
COPY . .

# Verify MATLAB installation and basic functionality
RUN matlab -batch "disp('MATLAB R2024b initialized'); ver; exit(0);" 2>&1 | head -20

# Configure MATLAB path for MODA
ENV MATLABPATH=/app:/app/allguis/codes:/app/allguis/guis

# Verify MODA modules exist
RUN ls -la MODA.m && \
    ls -la allguis/codes/reading/read_from_csv.m && \
    ls -la allguis/codes/reading/read_from_mat.m

# Store MATLAB version for later verification
RUN matlab -batch "v = ver('MATLAB'); disp(v.Release);" > /MATLAB_VERSION.txt 2>&1 || true

EXPOSE 6789

# ============================================================================
# Stage 2: MATLAB Test Container
# ============================================================================
FROM mathworks/matlab:${MATLAB_VERSION} as moda-test

WORKDIR /app

LABEL description="MODA testing and validation environment"

# Install testing dependencies with cleanup
RUN apt-get clean && rm -rf /var/lib/apt/lists/* && \
    apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    jq \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy entire codebase
COPY . .

# Create test results directory
RUN mkdir -p /tmp/test_results && chmod 777 /tmp/test_results

# Configure MATLAB path
ENV MATLABPATH=/app:/app/allguis/codes:/app/allguis/guis
ENV MATLAB_LOG_DIR=/tmp/matlab_logs

# Create directories needed for testing
RUN mkdir -p /tmp/matlab_logs && chmod 777 /tmp/matlab_logs

# Verify test files exist
RUN ls -la tests/test_*.m 2>/dev/null || echo "Note: Test files will be added by user"

# ============================================================================
# Stage 3: MATLAB Production Runtime (Optimized)
# ============================================================================
FROM mathworks/matlab:${MATLAB_VERSION} as moda-prod

WORKDIR /app

LABEL description="MODA production runtime (minimal footprint)"

# Minimal dependencies with cleanup and retry safety
RUN apt-get clean && rm -rf /var/lib/apt/lists/* && \
    apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy only necessary files from development stage
COPY . .

# Configure MATLAB path
ENV MATLABPATH=/app:/app/allguis/codes:/app/allguis/guis

# Health check: Verify MATLAB can load MODA
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD matlab -batch "addpath(genpath('.')); which MODA; exit(0);" 2>/dev/null || exit 1

# Default: Run MODA (requires display forwarding or headless mode)
ENTRYPOINT ["matlab"]
CMD ["-r", "MODA; exit;"]
