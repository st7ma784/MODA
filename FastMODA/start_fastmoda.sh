#!/bin/bash
# FastMODA Quick Start Script
# Usage: ./start_fastmoda.sh

echo "🔬 Starting FastMODA Web Server..."
echo ""
echo "Prerequisites:"
echo "  - open-ce conda environment (already configured)"
echo "  - Required packages: flask, plotly, ruptures (already installed)"
echo ""
echo "Starting server on http://127.0.0.1:5000"
echo "Press Ctrl+C to stop"
echo ""
echo "-------------------------------------------------------------------"

cd /data/MODA
conda run -n open-ce python FastMODA/app.py
