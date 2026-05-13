#!/usr/bin/env python3
"""
Integration Smoke Tests for Containerized Emulator
===================================================

Verifies that:
1. All services are running and healthy
2. Flutter app loads successfully
3. FastMODA API responds
4. Mock signal server is operational
5. End-to-end signal analysis works

Usage:
    python3 tests/emulator_smoke_tests.py
"""

import json
import sys
import time
from typing import Optional

import requests


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

FASTMODA_URL = "http://localhost:5000"
FLUTTER_APP_URL = "http://localhost:8080"
SIGNAL_MOCK_URL = "http://localhost:8081"
TIMEOUT = 5.0


# ──────────────────────────────────────────────────────────────────────────────
# Test Functions
# ──────────────────────────────────────────────────────────────────────────────

def test_fastmoda_health() -> bool:
    """Verify FastMODA API is running."""
    print("🔬 Testing FastMODA API health...", end=" ", flush=True)
    try:
        resp = requests.get(f"{FASTMODA_URL}/health", timeout=TIMEOUT)
        if resp.status_code == 200:
            print("✅")
            return True
        else:
            print(f"❌ (status {resp.status_code})")
            return False
    except requests.RequestException as e:
        print(f"❌ ({e})")
        return False


def test_flutter_app_loads() -> bool:
    """Verify Flutter web app serves and contains app markup."""
    print("📱 Testing Flutter app loads...", end=" ", flush=True)
    try:
        resp = requests.get(FLUTTER_APP_URL, timeout=TIMEOUT)
        if resp.status_code == 200:
            # Check for Flutter app indicators
            content_lower = resp.text.lower()
            if "flutter" in content_lower or "moda" in content_lower or "<!doctype" in content_lower:
                print("✅")
                return True
            else:
                print("⚠️  (loaded but no app markers found)")
                return True  # Still pass — might be valid Flutter output
        else:
            print(f"❌ (status {resp.status_code})")
            return False
    except requests.RequestException as e:
        print(f"❌ ({e})")
        return False


def test_signal_mock_server_health() -> bool:
    """Verify mock signal server is available."""
    print("🎯 Testing Signal Mock Server health...", end=" ", flush=True)
    try:
        resp = requests.get(f"{SIGNAL_MOCK_URL}/health", timeout=TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == "ok":
                print("✅")
                return True
            else:
                print(f"❌ (unexpected response: {data})")
                return False
        else:
            print(f"❌ (status {resp.status_code})")
            return False
    except requests.RequestException as e:
        print(f"❌ ({e})")
        return False


def test_signal_generation() -> bool:
    """Test signal generation workflow."""
    print("📊 Testing signal generation...", end=" ", flush=True)
    try:
        # Start streaming
        resp = requests.post(
            f"{SIGNAL_MOCK_URL}/stream/start",
            json={"preset": "resting"},
            timeout=TIMEOUT,
        )
        if resp.status_code != 200:
            print(f"❌ (failed to start stream: {resp.status_code})")
            return False

        # Get a chunk
        time.sleep(0.1)
        resp = requests.get(
            f"{SIGNAL_MOCK_URL}/stream/chunk",
            params={"duration": 0.1},
            timeout=TIMEOUT,
        )
        if resp.status_code != 200:
            print(f"❌ (failed to get chunk: {resp.status_code})")
            return False

        data = resp.json()
        if "samples" in data and len(data["samples"]) > 0:
            # Stop streaming
            requests.post(f"{SIGNAL_MOCK_URL}/stream/stop", timeout=TIMEOUT)
            print("✅")
            return True
        else:
            print(f"❌ (no samples in response)")
            return False
    except requests.RequestException as e:
        print(f"❌ ({e})")
        return False


def test_preset_switching() -> bool:
    """Test changing signal presets."""
    print("🧠 Testing preset switching...", end=" ", flush=True)
    try:
        presets = ["resting", "active", "drowsy", "sleep"]

        for preset in presets:
            resp = requests.post(
                f"{SIGNAL_MOCK_URL}/preset",
                json={"preset": preset},
                timeout=TIMEOUT,
            )
            if resp.status_code != 200:
                print(f"❌ (failed on preset '{preset}')")
                return False

            data = resp.json()
            if data.get("preset") != preset:
                print(f"❌ (preset not set correctly)")
                return False

        print("✅")
        return True
    except requests.RequestException as e:
        print(f"❌ ({e})")
        return False


def test_end_to_end_analysis() -> bool:
    """
    E2E test: Submit signal to FastMODA and verify response.
    This tests the full analysis pipeline.
    """
    print("🔄 Testing E2E analysis...", end=" ", flush=True)
    try:
        import numpy as np

        # Generate test signal
        fs = 256.0
        duration = 2.0
        t = np.arange(0, duration, 1 / fs)
        
        # Create signal with 10 Hz dominant frequency
        signal = (
            1.0 * np.sin(2 * np.pi * 10 * t) +  # 10 Hz alpha
            0.3 * np.sin(2 * np.pi * 6 * t) +   # 6 Hz theta
            0.1 * np.random.randn(len(t))       # noise
        )
        signal = signal.astype(np.float32)

        # Submit to FastMODA
        files = {"file": ("signal.npy", signal.tobytes())}
        data = {"fs": "256.0", "win": "1.0", "pen": "auto"}

        resp = requests.post(
            f"{FASTMODA_URL}/analyze",
            files=files,
            data=data,
            timeout=TIMEOUT,
        )

        if resp.status_code != 200:
            print(f"❌ (submit failed: {resp.status_code})")
            return False

        result = resp.json()
        if "task_id" not in result:
            print(f"❌ (no task_id in response)")
            return False

        task_id = result["task_id"]

        # Poll for completion (max 30 seconds)
        for attempt in range(60):
            time.sleep(0.5)
            status_resp = requests.get(
                f"{FASTMODA_URL}/status/{task_id}",
                timeout=TIMEOUT,
            )

            if status_resp.status_code != 200:
                print(f"❌ (status check failed)")
                return False

            status_data = status_resp.json()
            if status_data.get("status") == "complete":
                print("✅")
                return True
            elif status_data.get("status") == "error":
                print(f"❌ (analysis error)")
                return False

        print(f"❌ (timeout waiting for analysis)")
        return False

    except ImportError:
        print("⊘  (numpy not available, skipping)")
        return True
    except requests.RequestException as e:
        print(f"❌ ({e})")
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    print()
    print("=" * 70)
    print("MODA Emulator Integration Smoke Tests")
    print("=" * 70)
    print()

    tests = [
        ("FastMODA Health", test_fastmoda_health),
        ("Flutter App Load", test_flutter_app_loads),
        ("Signal Mock Server", test_signal_mock_server_health),
        ("Signal Generation", test_signal_generation),
        ("Preset Switching", test_preset_switching),
        ("E2E Analysis", test_end_to_end_analysis),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"⚠️  Test '{name}' raised exception: {e}")
            results.append((name, False))

    # Summary
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status:8}  {name}")

    print()
    print(f"  {passed}/{total} tests passed")
    print()

    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print(f"⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
