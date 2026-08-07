#!/usr/bin/env bash
# End-to-end melanoma MLflow pipeline: extract FastMODA features from the
# laser-Doppler recordings, then train + log three linear models.
#
#   bash ml/run_ml.sh /path/to/melanomadata
#
# Arg 1 (optional): directory containing data.mat / info2.xlsx.
#   Default: C:/Users/st7ma/Downloads/melanomadata/melanomadata
#
# Outputs land in ml/artifacts/ :  features.npz/.csv  and  mlruns/
# Browse afterwards:  mlflow ui --backend-store-uri ml/artifacts/mlruns
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${1:-C:/Users/st7ma/Downloads/melanomadata/melanomadata}"
IMAGE="${FASTMODA_IMAGE:-fastmoda:cpu}"
mkdir -p "${REPO}/ml/artifacts"

docker run --rm \
  -v "${DATA_DIR}:/data:ro" \
  -v "${REPO}/ml:/ml:ro" \
  -v "${REPO}/ml/artifacts:/artifacts" \
  -e MLFLOW_ALLOW_FILE_STORE=true -e GIT_PYTHON_REFRESH=quiet \
  "${IMAGE}" bash -c '
    export PYTHONPATH=/app
    pip install -q mlflow >/dev/null 2>&1
    # 1) FastMODA feature extraction (cent + norm + contrast; fs=40 Hz)
    python /ml/build_features.py --fs 40
    # 2) Train + log three linear targets. MLflow file store is written to the
    #    container fs (native ext4 avoids Windows-bind-mount quirks) then copied.
    python /ml/train_mlflow.py --tracking file:///tmp/mlruns
    rm -rf /artifacts/mlruns && cp -r /tmp/mlruns /artifacts/mlruns
    echo "artifacts: $(ls /artifacts)"
  '
