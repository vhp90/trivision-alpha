#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! python -c "import torch" >/dev/null 2>&1; then
  python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
fi

python -m pip install -r backend/requirements.txt
python -m pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

if ! python -c "import flash_attn" >/dev/null 2>&1; then
  python -m pip install flash-attn==2.7.3
fi

mkdir -p /tmp/trivision-alpha

if ! python -c "import nvdiffrast.torch" >/dev/null 2>&1; then
  if [ ! -d /tmp/trivision-alpha/nvdiffrast ]; then
    git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/trivision-alpha/nvdiffrast
  fi
  python -m pip install /tmp/trivision-alpha/nvdiffrast --no-build-isolation
fi

if ! python -c "import cumesh" >/dev/null 2>&1; then
  if [ ! -d /tmp/trivision-alpha/CuMesh ]; then
    git clone https://github.com/JeffreyXiang/CuMesh.git /tmp/trivision-alpha/CuMesh --recursive
  fi
  python -m pip install /tmp/trivision-alpha/CuMesh --no-build-isolation
fi

if ! python -c "import flex_gemm" >/dev/null 2>&1; then
  if [ ! -d /tmp/trivision-alpha/FlexGEMM ]; then
    git clone https://github.com/JeffreyXiang/FlexGEMM.git /tmp/trivision-alpha/FlexGEMM --recursive
  fi
  python -m pip install /tmp/trivision-alpha/FlexGEMM --no-build-isolation
fi

if ! python -c "import o_voxel" >/dev/null 2>&1; then
  python -m pip install ./o-voxel --no-build-isolation
fi

cd frontend
npm install

cd "$ROOT"
python scripts/preflight_runtime.py

echo
echo "Runtime dependencies installed."
echo "Backend: uvicorn backend.main:app --reload --app-dir ."
echo "Frontend: cd frontend && npm run dev"
