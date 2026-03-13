# Trivision Alpha

Trivision Alpha is a studio-style frontend for running the quantized TRELLIS.2 bundle with full parameter control and a no-offload CUDA-first runtime.

This app is intentionally **CUDA-only**. CPU mode is not exposed because the TRELLIS runtime still contains direct CUDA paths in key rendering and feature-extraction components.

## What This App Does

- uses the official TRELLIS.2 codebase as its runtime base
- adds remote `.int8.pt` checkpoint loading for the selective INT8 bundle
- defaults to keeping models resident on GPU instead of hard-coded low-VRAM offloading
- exposes TRELLIS generation controls in a dedicated frontend
- exports downloadable textured GLB files from the generated `MeshWithVoxel` output

## Layout

- `trellis2/`: official TRELLIS.2 runtime with minimal quantized-model and full-GPU patches
- `o-voxel/`: official O-Voxel dependency from the upstream repo
- `backend/`: FastAPI inference server and generation job runner
- `frontend/`: Vite + React studio UI
- `scripts/setup_runtime.sh`: CUDA/runtime dependency bootstrap

## Setup

```bash
bash scripts/setup_runtime.sh
uvicorn backend.main:app --reload --app-dir .
cd frontend && npm run dev
```

## Recommended Environment

For Lightning Studios, the safest fast-start baseline for this app is:

- Python `3.10`
- PyTorch `2.6.0`
- Torchvision `0.21.0`
- CUDA `12.4`

Why this combination:

- it matches the upstream TRELLIS setup direction
- `xformers` has Linux wheels for Python 3.10 on this stack
- `spconv-cu124` has Linux wheels for Python 3.10 on this stack
- `flash-attn` on PyPI is source-only, so it is no longer the default backend here
- `transformers==4.56.2` is new enough for `DINOv3ViTModel`

Before starting a paid GPU session, you can also run:

```bash
python scripts/preflight_runtime.py
```

That checks the exact extension imports and the `transformers` DINOv3 class import before you try loading the model.

## Hugging Face Auth

This app expects a `.env` file at the repo root when you are using gated or rate-limited model repos.

Supported variables:

- `HF_TOKEN`
- `HUGGINGFACE_TOKEN`
- `HUGGINGFACE_HUB_TOKEN`
- `HF_HOME`
- `TRANSFORMERS_CACHE`

You can start from:

- `.env.example`

The runtime loads `.env` automatically before any TRELLIS, DINO, or BiRefNet Hugging Face downloads happen.

## Download Behavior

- Hugging Face downloads use the token loaded from `.env`
- `HF_HUB_ENABLE_HF_TRANSFER=1` is enabled automatically for faster downloads when `hf_transfer` is installed
- the quantized TRELLIS repo is snapshot-downloaded into the Hugging Face cache before model initialization, so first-load fetches happen in parallel instead of file-by-file
- pipeline configs, quantized `.int8.pt` weights, DINO, and BiRefNet all share the same auth/bootstrap path

## Build-Time Strategy

This repo now prefers prebuilt wheels where they actually exist:

- attention backend: `xformers`
- sparse conv backend: `spconv-cu124`

Packages still likely to build from source on a fresh machine:

- `nvdiffrast`
- `CuMesh`
- `o-voxel`

`FlexGEMM` and `flash-attn` are no longer part of the default install path.

## Default Quantized Model

The app defaults to `ahp93/TRELLIS.2-4B-INT8`, but you can point it to any compatible local path or Hugging Face repo id from the UI.
