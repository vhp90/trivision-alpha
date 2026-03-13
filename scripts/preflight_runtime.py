from __future__ import annotations

import importlib
import sys
from importlib import metadata


CHECKS = {
    "torch": "PyTorch core",
    "torchvision": "Torchvision",
    "transformers": "Transformers",
    "xformers": "xFormers attention backend",
    "spconv.pytorch": "spconv sparse convolution backend",
    "nvdiffrast.torch": "nvdiffrast CUDA rasterizer",
    "cumesh": "CuMesh",
    "o_voxel": "O-Voxel",
}

REQUIRED_PYTHON = (3, 10)
REQUIRED_TORCH = "2.6.0"
REQUIRED_TORCHAO = "0.16.0"


def main() -> int:
    failures: list[str] = []

    if sys.version_info[:2] != REQUIRED_PYTHON:
        current = f"{sys.version_info.major}.{sys.version_info.minor}"
        required = ".".join(str(part) for part in REQUIRED_PYTHON)
        print(f"[FAIL] Python runtime -> found {current}, expected {required}")
        failures.append("Python version")

    try:
        import torch

        if not torch.__version__.startswith(REQUIRED_TORCH):
            print(f"[FAIL] PyTorch version -> found {torch.__version__}, expected {REQUIRED_TORCH}.x")
            failures.append("PyTorch version")
        else:
            print(f"[OK] PyTorch version -> {torch.__version__}")
    except Exception as exc:
        print(f"[FAIL] PyTorch version -> {exc}")
        failures.append("PyTorch version")

    try:
        torchao_version = metadata.version("torchao")
        if torchao_version != REQUIRED_TORCHAO:
            print(f"[FAIL] TorchAO version -> found {torchao_version}, expected {REQUIRED_TORCHAO}")
            failures.append("TorchAO version")
        else:
            print(f"[OK] TorchAO version -> {torchao_version}")
    except Exception as exc:
        print(f"[FAIL] TorchAO version -> {exc}")
        failures.append("TorchAO version")

    for module_name, label in CHECKS.items():
        try:
            importlib.import_module(module_name)
            print(f"[OK] {label}: {module_name}")
        except Exception as exc:
            print(f"[FAIL] {label}: {module_name} -> {exc}")
            failures.append(label)

    try:
        from transformers import DINOv3ViTModel  # noqa: F401

        print("[OK] Transformers DINOv3ViTModel import")
    except Exception as exc:
        print(f"[FAIL] Transformers DINOv3ViTModel import -> {exc}")
        failures.append("DINOv3ViTModel")

    if failures:
        print()
        print("Preflight failed. Fix the missing imports above before running the app.")
        return 1

    print()
    print("Preflight passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
