from __future__ import annotations

import importlib
import sys


CHECKS = {
    "torch": "PyTorch core",
    "torchvision": "Torchvision",
    "transformers": "Transformers",
    "nvdiffrast.torch": "nvdiffrast CUDA rasterizer",
    "cumesh": "CuMesh",
    "flex_gemm": "FlexGEMM",
    "o_voxel": "O-Voxel",
}


def main() -> int:
    failures: list[str] = []

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
