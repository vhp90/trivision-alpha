from __future__ import annotations

import os
from typing import Any, Iterable, Optional

import torch
import torch.nn as nn

from .modules.sparse.linear import SparseLinear


INT8_CHECKPOINT_SUFFIX = ".int8.pt"
INT8_FORMAT = "trellis2-selective-int8-weight-only"
INT8_FORMAT_VERSION = 1
ATTN_LINEAR_NAMES = {"to_qkv", "to_q", "to_kv", "to_out"}


def _import_torchao():
    try:
        import torchao
        from torchao.quantization import Int8WeightOnlyConfig, quantize_
    except ImportError as exc:
        raise ImportError(
            "Selective INT8 export/load requires torchao. Install it with `pip install torchao`."
        ) from exc
    return torchao, Int8WeightOnlyConfig, quantize_


def is_int8_checkpoint(path: str) -> bool:
    return os.path.exists(f"{path}.json") and os.path.exists(f"{path}{INT8_CHECKPOINT_SUFFIX}")


def get_selective_int8_module_names(model: nn.Module) -> list[str]:
    target_names: list[str] = []
    for name, module in model.named_modules():
        if not name.startswith("blocks."):
            continue
        if not isinstance(module, (nn.Linear, SparseLinear)):
            continue
        leaf_name = name.rsplit(".", 1)[-1]
        if leaf_name in ATTN_LINEAR_NAMES or ".mlp.mlp." in name:
            target_names.append(name)
    return sorted(set(target_names))


def _filter_named_linear_modules(target_names: set[str]):
    return lambda module, fqn: fqn in target_names and isinstance(module, (nn.Linear, SparseLinear))


def quantize_model_int8_inplace(model: nn.Module, module_names: Optional[Iterable[str]] = None) -> list[str]:
    _, Int8WeightOnlyConfig, quantize_ = _import_torchao()
    target_names = sorted(set(module_names or get_selective_int8_module_names(model)))
    if not target_names:
        return []
    quantize_(
        model,
        Int8WeightOnlyConfig(version=2, set_inductor_config=False),
        filter_fn=_filter_named_linear_modules(set(target_names)),
    )
    return target_names


def load_int8_checkpoint_package(
    model: nn.Module,
    checkpoint_path: str,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    package = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if not isinstance(package, dict) or package.get("format") != INT8_FORMAT:
        raise ValueError(f"Unsupported quantized checkpoint format in {checkpoint_path}")
    quantized_fqns = package.get("quantized_fqns", [])
    quantize_model_int8_inplace(model, quantized_fqns)
    model.load_state_dict(package["state_dict"], strict=False)
    return package
