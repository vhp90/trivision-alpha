from __future__ import annotations

import gc
import json
import os
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MODEL_ID = os.getenv("TRIVISION_MODEL_ID", "ahp93/TRELLIS.2-4B-INT8")
OUTPUT_ROOT = ROOT / "backend" / "generated"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def _configure_cuda() -> None:
    os.environ.setdefault("ATTN_BACKEND", "xformers")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers")
    os.environ.setdefault("SPARSE_CONV_BACKEND", "spconv")
    os.environ.setdefault("SPCONV_ALGO", "implicit_gemm")

    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _seed_from_settings(settings: dict[str, Any]) -> int:
    if settings.get("randomize_seed", True):
        return int(np.random.randint(0, 2**31 - 1))
    return int(settings["seed"])


def _merge_sampler_settings(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    merged.update(overrides)
    return merged


def _make_preview_sheet(frames: list[np.ndarray]) -> Image.Image:
    tiles = [Image.fromarray(frame) for frame in frames]
    canvas = Image.new("RGB", (sum(tile.width for tile in tiles), max(tile.height for tile in tiles)))
    cursor = 0
    for tile in tiles:
        canvas.paste(tile, (cursor, 0))
        cursor += tile.width
    return canvas


def default_settings() -> dict[str, Any]:
    return {
        "model_id": DEFAULT_MODEL_ID,
        "device": "cuda",
        "keep_models_on_gpu": True,
        "pipeline_type": "1024_cascade",
        "num_samples": 1,
        "seed": 0,
        "randomize_seed": True,
        "preprocess_image": True,
        "max_num_tokens": 49152,
        "decimation_target": 500000,
        "texture_size": 2048,
        "preview_resolution": 768,
        "preview_views": 6,
        "sparse_structure": {
            "steps": 12,
            "guidance_strength": 7.5,
            "guidance_rescale": 0.7,
            "rescale_t": 5.0,
            "overrides": {},
        },
        "shape_slat": {
            "steps": 12,
            "guidance_strength": 7.5,
            "guidance_rescale": 0.5,
            "rescale_t": 3.0,
            "overrides": {},
        },
        "tex_slat": {
            "steps": 12,
            "guidance_strength": 1.0,
            "guidance_rescale": 0.0,
            "rescale_t": 3.0,
            "overrides": {},
        },
    }


def normalize_settings(payload: dict[str, Any]) -> dict[str, Any]:
    settings = default_settings()
    for key, value in payload.items():
        if isinstance(value, dict) and isinstance(settings.get(key), dict):
            settings[key].update(value)
        else:
            settings[key] = value
    return settings


@dataclass
class JobRecord:
    id: str
    created_at: str
    updated_at: str
    status: str = "queued"
    stage: str = "pending"
    message: str = "Queued"
    error: str | None = None
    seed: int | None = None
    settings: dict[str, Any] = field(default_factory=dict)
    samples: list[dict[str, str]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "stage": self.stage,
            "message": self.message,
            "error": self.error,
            "seed": self.seed,
            "settings": self.settings,
            "samples": self.samples,
        }


class TrivisionEngine:
    def __init__(self) -> None:
        _configure_cuda()
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._pipeline: Any | None = None
        self._model_id: str | None = None
        self._keep_models_on_gpu = True
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._jobs: dict[str, JobRecord] = {}

    @property
    def jobs(self) -> dict[str, JobRecord]:
        return self._jobs

    def _set_job(self, job_id: str, **updates: Any) -> None:
        with self._lock:
            job = self._jobs[job_id]
            for key, value in updates.items():
                setattr(job, key, value)
            job.updated_at = _now()

    def gpu_stats(self) -> dict[str, Any]:
        if not torch.cuda.is_available():
            return {"available": False}
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        props = torch.cuda.get_device_properties(0)
        return {
            "available": True,
            "name": props.name,
            "total_vram_gb": round(total_bytes / 1024**3, 2),
            "free_vram_gb": round(free_bytes / 1024**3, 2),
            "allocated_vram_gb": round(torch.cuda.memory_allocated(0) / 1024**3, 2),
            "reserved_vram_gb": round(torch.cuda.memory_reserved(0) / 1024**3, 2),
        }

    def status(self) -> dict[str, Any]:
        return {
            "model_loaded": self._pipeline is not None,
            "model_id": self._model_id,
            "keep_models_on_gpu": self._keep_models_on_gpu,
            "device": self._device,
            "gpu": self.gpu_stats(),
            "defaults": default_settings(),
        }

    def _get_pipeline_class(self):
        from trellis2.pipelines import Trellis2ImageTo3DPipeline

        return Trellis2ImageTo3DPipeline

    def _resolve_model_source(self, model_id: str) -> str:
        if os.path.isdir(model_id):
            return model_id

        from trellis2.hf import snapshot_download_with_auth

        return snapshot_download_with_auth(model_id)

    def _get_render_utils(self):
        from trellis2.utils import render_utils

        return render_utils

    def _get_o_voxel(self):
        import o_voxel

        return o_voxel

    def unload(self) -> None:
        with self._lock:
            self._pipeline = None
            self._model_id = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_model(self, model_id: str, keep_models_on_gpu: bool = True, device: str | None = None) -> dict[str, Any]:
        target_device = device or self._device
        if target_device != "cuda":
            raise RuntimeError("Trivision Alpha is currently CUDA-only. CPU mode is not supported.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available on this machine.")

        if self._pipeline is not None and self._model_id == model_id and self._keep_models_on_gpu == keep_models_on_gpu:
            return self.status()

        self.unload()

        pipeline_cls = self._get_pipeline_class()
        source = self._resolve_model_source(model_id)
        pipeline = pipeline_cls.from_pretrained(source)
        pipeline.low_vram = not keep_models_on_gpu
        pipeline._device = torch.device(target_device)

        if keep_models_on_gpu and target_device == "cuda":
            pipeline.cuda()

        self._pipeline = pipeline
        self._model_id = model_id
        self._keep_models_on_gpu = keep_models_on_gpu
        self._device = target_device
        return self.status()

    def submit(self, image_bytes: bytes, payload: dict[str, Any]) -> JobRecord:
        settings = normalize_settings(payload)
        self.load_model(
            settings["model_id"],
            keep_models_on_gpu=bool(settings.get("keep_models_on_gpu", True)),
            device=settings.get("device"),
        )

        job_id = uuid.uuid4().hex
        record = JobRecord(
            id=job_id,
            created_at=_now(),
            updated_at=_now(),
            settings=settings,
        )
        with self._lock:
            self._jobs[job_id] = record
        self._executor.submit(self._run_job, job_id, image_bytes)
        return record

    def _run_job(self, job_id: str, image_bytes: bytes) -> None:
        try:
            if self._pipeline is None:
                raise RuntimeError("Pipeline is not loaded.")

            settings = self._jobs[job_id].settings
            seed = _seed_from_settings(settings)
            self._set_job(job_id, status="running", stage="preprocess", message="Preparing image and sampler inputs", seed=seed)

            image = Image.open(BytesIO(image_bytes)).convert("RGBA")

            sparse_structure_params = _merge_sampler_settings(
                {
                    "steps": settings["sparse_structure"]["steps"],
                    "guidance_strength": settings["sparse_structure"]["guidance_strength"],
                    "guidance_rescale": settings["sparse_structure"]["guidance_rescale"],
                    "rescale_t": settings["sparse_structure"]["rescale_t"],
                },
                settings["sparse_structure"].get("overrides", {}),
            )
            shape_slat_params = _merge_sampler_settings(
                {
                    "steps": settings["shape_slat"]["steps"],
                    "guidance_strength": settings["shape_slat"]["guidance_strength"],
                    "guidance_rescale": settings["shape_slat"]["guidance_rescale"],
                    "rescale_t": settings["shape_slat"]["rescale_t"],
                },
                settings["shape_slat"].get("overrides", {}),
            )
            tex_slat_params = _merge_sampler_settings(
                {
                    "steps": settings["tex_slat"]["steps"],
                    "guidance_strength": settings["tex_slat"]["guidance_strength"],
                    "guidance_rescale": settings["tex_slat"]["guidance_rescale"],
                    "rescale_t": settings["tex_slat"]["rescale_t"],
                },
                settings["tex_slat"].get("overrides", {}),
            )

            self._set_job(job_id, stage="generate", message="Running quantized TRELLIS.2 on GPU")
            outputs = self._pipeline.run(
                image,
                num_samples=int(settings["num_samples"]),
                seed=seed,
                sparse_structure_sampler_params=sparse_structure_params,
                shape_slat_sampler_params=shape_slat_params,
                tex_slat_sampler_params=tex_slat_params,
                preprocess_image=bool(settings["preprocess_image"]),
                pipeline_type=settings["pipeline_type"],
                max_num_tokens=int(settings["max_num_tokens"]),
            )

            self._set_job(job_id, stage="export", message="Rendering previews and packaging GLB samples")
            job_root = OUTPUT_ROOT / job_id
            job_root.mkdir(parents=True, exist_ok=True)

            samples: list[dict[str, str]] = []
            for index, mesh in enumerate(outputs, start=1):
                mesh.simplify(16777216)
                render_utils = self._get_render_utils()
                previews = render_utils.render_snapshot(
                    mesh,
                    resolution=int(settings["preview_resolution"]),
                    r=2,
                    fov=36,
                    nviews=int(settings["preview_views"]),
                    verbose=False,
                )
                preview_sheet = _make_preview_sheet(previews["color"])
                preview_name = f"sample_{index:02d}_preview.png"
                preview_path = job_root / preview_name
                preview_sheet.save(preview_path)

                o_voxel = self._get_o_voxel()
                glb = o_voxel.postprocess.to_glb(
                    vertices=mesh.vertices,
                    faces=mesh.faces,
                    attr_volume=mesh.attrs,
                    coords=mesh.coords,
                    attr_layout=self._pipeline.pbr_attr_layout,
                    grid_size=int(round(1 / mesh.voxel_size)),
                    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                    decimation_target=int(settings["decimation_target"]),
                    texture_size=int(settings["texture_size"]),
                    remesh=True,
                    remesh_band=1,
                    remesh_project=0,
                    use_tqdm=True,
                )
                glb_name = f"sample_{index:02d}.glb"
                glb_path = job_root / glb_name
                glb.export(glb_path, extension_webp=True)

                samples.append(
                    {
                        "label": f"Sample {index}",
                        "preview_url": f"/api/jobs/{job_id}/artifacts/{preview_name}",
                        "glb_url": f"/api/jobs/{job_id}/artifacts/{glb_name}",
                        "glb_name": glb_name,
                    }
                )

            metadata_path = job_root / "metadata.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "job_id": job_id,
                        "seed": seed,
                        "settings": settings,
                        "completed_at": _now(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            self._set_job(job_id, status="completed", stage="done", message="Generation finished", samples=samples)
        except Exception as exc:
            self._set_job(job_id, status="failed", stage="error", message="Generation failed", error=str(exc))
        finally:
            gc.collect()
            if torch.cuda.is_available() and not self._keep_models_on_gpu:
                torch.cuda.empty_cache()


ENGINE = TrivisionEngine()
