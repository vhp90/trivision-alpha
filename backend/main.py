from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .service import ENGINE, OUTPUT_ROOT


app = FastAPI(title="Trivision Alpha", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/status")
def api_status() -> dict:
    return ENGINE.status()


@app.post("/api/load")
def api_load(payload: dict) -> dict:
    model_id = payload.get("model_id")
    if not model_id:
        raise HTTPException(status_code=400, detail="model_id is required")
    return ENGINE.load_model(
        model_id,
        keep_models_on_gpu=bool(payload.get("keep_models_on_gpu", True)),
        device=payload.get("device"),
    )


@app.post("/api/generate")
async def api_generate(
    image: UploadFile = File(...),
    settings: str = Form("{}"),
) -> JSONResponse:
    try:
        payload = json.loads(settings)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid settings JSON: {exc}") from exc

    record = ENGINE.submit(await image.read(), payload)
    return JSONResponse(record.as_dict())


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str) -> dict:
    record = ENGINE.jobs.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return record.as_dict()


@app.get("/api/jobs/{job_id}/artifacts/{name}")
def api_artifact(job_id: str, name: str) -> FileResponse:
    artifact_path = OUTPUT_ROOT / job_id / name
    if not artifact_path.exists():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(artifact_path)


frontend_dist = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
