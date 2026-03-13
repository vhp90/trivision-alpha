import { useEffect, useMemo, useState } from "react";
import {
  Cpu,
  Cuboid,
  Gauge,
  ImagePlus,
  LoaderCircle,
  Rocket,
  Settings2,
  Sparkles,
  Zap,
} from "lucide-react";

import { createJob, fetchJob, fetchStatus, loadModel } from "./api";
import type { JobSample, SamplerPanel, StudioJob, StudioStatus } from "./types";

const DEFAULT_SAMPLER: SamplerPanel = {
  steps: 12,
  guidance_strength: 7.5,
  guidance_rescale: 0.5,
  rescale_t: 3,
  overrides: "{}",
};

function samplerPayload(panel: SamplerPanel) {
  return {
    steps: panel.steps,
    guidance_strength: panel.guidance_strength,
    guidance_rescale: panel.guidance_rescale,
    rescale_t: panel.rescale_t,
    overrides: JSON.parse(panel.overrides || "{}"),
  };
}

export default function App() {
  const [status, setStatus] = useState<StudioStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [job, setJob] = useState<StudioJob | null>(null);
  const [selectedSample, setSelectedSample] = useState<JobSample | null>(null);

  const [modelId, setModelId] = useState("ahp93/TRELLIS.2-4B-INT8");
  const [device] = useState("cuda");
  const [keepModelsOnGpu, setKeepModelsOnGpu] = useState(true);
  const [pipelineType, setPipelineType] = useState("1024_cascade");
  const [numSamples, setNumSamples] = useState(1);
  const [seed, setSeed] = useState(0);
  const [randomizeSeed, setRandomizeSeed] = useState(true);
  const [preprocessImage, setPreprocessImage] = useState(true);
  const [maxNumTokens, setMaxNumTokens] = useState(49152);
  const [decimationTarget, setDecimationTarget] = useState(500000);
  const [textureSize, setTextureSize] = useState(2048);
  const [previewResolution, setPreviewResolution] = useState(768);
  const [previewViews, setPreviewViews] = useState(6);
  const [sparseStructure, setSparseStructure] = useState<SamplerPanel>({
    ...DEFAULT_SAMPLER,
    guidance_rescale: 0.7,
    rescale_t: 5,
  });
  const [shapeSlat, setShapeSlat] = useState<SamplerPanel>(DEFAULT_SAMPLER);
  const [texSlat, setTexSlat] = useState<SamplerPanel>({
    ...DEFAULT_SAMPLER,
    guidance_strength: 1,
    guidance_rescale: 0,
  });

  useEffect(() => {
    void refreshStatus();
  }, []);

  useEffect(() => {
    if (!job || job.status === "completed" || job.status === "failed") {
      return;
    }
    const timer = window.setInterval(async () => {
      const latest = await fetchJob(job.id);
      setJob(latest);
      if (latest.samples.length > 0) {
        setSelectedSample((current) => current ?? latest.samples[0]);
      }
    }, 2000);
    return () => window.clearInterval(timer);
  }, [job]);

  async function refreshStatus() {
    try {
      const next = await fetchStatus();
      setStatus(next);
      if (next.model_id) {
        setModelId(next.model_id);
        setKeepModelsOnGpu(next.keep_models_on_gpu);
        
      }
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Unable to reach the backend.");
    }
  }

  async function handleLoadModel() {
    setBusy(true);
    try {
      setNotice(null);
      const next = await loadModel({
        model_id: modelId,
        keep_models_on_gpu: keepModelsOnGpu,
        device,
      });
      setStatus(next);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Model load failed.");
    } finally {
      setBusy(false);
    }
  }

  async function handleGenerate() {
    if (!imageFile) {
      return;
    }
    setBusy(true);
    try {
      setNotice(null);
      const nextJob = await createJob(imageFile, {
        model_id: modelId,
        device,
        keep_models_on_gpu: keepModelsOnGpu,
        pipeline_type: pipelineType,
        num_samples: numSamples,
        seed,
        randomize_seed: randomizeSeed,
        preprocess_image: preprocessImage,
        max_num_tokens: maxNumTokens,
        decimation_target: decimationTarget,
        texture_size: textureSize,
        preview_resolution: previewResolution,
        preview_views: previewViews,
        sparse_structure: samplerPayload(sparseStructure),
        shape_slat: samplerPayload(shapeSlat),
        tex_slat: samplerPayload(texSlat),
      });
      setJob(nextJob);
      setSelectedSample(null);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Generation request failed.");
    } finally {
      setBusy(false);
    }
  }

  function updatePreview(file: File | null) {
    setImageFile(file);
    setSelectedSample(null);
    setJob(null);
    if (!file) {
      setImagePreview(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setImagePreview(url);
  }

  const gpuLabel = useMemo(() => {
    if (!status?.gpu.available) {
      return "CPU mode";
    }
    return `${status.gpu.name} · ${status.gpu.free_vram_gb} GB free / ${status.gpu.total_vram_gb} GB total`;
  }, [status]);

  return (
    <div className="shell">
      <aside className="control-column">
        <div className="brand-card">
          <div className="eyebrow">Trivision Alpha</div>
          <h1>Quantized TRELLIS Studio</h1>
          <p>
            Full-GPU image-to-3D generation tuned for the INT8 bundle. No forced offload, full
            parameter control, and export-ready GLB output.
          </p>
        </div>

        <section className="panel">
          <div className="panel-header">
            <Gauge size={18} />
            <h2>Runtime</h2>
          </div>
          <div className="stat-grid">
            <div className="stat-card">
              <span>Compute</span>
              <strong>{gpuLabel}</strong>
            </div>
            <div className="stat-card">
              <span>Loaded Model</span>
              <strong>{status?.model_id ?? "Not loaded"}</strong>
            </div>
          </div>
          <label className="field">
            <span>Quantized Model Repo</span>
            <input value={modelId} onChange={(event) => setModelId(event.target.value)} />
          </label>
          <div className="field-grid">
            <label className="field">
              <span>Device</span>
              <input value="CUDA only" disabled />
            </label>
            <label className="toggle">
              <input
                type="checkbox"
                checked={keepModelsOnGpu}
                onChange={(event) => setKeepModelsOnGpu(event.target.checked)}
              />
              <span>Keep all models on GPU</span>
            </label>
          </div>
          <button className="primary-button" onClick={handleLoadModel} disabled={busy}>
            {busy ? <LoaderCircle className="spin" size={18} /> : <Rocket size={18} />}
            Load Runtime
          </button>
          {notice ? <p className="error-text">{notice}</p> : null}
        </section>

        <section className="panel">
          <div className="panel-header">
            <Settings2 size={18} />
            <h2>Generation</h2>
          </div>
          <div className="field-grid triple">
            <label className="field">
              <span>Pipeline</span>
              <select value={pipelineType} onChange={(event) => setPipelineType(event.target.value)}>
                <option value="512">512</option>
                <option value="1024">1024</option>
                <option value="1024_cascade">1024 Cascade</option>
                <option value="1536_cascade">1536 Cascade</option>
              </select>
            </label>
            <label className="field">
              <span>Samples</span>
              <input
                type="number"
                min={1}
                max={4}
                value={numSamples}
                onChange={(event) => setNumSamples(Number(event.target.value))}
              />
            </label>
            <label className="field">
              <span>Seed</span>
              <input
                type="number"
                value={seed}
                onChange={(event) => setSeed(Number(event.target.value))}
                disabled={randomizeSeed}
              />
            </label>
          </div>
          <div className="field-grid triple">
            <label className="toggle">
              <input
                type="checkbox"
                checked={randomizeSeed}
                onChange={(event) => setRandomizeSeed(event.target.checked)}
              />
              <span>Randomize seed</span>
            </label>
            <label className="toggle">
              <input
                type="checkbox"
                checked={preprocessImage}
                onChange={(event) => setPreprocessImage(event.target.checked)}
              />
              <span>Auto-remove background</span>
            </label>
            <label className="field">
              <span>Max Tokens</span>
              <input
                type="number"
                value={maxNumTokens}
                onChange={(event) => setMaxNumTokens(Number(event.target.value))}
              />
            </label>
          </div>
          <div className="field-grid triple">
            <label className="field">
              <span>Decimation Target</span>
              <input
                type="number"
                value={decimationTarget}
                onChange={(event) => setDecimationTarget(Number(event.target.value))}
              />
            </label>
            <label className="field">
              <span>Texture Size</span>
              <select value={textureSize} onChange={(event) => setTextureSize(Number(event.target.value))}>
                <option value={1024}>1024</option>
                <option value={2048}>2048</option>
                <option value={3072}>3072</option>
                <option value={4096}>4096</option>
              </select>
            </label>
            <label className="field">
              <span>Preview Views</span>
              <input
                type="number"
                min={4}
                max={12}
                value={previewViews}
                onChange={(event) => setPreviewViews(Number(event.target.value))}
              />
            </label>
          </div>
        </section>

        <section className="panel">
          <div className="panel-header">
            <Sparkles size={18} />
            <h2>Sampler Controls</h2>
          </div>
          <SamplerEditor title="Sparse Structure" state={sparseStructure} onChange={setSparseStructure} accent="blue" />
          <SamplerEditor title="Shape SLat" state={shapeSlat} onChange={setShapeSlat} accent="gold" />
          <SamplerEditor title="Texture SLat" state={texSlat} onChange={setTexSlat} accent="mint" />
        </section>
      </aside>

      <main className="stage-column">
        <section className="hero-panel">
          <div className="hero-copy">
            <div className="eyebrow">Studio Session</div>
            <h2>Push the quantized model as hard as your VRAM allows</h2>
            <p>
              Load the INT8 bundle once, keep it resident on CUDA, and iterate with the same
              sampler controls the original TRELLIS app exposed.
            </p>
          </div>
          <div className="performance-note">
            <Zap size={18} />
            <span>Low-VRAM offloading is disabled by default in this app runtime.</span>
          </div>
        </section>

        <section className="workspace-grid">
          <div className="drop-panel">
            <div className="panel-header">
              <ImagePlus size={18} />
              <h2>Source Image</h2>
            </div>
            <label className="dropzone">
              <input
                type="file"
                accept="image/png,image/jpeg,image/webp"
                onChange={(event) => updatePreview(event.target.files?.[0] ?? null)}
              />
              {imagePreview ? <img src={imagePreview} alt="Prompt preview" /> : <span>Drop or browse an image prompt</span>}
            </label>
            <button className="launch-button" onClick={handleGenerate} disabled={busy || !imageFile}>
              {busy ? <LoaderCircle className="spin" size={20} /> : <Cuboid size={20} />}
              Generate 3D Asset
            </button>
          </div>

          <div className="result-panel">
            <div className="panel-header">
              <Cpu size={18} />
              <h2>Result</h2>
            </div>
            <div className="job-status">
              <span className={`status-pill status-${job?.status ?? "idle"}`}>{job?.status ?? "idle"}</span>
              <strong>{job?.message ?? "Load a model, then launch a generation job."}</strong>
              {job?.seed ? <span>Seed: {job.seed}</span> : null}
              {job?.error ? <span className="error-text">{job.error}</span> : null}
            </div>

            {job?.samples.length ? (
              <>
                <div className="sample-strip">
                  {job.samples.map((sample) => (
                    <button
                      key={sample.glb_name}
                      className={selectedSample?.glb_name === sample.glb_name ? "sample-chip active" : "sample-chip"}
                      onClick={() => setSelectedSample(sample)}
                    >
                      {sample.label}
                    </button>
                  ))}
                </div>
                <div className="preview-frame">
                  {selectedSample ? (
                    <>
                      <img className="preview-sheet" src={selectedSample.preview_url} alt={selectedSample.label} />
                      <model-viewer
                        src={selectedSample.glb_url}
                        alt={selectedSample.label}
                        cameraControls
                        autoplay
                        interactionPrompt="auto"
                        shadowIntensity="1"
                        exposure="1.05"
                        environmentImage="neutral"
                      />
                      <a className="download-link" href={selectedSample.glb_url}>
                        Download {selectedSample.glb_name}
                      </a>
                    </>
                  ) : null}
                </div>
              </>
            ) : (
              <div className="result-placeholder">Your generated previews and downloadable GLB files will appear here.</div>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}

function SamplerEditor({
  title,
  state,
  onChange,
  accent,
}: {
  title: string;
  state: SamplerPanel;
  onChange: (next: SamplerPanel) => void;
  accent: string;
}) {
  function patch(key: keyof SamplerPanel, value: number | string) {
    onChange({ ...state, [key]: value });
  }

  return (
    <div className={`sampler-card accent-${accent}`}>
      <div className="sampler-head">
        <h3>{title}</h3>
      </div>
      <div className="field-grid quad">
        <label className="field">
          <span>Steps</span>
          <input type="number" min={1} max={50} value={state.steps} onChange={(event) => patch("steps", Number(event.target.value))} />
        </label>
        <label className="field">
          <span>Guidance</span>
          <input
            type="number"
            step="0.1"
            value={state.guidance_strength}
            onChange={(event) => patch("guidance_strength", Number(event.target.value))}
          />
        </label>
        <label className="field">
          <span>Rescale</span>
          <input
            type="number"
            step="0.05"
            value={state.guidance_rescale}
            onChange={(event) => patch("guidance_rescale", Number(event.target.value))}
          />
        </label>
        <label className="field">
          <span>Rescale T</span>
          <input
            type="number"
            step="0.1"
            value={state.rescale_t}
            onChange={(event) => patch("rescale_t", Number(event.target.value))}
          />
        </label>
      </div>
      <label className="field">
        <span>Advanced JSON Overrides</span>
        <textarea rows={3} value={state.overrides} onChange={(event) => patch("overrides", event.target.value)} />
      </label>
    </div>
  );
}
