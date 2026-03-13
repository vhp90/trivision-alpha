export type StudioStatus = {
  model_loaded: boolean;
  model_id: string | null;
  keep_models_on_gpu: boolean;
  device: string;
  gpu: {
    available: boolean;
    name?: string;
    total_vram_gb?: number;
    free_vram_gb?: number;
    allocated_vram_gb?: number;
    reserved_vram_gb?: number;
  };
  defaults: Record<string, unknown>;
};

export type JobSample = {
  label: string;
  preview_url: string;
  glb_url: string;
  glb_name: string;
};

export type StudioJob = {
  id: string;
  created_at: string;
  updated_at: string;
  status: "queued" | "running" | "completed" | "failed";
  stage: string;
  message: string;
  error?: string | null;
  seed?: number | null;
  settings: Record<string, unknown>;
  samples: JobSample[];
};

export type SamplerPanel = {
  steps: number;
  guidance_strength: number;
  guidance_rescale: number;
  rescale_t: number;
  overrides: string;
};
