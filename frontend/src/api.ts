import type { StudioJob, StudioStatus } from "./types";

export async function fetchStatus(): Promise<StudioStatus> {
  const response = await fetch("/api/status");
  if (!response.ok) {
    throw new Error("Unable to fetch backend status.");
  }
  return response.json();
}

export async function loadModel(payload: {
  model_id: string;
  keep_models_on_gpu: boolean;
  device: string;
}): Promise<StudioStatus> {
  const response = await fetch("/api/load", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json();
}

export async function createJob(image: File, settings: Record<string, unknown>): Promise<StudioJob> {
  const form = new FormData();
  form.append("image", image);
  form.append("settings", JSON.stringify(settings));

  const response = await fetch("/api/generate", {
    method: "POST",
    body: form
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json();
}

export async function fetchJob(jobId: string): Promise<StudioJob> {
  const response = await fetch(`/api/jobs/${jobId}`);
  if (!response.ok) {
    throw new Error("Unable to fetch job status.");
  }
  return response.json();
}
