/**
 * Every backend path the frontend uses, in one place.
 *
 * Components import functions, not strings, so a route change is a one-line
 * edit here rather than a search across the UI. Each function's return type is
 * the response model from `types.ts`.
 */

import { API_BASE, ApiError, errorFromResponse, request } from "./client";
import type {
  ArtifactListResponse,
  CancelResponse,
  CapabilitiesResponse,
  CleanupLevel,
  ComponentStatus,
  DeleteResponse,
  HealthResponse,
  JobCreateRequest,
  JobListResponse,
  JobStatusResponse,
  LanguagesResponse,
  LogsResponse,
  OptionsResponse,
  OutputResponse,
  PreflightResponse,
  ResumeResponse,
  SampleListResponse,
  SystemResponse,
  UploadResponse,
  VoicesResponse,
} from "./types";

// --- System / discovery -----------------------------------------------------

export function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  return request<HealthResponse>("/health", { signal });
}

export function getSystem(signal?: AbortSignal): Promise<SystemResponse> {
  return request<SystemResponse>("/system", { signal });
}

export function getModels(signal?: AbortSignal): Promise<ComponentStatus[]> {
  return request<ComponentStatus[]>("/models", { signal });
}

/**
 * The route matrix is opt-in because computing it imports the translation
 * package. The language explorer asks for it; nothing else should.
 */
export function getCapabilities(
  options: { includeRoutes?: boolean; signal?: AbortSignal } = {},
): Promise<CapabilitiesResponse> {
  return request<CapabilitiesResponse>("/capabilities", {
    query: options.includeRoutes ? { include_routes: true } : undefined,
    signal: options.signal,
  });
}

export function getLanguages(signal?: AbortSignal): Promise<LanguagesResponse> {
  return request<LanguagesResponse>("/languages", { signal });
}

// --- Bundled samples --------------------------------------------------------

/**
 * The bundled demo samples.
 *
 * The backend never lists a directory; this returns only the allow-listed
 * entries, each with an `available` flag so the demo page can gate itself on
 * what is actually present on this machine.
 */
export function getSamples(signal?: AbortSignal): Promise<SampleListResponse> {
  return request<SampleListResponse>("/samples", { signal });
}

/**
 * Stage a bundled sample as an ordinary upload.
 *
 * Returns the same `UploadResponse` as `uploadVideo`, so a demo starts a job
 * with the normal `createJob` call. The sample is copied server-side; the
 * client never sees or sends a path.
 */
export function claimSample(
  sampleId: string,
  signal?: AbortSignal,
): Promise<UploadResponse> {
  return request<UploadResponse>(
    `/samples/${encodeURIComponent(sampleId)}/claim`,
    { method: "POST", signal },
  );
}

/** The URL a `<video>` preview of a sample points at. */
export function sampleMediaUrl(sampleId: string): string {
  return `${API_BASE}/samples/${encodeURIComponent(sampleId)}/media`;
}

export function getOptions(signal?: AbortSignal): Promise<OptionsResponse> {
  return request<OptionsResponse>("/options", { signal });
}

export function getVoices(
  ttsModel: string,
  language: string,
  signal?: AbortSignal,
): Promise<VoicesResponse> {
  return request<VoicesResponse>("/voices", {
    query: { tts_model: ttsModel, language },
    signal,
  });
}

// --- Jobs -------------------------------------------------------------------

export function getJobs(signal?: AbortSignal): Promise<JobListResponse> {
  return request<JobListResponse>("/jobs", { signal });
}

export function getJob(
  jobId: string,
  signal?: AbortSignal,
): Promise<JobStatusResponse> {
  return request<JobStatusResponse>(`/jobs/${encodeURIComponent(jobId)}`, {
    signal,
  });
}

/**
 * Ask the backend to resume an interrupted job. Whether that is allowed is the
 * server's decision (`resume.can_resume`); this only sends the request.
 */
export function resumeJob(
  jobId: string,
  signal?: AbortSignal,
): Promise<ResumeResponse> {
  return request<ResumeResponse>(
    `/jobs/${encodeURIComponent(jobId)}/resume`,
    { method: "POST", signal },
  );
}

export function getJobLogs(
  jobId: string,
  options: { limit?: number; signal?: AbortSignal } = {},
): Promise<LogsResponse> {
  return request<LogsResponse>(`/jobs/${encodeURIComponent(jobId)}/logs`, {
    query: options.limit !== undefined ? { limit: options.limit } : undefined,
    signal: options.signal,
  });
}

export function getJobArtifacts(
  jobId: string,
  signal?: AbortSignal,
): Promise<ArtifactListResponse> {
  return request<ArtifactListResponse>(
    `/jobs/${encodeURIComponent(jobId)}/artifacts`,
    { signal },
  );
}

export function getJobOutput(
  jobId: string,
  signal?: AbortSignal,
): Promise<OutputResponse> {
  return request<OutputResponse>(`/jobs/${encodeURIComponent(jobId)}/output`, {
    signal,
  });
}

/**
 * Ask the backend to cancel a job.
 *
 * Cancellation is a request, not an outcome: model execution only observes it at
 * a safe checkpoint, so the returned `state` may still be `running`. The caller
 * renders "cancellation requested" and waits for the durable state to change
 * rather than assuming success.
 */
export function cancelJob(
  jobId: string,
  signal?: AbortSignal,
): Promise<CancelResponse> {
  return request<CancelResponse>(`/jobs/${encodeURIComponent(jobId)}/cancel`, {
    method: "POST",
    signal,
  });
}

export function deleteJob(
  jobId: string,
  level: CleanupLevel = "KEEP_ALL",
  signal?: AbortSignal,
): Promise<DeleteResponse> {
  return request<DeleteResponse>(`/jobs/${encodeURIComponent(jobId)}`, {
    method: "DELETE",
    query: { level },
    signal,
  });
}

export function createJob(
  body: JobCreateRequest,
  signal?: AbortSignal,
): Promise<JobStatusResponse> {
  return request<JobStatusResponse>("/jobs", {
    method: "POST",
    body,
    signal,
  });
}

/**
 * Ask the backend whether a render is ready, and why not if it is not.
 *
 * Advisory and read-only: it creates nothing. The UI uses it to explain a
 * disabled Start button; `createJob` remains the authority.
 */
export function preflightJob(
  body: JobCreateRequest,
  signal?: AbortSignal,
): Promise<PreflightResponse> {
  return request<PreflightResponse>("/jobs/preflight", {
    method: "POST",
    body,
    signal,
  });
}

// --- Uploads ----------------------------------------------------------------

/**
 * Stage a source video.
 *
 * The endpoint takes the file as a **raw request body**, not multipart, with the
 * filename in the query string — that is the contract in `src/api/routes/
 * uploads.py`, so the body is the bytes themselves. It uses `XMLHttpRequest`
 * rather than `fetch` because upload progress is not observable through `fetch`;
 * `onProgress` receives the loaded and total byte counts so a large file can be
 * shown as its own percentage, distinct from any pipeline progress.
 */
export function uploadVideo(
  file: File,
  handlers: {
    onProgress?: (loaded: number, total: number) => void;
    signal?: AbortSignal;
  } = {},
): Promise<UploadResponse> {
  return new Promise<UploadResponse>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const url = `${API_BASE}/uploads?filename=${encodeURIComponent(file.name)}`;
    xhr.open("POST", url);
    xhr.setRequestHeader("Content-Type", "application/octet-stream");
    xhr.setRequestHeader("Accept", "application/json");

    const onAbort = () => xhr.abort();
    if (handlers.signal) {
      if (handlers.signal.aborted) {
        reject(new DOMException("Aborted", "AbortError"));
        return;
      }
      handlers.signal.addEventListener("abort", onAbort, { once: true });
    }

    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) {
        handlers.onProgress?.(event.loaded, event.total);
      }
    };

    xhr.onload = () => {
      handlers.signal?.removeEventListener("abort", onAbort);
      let parsed: unknown = null;
      try {
        parsed = JSON.parse(xhr.responseText);
      } catch {
        parsed = null;
      }
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(parsed as UploadResponse);
        return;
      }
      reject(errorFromResponse(xhr.status, parsed));
    };

    xhr.onerror = () => {
      handlers.signal?.removeEventListener("abort", onAbort);
      reject(ApiError.offline());
    };

    xhr.onabort = () => {
      handlers.signal?.removeEventListener("abort", onAbort);
      reject(new DOMException("Aborted", "AbortError"));
    };

    xhr.send(file);
  });
}

/** Discard a staged upload that will not be used. Best-effort. */
export function discardUpload(
  uploadId: string,
  signal?: AbortSignal,
): Promise<void> {
  return request<void>(`/uploads/${encodeURIComponent(uploadId)}`, {
    method: "DELETE",
    signal,
  });
}
