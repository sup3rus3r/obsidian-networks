// All routes use the /api prefix so Next.js rewrites proxy them to the backend.
// In Docker: INTERNAL_API_URL=http://api:8000 (set in next.config.ts rewrite).
// In local dev: next.config.ts rewrites to http://localhost:8000.
//
// EXCEPTION: large file uploads go direct to the backend (NEXT_PUBLIC_UPLOAD_URL)
// to bypass the Next.js proxy which buffers the full body and causes ECONNRESET.

const UPLOAD_BASE  = process.env.NEXT_PUBLIC_UPLOAD_URL ?? ''

export const AppRoutes = {
  PlatformLimits:     ()            => `/api/platform/limits`,
  CreateSession:      ()            => `/api/platform/session`,
  UploadDataset:      (sid: string) => `${UPLOAD_BASE}/platform/upload/${sid}`,
  PreviewDataset:     (sid: string) => `/api/platform/preview/${sid}`,
  DatasetAnalysis:    (sid: string) => `/api/platform/analysis/${sid}`,
  ArtifactStatus:     (sid: string) => `/api/platform/status/${sid}`,
  CreateNotebook:     (sid: string) => `/api/platform/notebook/${sid}`,
  JobProgress:        (tid: string) => `/api/platform/progress/${tid}`,
  JobProgressOnce:    (tid: string) => `/api/platform/progress-once/${tid}`,
  DownloadModelFile:   (sid: string, filename: string) => `/api/platform/download/${sid}/model/${filename}`,
  DownloadDatasetFile: (sid: string, filename: string) => `/api/platform/download/${sid}/dataset/${filename}`,
  DownloadNotebook:   (sid: string) => `/api/platform/download/${sid}/notebook`,
  ViewImage:          (sid: string, filename: string) => `/api/platform/download/${sid}/image/${filename}`,
  TriggerCompile:     (sid: string) => `/api/platform/compile/${sid}`,
  ClearOutputs:       (sid: string) => `/api/platform/outputs/${sid}`,
  // ── Research Mode ────────────────────────────────────────────────────────
  ResearchCategories:    ()                           => `/api/platform/research/categories`,
  ResearchStart:         ()                           => `/api/platform/research/start`,
  ResearchStatus:        (rid: string)                => `/api/platform/research/${rid}/status`,
  ResearchStream:        (rid: string)                => `/internal/research-stream/${rid}`,
  ResearchCandidates:    (rid: string)                => `/api/platform/research/${rid}/candidates`,
  ResearchCandidate:     (rid: string, arch: string)  => `/api/platform/research/${rid}/candidate/${arch}`,
  ResearchCompile:       (rid: string)                => `/api/platform/research/${rid}/compile`,
  ResearchCancel:        (rid: string)                => `/api/platform/research/${rid}`,
}
