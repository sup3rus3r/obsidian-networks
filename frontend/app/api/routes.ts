// All routes use the /api prefix so Next.js rewrites proxy them to the backend.
// In Docker: INTERNAL_API_URL=http://api:8000 (set in next.config.ts rewrite).
// In local dev: next.config.ts rewrites to http://localhost:8000.

export const AppRoutes = {
  CreateSession:      ()            => `/api/platform/session`,
  UploadDataset:      (sid: string) => `/api/platform/upload/${sid}`,
  PreviewDataset:     (sid: string) => `/api/platform/preview/${sid}`,
  DatasetAnalysis:    (sid: string) => `/api/platform/analysis/${sid}`,
  ArtifactStatus:     (sid: string) => `/api/platform/status/${sid}`,
  CreateNotebook:     (sid: string) => `/api/platform/notebook/${sid}`,
  JobProgress:        (tid: string) => `/api/platform/progress/${tid}`,
  DownloadModelFile:  (sid: string, filename: string) => `/api/platform/download/${sid}/model/${filename}`,
  DownloadNotebook:   (sid: string) => `/api/platform/download/${sid}/notebook`,
  ViewImage:          (sid: string, filename: string) => `/api/platform/download/${sid}/image/${filename}`,
  TriggerCompile:     (sid: string) => `/api/platform/compile/${sid}`,
}
