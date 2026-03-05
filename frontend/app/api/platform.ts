import { AppRoutes } from './routes'

export interface PlatformLimits {
  max_training_minutes : number
  max_dataset_mb       : number
  max_memory_gb        : number
  max_output_gb        : number
  session_ttl_hours    : number
  max_epochs           : number
}

export const getPlatformLimits = async (): Promise<PlatformLimits | null> => {
  try {
    const res = await fetch(AppRoutes.PlatformLimits())
    if (!res.ok) return null
    return res.json()
  } catch {
    return null
  }
}

export const createSession = async (): Promise<{ session_id: string } | null> => {
  try {
    const res = await fetch(AppRoutes.CreateSession(), { method: 'POST', credentials: 'include' })
    if (!res.ok) return null
    return res.json()
  } catch {
    return null
  }
}

export const uploadDataset = async (
  sessionId: string,
  file: File,
  onProgress?: (pct: number) => void
): Promise<{ path: string; size: number } | null> => {
  try {
    const form = new FormData()
    form.append('file', file)

    // Use XMLHttpRequest for upload progress
    return await new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest()
      xhr.open('POST', AppRoutes.UploadDataset(sessionId))
      xhr.withCredentials = true

      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable && onProgress) {
          onProgress(Math.round((e.loaded / e.total) * 100))
        }
      }

      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(JSON.parse(xhr.responseText))
        } else {
          reject(new Error(`Upload failed: ${xhr.statusText}`))
        }
      }

      xhr.onerror = () => reject(new Error('Upload network error'))
      xhr.send(form)
    })
  } catch {
    return null
  }
}

export const getDatasetPreview = async (sessionId: string) => {
  try {
    const res = await fetch(AppRoutes.PreviewDataset(sessionId), { credentials: 'include' })
    if (!res.ok) return null
    return res.json()
  } catch {
    return null
  }
}

export interface DatasetAnalysis {
  dataset_type          : 'tabular' | 'time_series' | 'nlp' | 'image'
  task_type             : 'binary_classification' | 'multiclass_classification' | 'regression' | 'rl_trading'
  n_rows                : number
  n_features            : number
  target_col            : string
  n_classes             : number
  class_imbalance_ratio : number | null
  fraction_categorical  : number
  fraction_missing      : number
  max_cardinality       : number
  datetime_detected     : boolean
  datetime_col          : string | null
  ts_frequency          : string | null
  avg_string_length     : number
  columns               : string[]
  dtypes                : Record<string, string>
}

export const getDatasetAnalysis = async (sessionId: string): Promise<DatasetAnalysis | null> => {
  try {
    const res = await fetch(AppRoutes.DatasetAnalysis(sessionId), { credentials: 'include' })
    if (!res.ok) return null
    return res.json()
  } catch {
    return null
  }
}

export interface ArtifactStatus {
  notebook        : boolean
  notebook_mtime  : number | null
  models          : string[]
  images          : string[]
  epochs_run      : number | null
  epochs_max      : number | null
}

export const getArtifactStatus = async (sessionId: string): Promise<ArtifactStatus | null> => {
  try {
    const res = await fetch(AppRoutes.ArtifactStatus(sessionId), { credentials: 'include' })
    if (!res.ok) return null
    return res.json()
  } catch {
    return null
  }
}

export const downloadFile = (url: string, filename: string) => {
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
}

export const downloadModelFile = (sessionId: string, filename: string) =>
  downloadFile(AppRoutes.DownloadModelFile(sessionId, filename), filename)

export const downloadNotebook = (sessionId: string) =>
  downloadFile(AppRoutes.DownloadNotebook(sessionId), 'training_notebook.ipynb')

export const triggerCompile = async (sessionId: string): Promise<{ task_id: string } | null> => {
  try {
    const res = await fetch(AppRoutes.TriggerCompile(sessionId), {
      method: 'POST',
      credentials: 'include',
    })
    if (!res.ok) return null
    return res.json()
  } catch {
    return null
  }
}
