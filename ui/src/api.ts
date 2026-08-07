export type AppStatus = {
  state: string
  message: string
  checkpoint: string
  device: string
  error: string
  model_ready: boolean
  ref_ready: boolean
  streaming: boolean
  tracking: boolean
  busy: boolean
  fast_warming: boolean
  steps: number
  track_fps: number
  drive_pose: boolean
  show_mesh: boolean
  mirror: boolean
  use_iris: boolean
  use_body: boolean
  fast_mode: boolean
  batch2: boolean
  auto_sync_track: boolean
  gen_fps: number
  timing: string
  reference_name: string
  reference_path: string
  camera_index: number
  track_message: string
  body_label: string
  progress: number
  progress_label: string
  progress_kind: string
  compile_on: boolean
  compile_status: string
  compile_detail: string
}

export type Checkpoint = { label: string; path: string }
export type CameraInfo = { index: number; name: string }

export type ModelsStatus = {
  dit_dir: string
  has_models: boolean
  checkpoints: { name: string; path: string; bytes: number }[]
  download_configured: boolean
  download_message: string
  download: {
    status: string
    message: string
    progress: number
    current_file: string
    error: string
    files_done: number
    files_total: number
  }
}

export type FrameEvent = {
  type: 'frame'
  image: string | null
  width: number
  height: number
  keypoints?: number[][] | null
  elapsed?: number
  fps?: number
  timing?: string
}

export type WsEvent =
  | { type: 'status'; status: AppStatus }
  | FrameEvent
  | { type: 'pong' }

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = body.detail || JSON.stringify(body)
    } catch {
      /* ignore */
    }
    throw new Error(typeof detail === 'string' ? detail : String(detail))
  }
  return res.json() as Promise<T>
}

export const api = {
  status: () => fetch('/api/status').then((r) => json<AppStatus>(r)),
  checkpoints: () => fetch('/api/checkpoints').then((r) => json<Checkpoint[]>(r)),
  modelsStatus: () => fetch('/api/models/status').then((r) => json<ModelsStatus>(r)),
  startModelDownload: () =>
    fetch('/api/models/download', { method: 'POST' }).then((r) =>
      json<ModelsStatus['download']>(r),
    ),
  modelDownloadStatus: () =>
    fetch('/api/models/download/status').then((r) => json<ModelsStatus['download']>(r)),
  reloadModels: () =>
    fetch('/api/models/reload', { method: 'POST' }).then((r) => json<AppStatus>(r)),
  cameras: () =>
    fetch('/api/cameras').then((r) =>
      json<{ cameras: CameraInfo[]; preferred: number }>(r),
    ),
  setCheckpoint: (path: string) =>
    fetch('/api/checkpoint', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path }),
    }).then((r) => json<AppStatus>(r)),
  browseCheckpoint: () =>
    fetch('/api/checkpoint/browse', { method: 'POST' }).then((r) =>
      json<{ cancelled: boolean; status: AppStatus }>(r),
    ),
  settings: (body: Partial<AppStatus>) =>
    fetch('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }).then((r) => json<AppStatus>(r)),
  applyRef: (path: string) =>
    fetch('/api/reference', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path }),
    }).then((r) => json<{ ok: boolean; frame: FrameEvent; status: AppStatus }>(r)),
  uploadRef: (file: File) => {
    const fd = new FormData()
    fd.append('file', file)
    return fetch('/api/reference/upload', { method: 'POST', body: fd }).then((r) =>
      json<{ ok: boolean; path: string; frame: FrameEvent; status: AppStatus }>(r),
    )
  },
  defaultRef: () =>
    fetch('/api/reference/default').then((r) =>
      json<{ path: string; exists: string }>(r),
    ),
  startTracking: () =>
    fetch('/api/tracking/start', { method: 'POST' }).then((r) => json<AppStatus>(r)),
  stopTracking: () =>
    fetch('/api/tracking/stop', { method: 'POST' }).then((r) => json<AppStatus>(r)),
  calibrate: () =>
    fetch('/api/tracking/calibrate', { method: 'POST' }).then((r) => json<AppStatus>(r)),
  recenter: () =>
    fetch('/api/tracking/recenter', { method: 'POST' }).then((r) => json<AppStatus>(r)),
  generate: () =>
    fetch('/api/generate', { method: 'POST' }).then((r) => json<AppStatus>(r)),
  startStream: () =>
    fetch('/api/stream/start', { method: 'POST' }).then((r) => json<AppStatus>(r)),
  stopStream: () =>
    fetch('/api/stream/stop', { method: 'POST' }).then((r) => json<AppStatus>(r)),
  meshPress: (x: number, y: number) =>
    fetch('/api/mesh/press', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ x, y }),
    }),
  meshDrag: (x: number, y: number) =>
    fetch('/api/mesh/drag', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ x, y }),
    }),
  meshRelease: () => fetch('/api/mesh/release', { method: 'POST' }),
  meshReset: () => fetch('/api/mesh/reset', { method: 'POST' }),
}

export function openAppSocket(onEvent: (ev: WsEvent) => void): WebSocket {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws'
  const ws = new WebSocket(`${proto}://${location.host}/api/ws`)
  ws.onmessage = (msg) => {
    try {
      onEvent(JSON.parse(msg.data) as WsEvent)
    } catch {
      /* ignore */
    }
  }
  const ping = window.setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) ws.send('ping')
  }, 15000)
  ws.addEventListener('close', () => window.clearInterval(ping))
  return ws
}
