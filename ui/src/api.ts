export type BootStageState = 'idle' | 'run' | 'wait' | 'done' | 'error' | 'skip'

export type BootStage = {
  state: BootStageState
  progress: number
  label: string
}

export type BootStatus = {
  ready: boolean
  running: boolean
  error: string
  awaiting?: string
  suggested?: string
  progress?: number
  progress_label?: string
  stages: {
    model: BootStage
    character: BootStage
    lab: BootStage
  }
}

export type AppStatus = {
  state: string
  message: string
  checkpoint: string
  /** Picked in the Model list, not loaded yet. Start stream loads it. */
  pending_checkpoint?: string
  keypoint_layout?: string
  device: string
  error: string
  model_ready: boolean
  ref_ready: boolean
  streaming: boolean
  paused?: boolean
  tracking: boolean
  track_busy?: boolean
  busy: boolean
  models_on_gpu?: boolean
  fast_warming: boolean
  steps: number
  pose_cfg: number
  id_cfg: number
  frame_blend: number
  inbetweens?: number
  interpolate?: boolean
  /** Cap on generated keys per second; 0 = as fast as the GPU goes. */
  max_fps?: number
  hold_last?: boolean
  track_fps: number
  drive_pose: boolean
  show_mesh: boolean
  show_hair: boolean
  show_outline?: boolean
  show_brows?: boolean
  show_eyes?: boolean
  show_nose?: boolean
  show_mouth?: boolean
  show_iris_overlay?: boolean
  show_skeleton?: boolean
  show_limiters?: boolean
  mirror: boolean
  use_iris: boolean
  use_body: boolean
  fast_mode: boolean
  compile_model: boolean
  batch2: boolean
  auto_sync_track: boolean
  gen_fps: number
  show_fps?: number
  timing: string
  reference_name: string
  reference_path: string
  character_id?: string
  character_name?: string
  camera_index: number
  track_message: string
  body_label: string
  progress: number
  progress_label: string
  progress_kind: string
  compile_on: boolean
  compile_status: string
  compile_detail: string
  virtual_cam: boolean
  virtual_cam_device: string
  virtual_cam_error: string
  virtual_cam_width?: number
  virtual_cam_height?: number
  pose_frozen?: boolean
  pose_key_count?: number
  travel_box?: TravelBox
  boot?: BootStatus
}

export type TravelBox = {
  version?: number
  enabled: boolean
  left: number
  right: number
  up: number
  down: number
  body_left: number
  body_right: number
  body_up: number
  body_down: number
  turn_left: number
  turn_right: number
  tilt_left: number
  tilt_right: number
  /** Saved before turn / tilt had sides: one cap for both. */
  yaw?: number
  roll?: number
  pitch_up: number
  pitch_down: number
  eye: number
  size: number
}

export type Checkpoint = { label: string; path: string; source?: string }
export type CatalogOffer = {
  name: string
  label: string
  path: string
  is_new: boolean
  published?: string
  badge: string
}
export type FitBox = [number, number, number, number] | null

export type CharacterFit = {
  width: number
  height: number
  face_height: number
  hair: { class: string; polygon: number[][] }[]
  skeleton: { id: number; label: string; x: number; y: number }[]
  boxes: {
    head_tight: FitBox
    head: FitBox
    body_tight: FitBox
    body: FitBox
  }
}

export type CharacterCard = {
  id: string
  name: string
  path: string
  preview_url: string
  thumb_url?: string
  author?: string
  version?: number
  shapes_compatible?: boolean
  has_shapes?: boolean
  shapes_path?: string
}

export type CharacterInfo = {
  id: string
  name: string
  version: number
  created_at: string
  updated_at: string
  author: string
  license: string
  description: string
  model: {
    checkpoint: string
    image_size: number
    latent_shape: number[]
  }
  model_match: boolean
  includes: {
    pose_keys: number
    blendshapes: boolean
    hair: boolean
    skeleton: boolean
    travel_box: boolean
    source_image: boolean
  }
  size_bytes: number
}

export type CharacterMeta = {
  id: string
  name?: string
  author?: string
  license?: string
  description?: string
}

/** Small grid image; older backends only serve the full preview. */
export function characterThumb(card: CharacterCard): string {
  return card.thumb_url || card.preview_url
}

export function characterExportUrl(id: string): string {
  return `/api/characters/${encodeURIComponent(id)}/export`
}

function attachmentName(res: Response, fallback: string): string {
  const header = res.headers.get('Content-Disposition') || ''
  const star = /filename\*\s*=\s*(?:UTF-8'')?([^;]+)/i.exec(header)
  if (star) {
    try {
      return decodeURIComponent(star[1].trim().replace(/^"|"$/g, ''))
    } catch {
      /* fall through */
    }
  }
  const plain = /filename\s*=\s*"?([^";]+)"?/i.exec(header)
  return plain ? plain[1].trim() : fallback
}

/**
 * Fetch the .vtm first so a server error shows in the desk instead of landing
 * on disk, then hand the bytes to the browser / WebView2 as a download.
 * In pywebview this needs webview.settings['ALLOW_DOWNLOADS'] = True, which
 * opens a native Save dialog for the file.
 */
async function downloadCharacter(id: string, name: string): Promise<string> {
  const res = await fetch(characterExportUrl(id))
  if (!res.ok) await json<unknown>(res)
  const blob = await res.blob()
  const filename = attachmentName(res, `${name || id}.vtm`)
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  link.rel = 'noopener'
  link.style.display = 'none'
  document.body.appendChild(link)
  link.click()
  link.remove()
  window.setTimeout(() => URL.revokeObjectURL(url), 60_000)
  return filename
}

export type CharacterExportResult = {
  /** Where the file went: a full path (native dialog) or the download name. */
  saved: string
  native: boolean
  cancelled: boolean
}

/**
 * Prefer the desk's native Save dialog (pywebview); older backends lack the
 * endpoint, so a 404/405 (or a reply without `ok`) falls back to a download.
 */
async function exportCharacter(id: string, name: string): Promise<CharacterExportResult> {
  const res = await fetch('/api/characters/export-save', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ id }),
  })
  if (res.status !== 404 && res.status !== 405) {
    const body = await json<{ ok?: boolean; path?: string; cancelled?: boolean; error?: string }>(res)
    if (body && typeof body.ok === 'boolean') {
      if (body.ok) return { saved: body.path || '', native: true, cancelled: false }
      if (body.cancelled) return { saved: '', native: true, cancelled: true }
      throw new Error(body.error || 'Export failed')
    }
  }
  const saved = await downloadCharacter(id, name)
  return { saved, native: false, cancelled: false }
}

/** One line for the desk after an export; empty when the user cancelled. */
export function exportNote(name: string, res: CharacterExportResult): string {
  if (res.cancelled) return ''
  if (res.native) return res.saved ? `Saved ${name} to ${res.saved}` : `Saved ${name}.`
  return `Exported ${name} as ${res.saved}.`
}

export type CharacterLoadResult = {
  ok: boolean
  incompatible?: boolean
  message?: string
  character?: CharacterCard
  frame?: FrameEvent
  status?: AppStatus
}
export type CameraInfo = { index: number; name: string }
export type MixWeights = {
  A: number
  I: number
  U: number
  E: number
  smile: number
  sad: number
}

export type IrisCamHit = {
  side: string
  x: number
  y: number
  score: number
  visible: boolean
  method: string
}

export type LabLook = { x: number; y: number }

export type LabFeel = {
  response: number
  smoothing: number
  mouth: number
  use_visemes: number
  show_face: number
  show_skeleton: number
  show_hair: number
  show_ids: number
  hair_pin: number
  max_yaw_left: number
  max_yaw_right: number
  max_roll_left: number
  max_roll_right: number
  max_pitch_up: number
  max_pitch_down: number
  max_size: number
  max_look_x: number
  max_look_y: number
  gaze_gain: number
  gaze_smooth: number
}

export type LabCalib = {
  rest?: boolean
  capturing?: string
  progress?: number
  error?: string
  hint?: string
}

export type LabIfm = {
  host?: string
  port?: number
  listening?: boolean
  receiving?: boolean
  fps?: number
  peer?: string
  hint?: string
  local?: string[]
  primary?: string
}

export type LabStatus = {
  type?: string
  online: boolean
  ok?: boolean
  live?: boolean
  error?: string
  feel?: LabFeel
  weights?: MixWeights
  points?: number[][]
  skeleton?: { id: number; x: number; y: number; score?: number }[]
  hair?: { class: string; polygon: number[][] }[]
  source?: 'camera' | 'ifm'
  camera_index?: number
  cameras?: CameraInfo[]
  calib?: LabCalib
  ifm?: LabIfm
  commands?: string[]
  handshake?: boolean
  loaded?: boolean
  point_offsets?: { id: number; dx: number; dy: number }[]
  shapes?: Record<string, number[][]>
  presets?: { id: string; label: string; ready?: boolean }[]
  active?: string
  iris_method?: string
  iris?: { id: number; x: number; y: number; score: number; visible?: boolean }[]
  iris_cam?: IrisCamHit[]
  look?: LabLook | null
  head?: { pitch: number; yaw: number; roll: number }
  blink?: { l?: number; r?: number }
}

export const ZERO_LAB_FEEL: LabFeel = {
  response: 0.65,
  smoothing: 0.48,
  mouth: 0.5,
  use_visemes: 1,
  show_face: 1,
  show_skeleton: 1,
  show_hair: 1,
  show_ids: 0,
  hair_pin: 0.7,
  max_yaw_left: 1,
  max_yaw_right: 1,
  max_roll_left: 1,
  max_roll_right: 1,
  max_pitch_up: 1,
  max_pitch_down: 1,
  max_size: 1,
  max_look_x: 1,
  max_look_y: 1,
  gaze_gain: 1,
  gaze_smooth: 0.28,
}

export const ZERO_WEIGHTS: MixWeights = {
  A: 0,
  I: 0,
  U: 0,
  E: 0,
  smile: 0,
  sad: 0,
}

export function labBannerError(lab: LabStatus | null | undefined): string {
  const err = String(lab?.error || '').trim()
  if (!err) return ''
  if (!lab?.live && /track a face first so rest exists/i.test(err)) return ''
  return err
}

export function labSourceOf(lab: LabStatus | null | undefined): 'camera' | 'ifm' {
  return lab?.source === 'ifm' ? 'ifm' : 'camera'
}

export function mergeLabStatus(cur: LabStatus | null | undefined, next: LabStatus): LabStatus {
  const server = next.source === 'ifm' ? 'ifm' : next.source === 'camera' ? 'camera' : undefined
  const local = cur?.source === 'ifm' ? 'ifm' : cur?.source === 'camera' ? 'camera' : undefined
  const source: 'camera' | 'ifm' = next.online
    ? (server ?? local ?? 'camera')
    : (local ?? server ?? 'camera')
  return { ...cur, ...next, source }
}

export type LabInputHold = {
  source: 'camera' | 'ifm'
  gen: number
  settled: boolean
}

/** Keep Camera / iFacialMocap on the click until a poll that started after the command agrees. */
export function holdLabInput(
  cur: LabStatus | null | undefined,
  next: LabStatus,
  hold: LabInputHold | null,
  seen: number,
): { lab: LabStatus; release: boolean } {
  const merged = mergeLabStatus(cur, next)
  if (!hold) return { lab: merged, release: false }
  const agrees = merged.source === hold.source
  const release = Boolean(hold.settled && seen >= hold.gen && agrees)
  if (agrees) return { lab: merged, release }
  return { lab: { ...merged, source: hold.source }, release: false }
}

export type LabCommandReply = {
  ok?: boolean
  error?: string
  status?: LabStatus
  online?: boolean
}

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
    let detail: unknown = res.statusText
    try {
      const body = await res.json()
      detail = body.detail ?? body
    } catch {
      /* ignore */
    }
    if (typeof detail !== 'string') {
      detail = Array.isArray(detail)
        ? detail
            .map((item) =>
              item && typeof item === 'object' && 'msg' in item
                ? String((item as { msg: string }).msg)
                : JSON.stringify(item),
            )
            .join('; ')
        : JSON.stringify(detail)
    }
    throw new Error(String(detail))
  }
  return res.json() as Promise<T>
}

export const api = {
  status: () => fetch('/api/status').then((r) => json<AppStatus>(r)),
  boot: () => fetch('/api/boot').then((r) => json<BootStatus>(r)),
  startBoot: () =>
    fetch('/api/boot', { method: 'POST' }).then((r) => json<BootStatus>(r)),
  checkpoints: () => fetch('/api/checkpoints').then((r) => json<Checkpoint[]>(r)),
  cameras: () =>
    fetch('/api/cameras').then((r) =>
      json<{ cameras: CameraInfo[]; preferred: number }>(r),
    ),
  modelCatalog: () =>
    fetch('/api/models/catalog').then((r) => json<{ offers: CatalogOffer[] }>(r)),
  startModelDownload: (name?: string) =>
    fetch('/api/models/download', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(name ? { name } : {}),
    }).then((r) => json<ModelsStatus['download']>(r)),
  setCheckpoint: (path: string) =>
    fetch('/api/checkpoint', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path }),
    }).then((r) => json<AppStatus>(r)),
  browseCheckpoint: () =>
    fetch('/api/checkpoint/browse', { method: 'POST' }).then((r) =>
      json<{ cancelled: boolean; path?: string | null; status: AppStatus }>(r),
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
  characters: () =>
    fetch('/api/characters').then((r) => json<{ characters: CharacterCard[] }>(r)),
  characterFit: () => fetch('/api/characters/fit').then((r) => json<CharacterFit>(r)),
  fitHair: (body: { part: string; points: number[][]; radius: number; erase: boolean }) =>
    fetch('/api/characters/fit/hair', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }).then((r) => json<{ ok: boolean; fit: CharacterFit }>(r)),
  fitSkeleton: (id: number, x: number, y: number) =>
    fetch('/api/characters/fit/skeleton', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id, x, y }),
    }).then((r) => json<{ ok: boolean; fit: CharacterFit }>(r)),
  createCharacter: (file: File) => {
    const fd = new FormData()
    fd.append('file', file)
    return fetch('/api/characters/create', { method: 'POST', body: fd }).then((r) =>
      json<{
        ok: boolean
        character: CharacterCard
        frame: FrameEvent
        status: AppStatus
      }>(r),
    )
  },
  loadCharacter: (id: string, opts: { repair?: boolean } = {}) =>
    fetch('/api/characters/load', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id, repair: Boolean(opts.repair) }),
    }).then((r) => json<CharacterLoadResult>(r)),
  removeCharacter: (id: string) =>
    fetch('/api/characters/remove', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id }),
    }).then((r) =>
      json<{
        ok: boolean
        status: AppStatus
        characters: CharacterCard[]
      }>(r),
    ),
  importCharacter: (file: File) => {
    const fd = new FormData()
    fd.append('file', file)
    return fetch('/api/characters/add', { method: 'POST', body: fd }).then((r) =>
      json<{ ok: boolean; character: CharacterCard; status?: AppStatus }>(r),
    )
  },
  exportCharacter: (id: string, name = '') => exportCharacter(id, name),
  /** Open Explorer with the character's .vtm selected. */
  revealCharacter: (id: string) =>
    fetch('/api/characters/reveal', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id }),
    }).then((r) => json<{ ok: boolean; path: string }>(r)),
  characterInfo: (id: string) =>
    fetch(`/api/characters/${encodeURIComponent(id)}/info`).then((r) =>
      json<CharacterInfo>(r),
    ),
  characterMeta: (body: CharacterMeta) =>
    fetch('/api/characters/meta', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }).then((r) =>
      json<{
        ok: boolean
        character: CharacterCard
        characters: CharacterCard[]
      }>(r),
    ),
  renameCharacter: (id: string, name: string) =>
    fetch('/api/characters/rename', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id, name }),
    }).then((r) =>
      json<{
        ok: boolean
        character: CharacterCard
        status: AppStatus
        characters: CharacterCard[]
      }>(r),
    ),
  startTracking: () =>
    fetch('/api/tracking/start', { method: 'POST' }).then((r) => json<AppStatus>(r)),
  stopTracking: () =>
    fetch('/api/tracking/stop', { method: 'POST' }).then((r) => json<AppStatus>(r)),
  recenter: () =>
    fetch('/api/tracking/recenter', { method: 'POST' }).then((r) => json<AppStatus>(r)),
  generate: () =>
    fetch('/api/generate', { method: 'POST' }).then((r) => json<AppStatus>(r)),
  startStream: () =>
    fetch('/api/stream/start', { method: 'POST' }).then((r) => json<AppStatus>(r)),
  stopStream: () =>
    fetch('/api/stream/stop', { method: 'POST' }).then((r) => json<AppStatus>(r)),
  pauseStream: () =>
    fetch('/api/stream/pause', { method: 'POST' }).then((r) => json<AppStatus>(r)),
  resumeStream: () =>
    fetch('/api/stream/resume', { method: 'POST' }).then((r) => json<AppStatus>(r)),
  startVirtualCam: () =>
    fetch('/api/virtual-cam/start', { method: 'POST' }).then((r) => json<AppStatus>(r)),
  stopVirtualCam: () =>
    fetch('/api/virtual-cam/stop', { method: 'POST' }).then((r) => json<AppStatus>(r)),
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
  freezePose: () =>
    fetch('/api/pose/freeze', { method: 'POST' }).then((r) => json<AppStatus>(r)),
  unfreezePose: () =>
    fetch('/api/pose/unfreeze', { method: 'POST' }).then((r) => json<AppStatus>(r)),
  labStatus: () => fetch('/api/lab/status').then((r) => json<LabStatus>(r)),
  labConnect: () =>
    fetch('/api/lab/connect', { method: 'POST' }).then((r) => json<LabStatus>(r)),
  labCommand: (op: string, body: Record<string, unknown> = {}) =>
    fetch('/api/lab/command', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ op, body }),
    }).then((r) => json<LabCommandReply>(r)),
  reloadBackend: () =>
    fetch('/api/reload', { method: 'POST' }).then((r) =>
      json<{ ok: boolean; reloading?: boolean }>(r),
    ),
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
