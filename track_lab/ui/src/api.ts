export type MouthPreset = {
  id: string
  label: string
  ready: boolean
}

export type MixWeights = {
  A: number
  I: number
  U: number
  E: number
  O: number
  smile: number
  sad: number
  rest?: number
}

export type CalibShape = {
  id: string
  label: string
  ready: boolean
}

export type CalibBank = {
  id: string
  label: string
  shapes: CalibShape[]
  ready: boolean
}

export type CalibStatus = {
  rest: boolean
  smile?: boolean
  sad?: boolean
  vowels: Record<string, boolean>
  banks?: CalibBank[]
  capturing: string
  progress: number
  error?: string
  hint?: string
}

export type FeelSettings = {
  response: number
  smoothing: number
  mouth: number
  use_visemes: number
  show_face: number
  show_skeleton: number
  show_hair: number
  show_ids: number
  hair_pin: number
  gaze_gain: number
  gaze_smooth: number
}

export type IrisCamHit = {
  side: string
  x: number
  y: number
  score: number
  visible: boolean
  method: string
}

export type CameraDevice = {
  index: number
  name: string
  backend?: string
}

export type IfmStatus = {
  host: string
  port: number
  listening: boolean
  receiving: boolean
  fps: number
  peer: string
  last_peer?: string
  local?: string[]
  primary?: string
  hint?: string
}

export type LabStatus = {
  ok: boolean
  ready: boolean
  has_source: boolean
  source_path: string
  width: number
  height: number
  tracker: string
  faces: number
  ms: number
  generation: number
  error: string
  active: string
  mouth_slots: number[]
  presets: MouthPreset[]
  points: number[][]
  shapes: Record<string, number[][]>
  live?: boolean
  weights?: MixWeights
  head?: { pitch: number; yaw: number; roll: number }
  blink?: { l: number; r: number }
  camera_index?: number
  cameras?: CameraDevice[]
  source?: 'camera' | 'ifm'
  ifm?: IfmStatus
  calib?: CalibStatus
  feel?: FeelSettings
  mouth_points?: { id: number; on: boolean; ring: 'in' | 'out'; to: number | null }[]
  eye_points?: { id: number; on: boolean; side: 'l' | 'r'; artificial?: boolean; to: number | null }[]
  hair?: { class: string; polygon: number[][] }[]
  skeleton?: { id: number; x: number; y: number; score: number }[]
  iris?: { id: number; x: number; y: number; score: number; visible?: boolean }[]
  iris_method?: string
  iris_cam?: IrisCamHit[]
  look?: { x: number; y: number } | null
  mirror?: boolean
  point_offsets?: { id: number; dx: number; dy: number }[]
  gen?: boolean
  gen_ms?: number
  message?: string
}

export type LiveStatus = {
  ok: boolean
  live: boolean
  tracker: string
  faces: number
  ms: number
  error: string
  points: number[][]
  weights: MixWeights
  head: { pitch: number; yaw: number; roll: number }
  blink: { l: number; r: number }
  camera_index: number
  source?: 'camera' | 'ifm'
  ifm?: IfmStatus
  calib?: CalibStatus
  mouth_points?: { id: number; on: boolean; ring: 'in' | 'out'; to: number | null }[]
  eye_points?: { id: number; on: boolean; side: 'l' | 'r'; artificial?: boolean; to: number | null }[]
  hair?: { class: string; polygon: number[][] }[]
  skeleton?: { id: number; x: number; y: number; score: number }[]
  iris?: { id: number; x: number; y: number; score: number; visible?: boolean }[]
  iris_method?: string
  iris_cam?: IrisCamHit[]
  look?: { x: number; y: number } | null
  feel?: FeelSettings
  point_offsets?: { id: number; dx: number; dy: number }[]
}

async function read(res: Response): Promise<LabStatus> {
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return (await res.json()) as LabStatus
}

async function readLive(res: Response): Promise<LiveStatus> {
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return (await res.json()) as LiveStatus
}

export const api = {
  status: () => fetch('/api/status').then(read),
  track: () => fetch('/api/track', { method: 'POST' }).then(read),
  generate: (overlay?: {
    points?: number[][]
    hair?: { class: string; polygon: number[][] }[]
    skeleton?: { id: number; x: number; y: number; score: number }[]
    iris?: { id: number; x: number; y: number; score: number; visible?: boolean }[]
  }) =>
    fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(overlay ?? {}),
    }).then(read),
  reset: () => fetch('/api/reset', { method: 'POST' }).then(read),
  osfStart: (opts: { camera?: number; source?: 'camera' | 'ifm'; host?: string; port?: number }) =>
    fetch('/api/osf/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(opts),
    }).then(read),
  osfStop: () => fetch('/api/osf/stop', { method: 'POST' }).then(read),
  setInput: (source: 'camera' | 'ifm') =>
    fetch('/api/input', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source }),
    }).then(read),
  setIfm: (host: string, port: number) =>
    fetch('/api/ifm', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ host, port }),
    }).then(read),
  setCamera: (index: number) =>
    fetch('/api/camera', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ index }),
    }).then(read),
  live: () => fetch('/api/live').then(readLive),
  upload: (file: File) =>
    fetch('/api/source', {
      method: 'POST',
      headers: {
        'Content-Type': file.type || 'application/octet-stream',
        'x-filename': encodeURIComponent(file.name),
      },
      body: file,
    }).then(read),
  applyPreset: (id: string) =>
    fetch('/api/preset/apply', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id }),
    }).then(read),
  setMouth: (id: string, mouth: Record<string, [number, number, number]>) =>
    fetch('/api/preset/mouth', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id, mouth }),
    }).then(read),
  calibrate: (id: string) =>
    fetch('/api/calibrate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id }),
    }).then(read),
  setFeel: (feel: Partial<FeelSettings>) =>
    fetch('/api/feel', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(feel),
    }).then(read),
  setMirror: (on: boolean) =>
    fetch('/api/mirror', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ on }),
    }).then(read),
  setMouthPoint: (id: number, patch: { on?: boolean; to?: number | null }) =>
    fetch('/api/mouth-points', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id, ...patch }),
    }).then(read),
  setEyePoint: (id: number, patch: { on?: boolean; to?: number | null }) =>
    fetch('/api/eye-points', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id, ...patch }),
    }).then(read),
  setSkeletonPoint: (id: number, x: number, y: number) =>
    fetch('/api/skeleton-point', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id, x, y }),
    }).then(read),
  setPoint: (id: number, x: number, y: number) =>
    fetch('/api/overlay-point', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id, x, y }),
    }).then(read),
  resetPoints: (id?: number) =>
    fetch('/api/overlay-points/reset', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(id == null ? {} : { id }),
    }).then(read),
}

export function frameUrl(kind: 'source' | 'overlay' | 'camera' | 'gen', bust: number): string {
  return `/api/frame/${kind}?t=${bust}`
}
