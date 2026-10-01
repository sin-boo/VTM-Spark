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
  hair_width: number
  gaze_gain: number
  gaze_smooth: number
  head_sway: number
  body_turn: number
  max_yaw_left?: number
  max_yaw_right?: number
  max_roll_left?: number
  max_roll_right?: number
  max_pitch_up?: number
  max_pitch_down?: number
  max_size?: number
  max_look_x?: number
  max_look_y?: number
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

export type TravelRects = {
  head?: number[] | null
  head_wall?: number[] | null
  /** Oval head wall outline, [x, y] per point; drawn instead of head_wall. */
  head_oval?: number[][] | null
  body?: number[] | null
  body_wall?: number[] | null
}

export type HairPart = {
  class: string
  polygon: number[][]
  side?: 'l' | 'r' | 'mid'
  width?: number
  pin?: number
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
  mids?: string[]
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
  travel_box?: TravelBox
  travel_rects?: TravelRects
  mouth_points?: { id: number; on: boolean; ring: 'in' | 'out'; to: number | null }[]
  eye_points?: { id: number; on: boolean; side: 'l' | 'r'; artificial?: boolean; to: number | null }[]
  hair?: HairPart[]
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
  recording?: boolean
  record_frames?: number
  record_seconds?: number
  record_path?: string
  record_error?: string
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
  // Frame packets (/api/live once a frame exists) leave this out.
  camera_index?: number
  source?: 'camera' | 'ifm'
  ifm?: IfmStatus
  calib?: CalibStatus
  mouth_points?: { id: number; on: boolean; ring: 'in' | 'out'; to: number | null }[]
  eye_points?: { id: number; on: boolean; side: 'l' | 'r'; artificial?: boolean; to: number | null }[]
  hair?: HairPart[]
  skeleton?: { id: number; x: number; y: number; score: number }[]
  iris?: { id: number; x: number; y: number; score: number; visible?: boolean }[]
  iris_method?: string
  iris_cam?: IrisCamHit[]
  look?: { x: number; y: number } | null
  feel?: FeelSettings
  travel_box?: TravelBox
  travel_rects?: TravelRects
  point_offsets?: { id: number; dx: number; dy: number }[]
  recording?: boolean
  record_frames?: number
  record_seconds?: number
  record_path?: string
  record_error?: string
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
    hair?: HairPart[]
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
  refreshCameras: () => fetch('/api/cameras/refresh', { method: 'POST' }).then(read),
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
  moveMouth: (id: string, t: number) =>
    fetch('/api/preset/move', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id, t }),
    }).then(read),
  dropMouth: (id: string) =>
    fetch('/api/preset/drop', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id }),
    }).then(read),
  record: (on: boolean) =>
    fetch('/api/record', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ on }),
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
  setTravel: (travel: Partial<TravelBox>) =>
    fetch('/api/travel', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(travel),
    }).then(read),
  fitTravel: () => fetch('/api/travel/fit', { method: 'POST' }).then(read),
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
