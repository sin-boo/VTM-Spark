import { useEffect, useRef, useState, type JSX } from 'react'
import { api, frameUrl, type FeelSettings, type LabStatus, type TravelBox } from './api'
import { Side } from './components/Side'
import { DEFAULT_PRESETS, ZERO_FEEL, ZERO_WEIGHTS, type Busy } from './constants'
import { camRefOf, camShortOf, blendMouth, draftMouth, keyId, keyT, keysOn, pairId, sampleMouth, refOf } from './points'

type View = { scale: number; x: number; y: number }

const MIN_SCALE = 0.25
const MAX_SCALE = 16
const CAM_MIN = 1
const CAM_MAX = 12
const CAM_START = 2.2
const PAD = 12
const AUTO_GEN_KEY = 'track-lab-auto-gen'
const MOUTH = [20, 21, 22, 23, 24, 25, 26, 27]
const MOUTH_SET = new Set(MOUTH)
const HAIR_FILL: Record<string, string> = {
  hair_middle: 'rgba(255, 200, 0, 0.28)',
  hair_left: 'rgba(0, 180, 255, 0.28)',
  hair_right: 'rgba(255, 80, 160, 0.28)',
}
const SKELETON_BONES: [number, number][] = [
  [31, 32],
  [32, 33],
  [31, 34],
  [34, 35],
  [32, 34],
  [31, 36],
]
const LINES: { ids: number[]; color: string }[] = [
  { ids: [0, 1, 2, 3, 4], color: 'rgba(80, 200, 255, 0.45)' },
  { ids: [5, 6, 7], color: 'rgba(80, 255, 160, 0.45)' },
  { ids: [8, 9, 10], color: 'rgba(80, 255, 160, 0.45)' },
  { ids: [11, 12, 13], color: 'rgba(255, 120, 80, 0.7)' },
  { ids: [17, 18, 19], color: 'rgba(255, 120, 80, 0.7)' },
  { ids: [14, 16], color: 'rgba(255, 180, 80, 0.45)' },
  { ids: [23, 20, 21, 22, 26], color: 'rgba(180, 80, 255, 0.95)' },
  { ids: [23, 24, 25, 27, 26], color: 'rgba(180, 80, 255, 0.95)' },
]

function pinFace(pts: number[][] | undefined, slot: number | null, held: number[][] | undefined) {
  if (slot == null || !pts?.length || !held?.[slot]) return pts
  return pts.map((row, i) => (i === slot ? held[slot].slice() : row.slice()))
}

function pinRows<T extends { id: number; x: number; y: number }>(
  rows: T[] | undefined,
  held: { id: number; x: number; y: number } | null,
) {
  if (!rows || !held) return rows
  return rows.map((row) => (row.id === held.id ? { ...row, x: held.x, y: held.y } : row))
}

function mouthMap(pts: number[][]): Record<string, [number, number, number]> {
  const out: Record<string, [number, number, number]> = {}
  for (const i of MOUTH) {
    const row = pts[i]
    if (!row) continue
    out[String(i)] = [row[0], row[1], row[2] ?? 1]
  }
  return out
}

function mouthMesh(
  id: string,
  shapes: Record<string, number[][]> | undefined,
  rest: number[][] | null,
): number[][] | null {
  const stored = shapes?.[id]
  if (stored && stored.length >= 28) return stored.map((row) => row.slice())
  const ends = id.split('+')
  if (ends.length === 2 && pairId(ends[0], ends[1]) === id) {
    const left = mouthMesh(ends[0], shapes, rest)
    const right = mouthMesh(ends[1], shapes, rest)
    if (!left || !right) return null
    return blendMouth(left, right)
  }
  if (!rest || rest.length < 28) return null
  if (id === 'rest') return rest.map((row) => row.slice())
  return draftMouth(id, rest)
}

function pasteMouthOnto(base: number[][], clip: number[][]): number[][] {
  const out = base.map((row) => row.slice())
  for (const i of MOUTH) {
    const row = clip[i]
    if (!row) continue
    out[i] = row.slice()
  }
  return out
}

function localXY(el: HTMLElement, clientX: number, clientY: number) {
  const box = el.getBoundingClientRect()
  return {
    x: clientX - box.left - el.clientLeft,
    y: clientY - box.top - el.clientTop,
  }
}

function project(
  ix: number,
  iy: number,
  wellW: number,
  wellH: number,
  srcW: number,
  srcH: number,
  natW: number,
  natH: number,
  view: View,
) {
  const innerW = wellW - PAD * 2
  const innerH = wellH - PAD * 2
  const fit = Math.min(innerW / natW, innerH / natH)
  const px = PAD + (innerW - natW * fit) / 2 + (ix / srcW) * natW * fit
  const py = PAD + (innerH - natH * fit) / 2 + (iy / srcH) * natH * fit
  const cx = PAD + innerW / 2
  const cy = PAD + innerH / 2
  return {
    x: cx + view.x + (px - cx) * view.scale,
    y: cy + view.y + (py - cy) * view.scale,
  }
}

function unproject(
  sx: number,
  sy: number,
  wellW: number,
  wellH: number,
  srcW: number,
  srcH: number,
  natW: number,
  natH: number,
  view: View,
) {
  const innerW = wellW - PAD * 2
  const innerH = wellH - PAD * 2
  const fit = Math.min(innerW / natW, innerH / natH)
  const ox = PAD + (innerW - natW * fit) / 2
  const oy = PAD + (innerH - natH * fit) / 2
  const cx = PAD + innerW / 2
  const cy = PAD + innerH / 2
  const px = cx + (sx - cx - view.x) / view.scale
  const py = cy + (sy - cy - view.y) / view.scale
  return {
    x: ((px - ox) / (natW * fit)) * srcW,
    y: ((py - oy) / (natH * fit)) * srcH,
  }
}

export default function App() {
  const [status, setStatus] = useState<LabStatus | null>(null)
  const [busy, setBusy] = useState<Busy>('')
  const [error, setError] = useState('')
  const [bust, setBust] = useState(0)
  const [genBust, setGenBust] = useState(0)
  const [showGen, setShowGen] = useState(false)
  const [autoGen, setAutoGen] = useState(() => {
    try {
      return localStorage.getItem(AUTO_GEN_KEY) === '1'
    } catch {
      return false
    }
  })
  const [showOverlay, setShowOverlay] = useState(false)
  const [drag, setDrag] = useState(false)
  const [view, setView] = useState<View>({ scale: 1, x: 0, y: 0 })
  const [panning, setPanning] = useState(false)
  const [selected, setSelected] = useState('rest')
  const [panel, setPanel] = useState<'desk' | 'limiters' | 'blend'>('desk')
  const [menuOpen, setMenuOpen] = useState(false)
  const menuRef = useRef<HTMLDivElement>(null)
  const [travelFocus, setTravelFocus] = useState<'head' | 'body' | null>(null)
  const [points, setPoints] = useState<number[][]>([])
  const [clip, setClip] = useState<number[][] | null>(null)
  const [menu, setMenu] = useState<{ id: string; x: number; y: number } | null>(null)
  const [sourceMenu, setSourceMenu] = useState(false)
  const [nat, setNat] = useState({ w: 0, h: 0 })
  const [wellBox, setWellBox] = useState({ w: 1, h: 1 })
  const [camBust, setCamBust] = useState(0)
  const camBusy = useRef(false)
  const lastCamAt = useRef(0)
  const [camZoom, setCamZoom] = useState(false)
  const [ifmPort, setIfmPort] = useState('49983')
  const [copied, setCopied] = useState('')
  const ifmSynced = useRef(false)
  const [camView, setCamView] = useState<View>({ scale: CAM_START, x: 0, y: 0 })
  const [lipTo, setLipTo] = useState<Record<number, string>>({})
  const [eyeTo, setEyeTo] = useState<Record<number, string>>({})
  const camViewRef = useRef(camView)
  const camStage = useRef<HTMLDivElement>(null)
  const camPan = useRef<{ mx: number; my: number; x: number; y: number } | null>(null)
  const camDragged = useRef(false)
  const picker = useRef<HTMLInputElement>(null)
  const well = useRef<HTMLElement>(null)
  const viewRef = useRef(view)
  const hasSource = useRef(false)
  const dragPt = useRef<number | null>(null)
  const [dragging, setDragging] = useState(false)
  const skeletonDrag = useRef<{ id: number; x: number; y: number } | null>(null)
  const [draggingSkeleton, setDraggingSkeleton] = useState(false)
  const irisDrag = useRef<{ id: number; x: number; y: number } | null>(null)
  const [draggingIris, setDraggingIris] = useState(false)
  const pointsRef = useRef(points)
  const statusRef = useRef(status)
  const natRef = useRef(nat)
  const sourceNat = useRef({ w: 0, h: 0 })
  const autoGenRef = useRef(autoGen)
  const busyRef = useRef(busy)
  const genBusy = useRef(false)
  const genQueued = useRef(false)
  const runGenRef = useRef<() => Promise<void>>(async () => {})
  const wellBoxRef = useRef(wellBox)
  const pan = useRef<{ mx: number; my: number; x: number; y: number } | null>(null)

  const putView = (next: View) => {
    const scale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, next.scale))
    const clamped = { scale, x: next.x, y: next.y }
    viewRef.current = clamped
    setView(clamped)
  }

  const resetView = () => putView({ scale: 1, x: 0, y: 0 })

  const putCamView = (next: View) => {
    const scale = Math.min(CAM_MAX, Math.max(CAM_MIN, next.scale))
    const clamped = { scale, x: next.x, y: next.y }
    camViewRef.current = clamped
    setCamView(clamped)
  }

  const apply = (next: LabStatus, overlay?: boolean, refreshFrame = true, keepPoints = false) => {
    setStatus((s) => {
      const merged = s ? { ...s, ...next } : next
      return {
        ...merged,
        skeleton: pinRows(merged.skeleton, skeletonDrag.current) ?? merged.skeleton,
        iris: pinRows(merged.iris, irisDrag.current) ?? merged.iris,
        points: keepPoints && s?.points?.length ? s.points : merged.points,
      }
    })
    setError(next.error || '')
    if (refreshFrame) setBust(Date.now())
    if (!keepPoints) {
      setShowOverlay(overlay ?? next.faces > 0)
      if (next.active) setSelected(next.active)
      if (next.points?.length) {
        const pinned = pinFace(next.points, dragPt.current, pointsRef.current)
        if (pinned?.length) setPoints(pinned)
      }
    }
  }

  useEffect(() => {
    let alive = true
    const tick = async () => {
      try {
        const s = await api.status()
        if (!alive) return
        apply(s)
        if (!s.ready && !s.live) window.setTimeout(() => void tick(), 250)
      } catch (e) {
        if (alive) setError(String(e))
      }
    }
    void tick()
    return () => {
      alive = false
    }
  }, [])

  useEffect(() => {
    if (!status?.live) return
    let alive = true
    let seq = 0
    let applied = 0
    let inflight = 0
    const tick = async () => {
      if (inflight >= 2) return
      const mine = ++seq
      inflight += 1
      try {
        const next = await api.live()
        if (!alive || mine <= applied) return
        applied = mine
        const camNow = Date.now()
        if (camNow - lastCamAt.current >= 40) {
          lastCamAt.current = camNow
          camBusy.current = false
          setCamBust(camNow)
        }
        setStatus((s) =>
          s
            ? {
                ...s,
                live: next.live,
                tracker: next.tracker,
                faces: next.faces,
                ms: next.ms,
                error: next.error,
                weights: next.weights,
                head: next.head,
                blink: next.blink,
                camera_index: next.camera_index,
                source: next.source ?? s.source,
                ifm: next.ifm
                  ? {
                      ...next.ifm,
                      host: s.ifm?.host ?? next.ifm.host,
                      port: s.ifm?.port ?? next.ifm.port,
                    }
                  : s.ifm,
                calib: next.calib ?? s.calib,
                mouth_points: next.mouth_points ?? s.mouth_points,
                eye_points: next.eye_points ?? s.eye_points,
                hair: next.hair?.length ? next.hair : s.hair,
                skeleton: pinRows(
                  next.skeleton?.length ? next.skeleton : s.skeleton,
                  skeletonDrag.current,
                ),
                iris: pinRows(next.iris ?? s.iris, irisDrag.current),
                iris_cam: next.iris_cam ?? s.iris_cam,
                look: next.look ?? s.look,
                points: pinFace(
                  next.points?.length ? next.points : s.points,
                  dragPt.current,
                  pointsRef.current,
                ) ?? (next.points?.length ? next.points : s.points),
                iris_method: next.iris_method ?? s.iris_method,
                point_offsets: next.point_offsets ?? s.point_offsets,
                feel: next.feel ?? s.feel,
                recording: next.recording ?? s.recording,
                record_frames: next.record_frames ?? s.record_frames,
                record_seconds: next.record_seconds ?? s.record_seconds,
                record_path: next.recording ? next.record_path || '' : next.record_path || s.record_path,
                record_error: next.record_error ?? s.record_error,
              }
            : s,
        )
        if (next.points?.length) {
          const pinned = pinFace(next.points, dragPt.current, pointsRef.current)
          if (pinned?.length) setPoints(pinned)
        }
        setError(next.error || '')
      } catch (e) {
        if (alive) setError(String(e))
      } finally {
        inflight -= 1
      }
    }
    void tick()
    const id = window.setInterval(() => void tick(), 32)
    return () => {
      alive = false
      window.clearInterval(id)
    }
  }, [status?.live])

  useEffect(() => {
    if (!menu) return
    const close = () => setMenu(null)
    window.addEventListener('click', close)
    window.addEventListener('scroll', close, true)
    return () => {
      window.removeEventListener('click', close)
      window.removeEventListener('scroll', close, true)
    }
  }, [menu])

  useEffect(() => {
    if (!sourceMenu) return
    const close = () => setSourceMenu(false)
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setSourceMenu(false)
    }
    window.addEventListener('click', close)
    window.addEventListener('keydown', onKey)
    return () => {
      window.removeEventListener('click', close)
      window.removeEventListener('keydown', onKey)
    }
  }, [sourceMenu])

  useEffect(() => {
    if (ifmSynced.current || !status?.ifm) return
    setIfmPort(String(status.ifm.port || 49983))
    ifmSynced.current = true
  }, [status?.ifm])

  useEffect(() => {
    if (!status?.live) setCamZoom(false)
  }, [status?.live])

  useEffect(() => {
    if (!camZoom) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setCamZoom(false)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [camZoom])

  useEffect(() => {
    const el = camStage.current
    if (!camZoom || !el) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      e.stopPropagation()
      const box = el.getBoundingClientRect()
      const cur = camViewRef.current
      const factor = e.deltaY < 0 ? 1.14 : 1 / 1.14
      const scale = Math.min(CAM_MAX, Math.max(CAM_MIN, cur.scale * factor))
      const mx = e.clientX - box.left
      const my = e.clientY - box.top
      const cx = box.width / 2 + cur.x
      const cy = box.height / 2 + cur.y
      const grow = scale / cur.scale
      putCamView({
        scale,
        x: mx - (mx - cx) * grow - box.width / 2,
        y: my - (my - cy) * grow - box.height / 2,
      })
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [camZoom])

  hasSource.current = Boolean(status?.has_source)
  pointsRef.current = points
  statusRef.current = status
  natRef.current = nat
  wellBoxRef.current = wellBox
  autoGenRef.current = autoGen
  busyRef.current = busy

  useEffect(() => {
    const el = well.current
    if (!el) return
    const onWheel = (e: WheelEvent) => {
      if (!hasSource.current) return
      e.preventDefault()
      const box = el.getBoundingClientRect()
      const cur = viewRef.current
      const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12
      const scale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, cur.scale * factor))
      const mx = e.clientX - box.left
      const my = e.clientY - box.top
      const cx = box.width / 2 + cur.x
      const cy = box.height / 2 + cur.y
      const grow = scale / cur.scale
      putView({
        scale,
        x: mx - (mx - cx) * grow - box.width / 2,
        y: my - (my - cy) * grow - box.height / 2,
      })
    }
    const onMouseDown = (e: MouseEvent) => {
      if (e.button === 1) e.preventDefault()
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    el.addEventListener('mousedown', onMouseDown)
    const ro = new ResizeObserver(() => {
      setWellBox({ w: el.clientWidth, h: el.clientHeight })
    })
    ro.observe(el)
    setWellBox({ w: el.clientWidth, h: el.clientHeight })
    return () => {
      el.removeEventListener('wheel', onWheel)
      el.removeEventListener('mousedown', onMouseDown)
      ro.disconnect()
    }
  }, [])

  const run = async (kind: 'track' | 'reset') => {
    setBusy(kind)
    setError('')
    try {
      const next = kind === 'track' ? await api.track() : await api.reset()
      apply(next, kind === 'track' ? next.faces > 0 : Boolean(next.points?.length))
      if (kind === 'reset') {
        setSelected(next.active || 'rest')
        setShowGen(false)
      }
    } catch (e) {
      setError(String(e))
    } finally {
      setBusy('')
    }
  }

  const runRecord = async () => {
    setError('')
    try {
      const next = await api.record(!status?.recording)
      apply(next, false, false, true)
      if (next.record_error) setError(next.record_error)
      else if (next.record_path) setError('')
    } catch (e) {
      setError(String(e))
    }
  }

  const runGen = async () => {
    setBusy('gen')
    setError('')
    try {
      const st = statusRef.current
      const next = await api.generate({
        points: pointsRef.current,
        hair: st?.hair,
        skeleton: st?.skeleton,
        iris: st?.iris,
      })
      if (next.error) {
        setError(next.error)
        return
      }
      setStatus((s) =>
        s
          ? { ...s, gen: next.gen, gen_ms: next.gen_ms, message: next.message, error: '' }
          : s,
      )
      setShowGen(true)
      setGenBust((n) => n + 1)
    } catch (e) {
      setError(String(e))
    } finally {
      setBusy('')
    }
  }
  runGenRef.current = runGen

  const requestGen = () => {
    if (statusRef.current?.live || pointsRef.current.length < 28) return
    if (genBusy.current || busyRef.current === 'gen') {
      genQueued.current = true
      return
    }
    genBusy.current = true
    void runGenRef.current().finally(() => {
      genBusy.current = false
      if (!genQueued.current) return
      genQueued.current = false
      requestGen()
    })
  }

  const afterPointCommit = (next: LabStatus) => {
    apply(next, undefined, false, true)
    if (autoGenRef.current) requestGen()
  }

  const putAuto = (on: boolean) => {
    autoGenRef.current = on
    setAutoGen(on)
    try {
      localStorage.setItem(AUTO_GEN_KEY, on ? '1' : '0')
    } catch {
      /* ignore */
    }
  }

  const runOsf = async () => {
    setBusy('osf')
    setError('')
    try {
      const next = status?.live
        ? await api.osfStop()
        : await api.osfStart({
            camera: status?.camera_index ?? 0,
            source: status?.source === 'ifm' ? 'ifm' : 'camera',
            ...(status?.source === 'ifm'
              ? {
                  port: Number(ifmPort) || 49983,
                  ...(status.ifm?.host?.trim() ? { host: status.ifm.host.trim() } : {}),
                }
              : {}),
          })
      apply(next, true, false)
    } catch (e) {
      setError(String(e))
    } finally {
      setBusy('')
    }
  }

  useEffect(() => {
    if (!menuOpen) return
    const close = (e: PointerEvent) => {
      if (!menuRef.current?.contains(e.target as Node)) setMenuOpen(false)
    }
    window.addEventListener('pointerdown', close)
    return () => window.removeEventListener('pointerdown', close)
  }, [menuOpen])

  const openPanel = (next: 'camera' | 'ifm' | 'limiters' | 'blend') => {
    setMenuOpen(false)
    if (next === 'limiters' || next === 'blend') {
      setPanel(next)
      return
    }
    setPanel('desk')
    void setInput(next)
  }

  const setInput = async (next: 'camera' | 'ifm') => {
    if ((status?.source ?? 'camera') === next) return
    setStatus((s) => (s ? { ...s, source: next } : s))
    setError('')
    try {
      apply(await api.setInput(next), true, false)
    } catch (e) {
      setError(String(e))
    }
  }

  const saveIfm = () => {
    const port = Number(ifmPort) || 49983
    setIfmPort(String(port))
    void api
      .setIfm(status?.ifm?.host?.trim() ?? '', port)
      .then((next) => apply(next, true, false))
      .catch((e) => setError(String(e)))
  }

  const loadFile = async (file: File | undefined) => {
    if (!file) return
    setBusy('load')
    setError('')
    try {
      apply(await api.upload(file), false)
      setPoints([])
      setSelected('rest')
      setShowGen(false)
      resetView()
    } catch (e) {
      setError(String(e))
    } finally {
      setBusy('')
    }
  }

  const previewPreset = (id: string) => {
    setShowGen(false)
    setSelected(id)
    const shapes = statusRef.current?.shapes
    const rest =
      shapes?.rest && shapes.rest.length >= 28
        ? shapes.rest
        : pointsRef.current.length >= 28
          ? pointsRef.current
          : null
    const src = mouthMesh(id, shapes, rest)
    if (!src) return
    const next = src.map((row) => row.slice())
    pointsRef.current = next
    setPoints(next)
  }

  const pairMesh = (a: string, b: string, t: number, skip?: string) => {
    const shapes = statusRef.current?.shapes
    const rest =
      shapes?.rest && shapes.rest.length >= 28
        ? shapes.rest
        : pointsRef.current.length >= 28
          ? pointsRef.current
          : null
    const left = mouthMesh(a, shapes, rest)
    const right = mouthMesh(b, shapes, rest)
    if (!left || !right) return null
    const stops = keysOn(Object.keys(shapes ?? {}), a, b)
      .filter((key) => key.id !== skip)
      .map((key) => {
        const pts = shapes?.[key.id]
        return pts && pts.length >= 28 ? { t: key.t, pts } : null
      })
      .filter((key): key is { t: number; pts: number[][] } => key != null)
    return sampleMouth(left, right, stops, t)
  }

  const scrubPair = (a: string, b: string, t: number) => {
    const mesh = pairMesh(a, b, t)
    if (!mesh) return
    setSelected('')
    const next = mesh.map((row) => row.slice())
    pointsRef.current = next
    setPoints(next)
  }

  const addMid = async (a: string, b: string, t = 0.5) => {
    const id = keyId(a, b, t)
    if (!id) return
    const shapes = statusRef.current?.shapes
    if ((shapes?.[id]?.length ?? 0) >= 28) {
      previewPreset(id)
      return
    }
    const sampled = pairMesh(a, b, t)
    const mid =
      selected === '' && pointsRef.current.length >= 28 ? pointsRef.current.map((row) => row.slice()) : sampled
    if (!mid) return
    setSelected(id)
    setBusy('apply')
    setError('')
    try {
      const next = await api.setMouth(id, mouthMap(mid))
      if (next.error) {
        setError(next.error)
        return
      }
      const saved = next.shapes?.[id]?.length >= 28 ? next.shapes[id].map((row) => row.slice()) : mid
      apply(next, true, false, true)
      pointsRef.current = saved
      setPoints(saved)
    } catch (e) {
      setError(String(e))
    } finally {
      setBusy('')
    }
  }

  const moveKey = async (id: string, t: number) => {
    const at = id.lastIndexOf('@')
    const base = at < 0 ? id : id.slice(0, at)
    const parts = base.split('+')
    if (parts.length !== 2) return
    const nextId = keyId(parts[0], parts[1], t)
    if (!nextId || nextId === id) return
    setError('')
    try {
      const next = await api.moveMouth(id, t)
      if (next.error) {
        setError(next.error)
        return
      }
      apply(next, true, false, true)
      setSelected((cur) => (cur === id ? nextId : cur))
    } catch (e) {
      setError(String(e))
    }
  }

  const resetKey = async (id: string) => {
    const t = keyT(id)
    const base = id.split('@')[0]
    const parts = base.split('+')
    if (t == null || parts.length !== 2) return
    const mesh = pairMesh(parts[0], parts[1], t, id)
    if (!mesh) return
    setError('')
    try {
      const next = await api.setMouth(id, mouthMap(mesh))
      if (next.error) {
        setError(next.error)
        return
      }
      const saved = next.shapes?.[id]?.length >= 28 ? next.shapes[id].map((row) => row.slice()) : mesh
      apply(next, true, false, true)
      if (selected === id) {
        pointsRef.current = saved
        setPoints(saved)
      }
    } catch (e) {
      setError(String(e))
    }
  }

  const deleteKey = async (id: string) => {
    const viewing = selected === id
    setError('')
    try {
      const next = await api.dropMouth(id)
      if (next.error) {
        setError(next.error)
        return
      }
      apply(next, true, false, !viewing)
    } catch (e) {
      setError(String(e))
    }
  }

  const commitPreset = async (id = selected) => {
    if (!id) return
    setSelected(id)
    setBusy('apply')
    setError('')
    try {
      const authored = pointsRef.current.map((row) => row.slice())
      const next = await api.setMouth(id, mouthMap(authored))
      if (next.error) {
        setError(next.error)
        return
      }
      const saved =
        next.shapes?.[id]?.length >= 28
          ? next.shapes[id].map((row) => row.slice())
          : authored
      apply(next, true, false, true)
      pointsRef.current = saved
      setPoints(saved)
    } catch (e) {
      setError(String(e))
    } finally {
      setBusy('')
    }
  }

  const putMouthPoint = async (id: number, patch: { on?: boolean; to?: number | null }) => {
    try {
      apply(await api.setMouthPoint(id, patch), undefined, false)
    } catch (e) {
      setError(String(e))
    }
  }

  const putEyePoint = async (id: number, patch: { on?: boolean; to?: number | null }) => {
    try {
      apply(await api.setEyePoint(id, patch), undefined, false)
    } catch (e) {
      setError(String(e))
    }
  }

  const commitMap = (
    id: number,
    raw: string,
    setDraft: (fn: (cur: Record<number, string>) => Record<number, string>) => void,
    put: (id: number, patch: { to: number | null }) => void,
  ) => {
    const digits = raw.replace(/\D/g, '')
    setDraft((cur) => {
      const next = { ...cur }
      delete next[id]
      return next
    })
    if (digits === '') {
      put(id, { to: null })
      return
    }
    const slot = Number(digits)
    if (slot < 0 || slot > 27) return
    put(id, { to: slot })
  }

  const pointsFor = (id: string) => {
    if (id === selected && points.length >= 28) return points
    const stored = status?.shapes?.[id]
    return stored && stored.length >= 28 ? stored : null
  }

  const copyPreset = (id: string) => {
    const src = pointsFor(id)
    if (!src) {
      setError(`Nothing to copy from ${id}`)
      return
    }
    setClip(src.map((row) => row.slice()))
    setError('')
    setMenu(null)
  }

  const pastePreset = async (id: string) => {
    if (!clip) return
    setMenu(null)
    setBusy('apply')
    setError('')
    try {
      const next = await api.setMouth(id, mouthMap(clip))
      if (next.error) {
        setError(next.error)
        return
      }
      const shapes = statusRef.current?.shapes ?? {}
      const base =
        (next.shapes?.[id] && next.shapes[id].length >= 28 && next.shapes[id]) ||
        (shapes[id] && shapes[id].length >= 28 && shapes[id]) ||
        (shapes.rest && shapes.rest.length >= 28 && shapes.rest) ||
        clip
      const pasted = next.shapes?.[id]?.length >= 28 ? next.shapes[id].map((row) => row.slice()) : pasteMouthOnto(base, clip)
      apply(next, true, false)
      setSelected(id)
      setStatus((s) => {
        const merged = s ? { ...s, ...next } : next
        const shapesNext = { ...(merged.shapes ?? {}), [id]: pasted }
        const out = { ...merged, active: id, shapes: shapesNext }
        statusRef.current = out
        return out
      })
      pointsRef.current = pasted
      setPoints(pasted.map((row) => row.slice()))
    } catch (e) {
      setError(String(e))
    } finally {
      setBusy('')
    }
  }

  const srcW = status?.width || 0
  const srcH = status?.height || 0
  const live = Boolean(status?.live)
  const source = status?.source === 'ifm' ? 'ifm' : 'camera'
  const ifm = status?.ifm
  const localIps = ifm?.local?.length ? ifm.local : ifm?.primary ? [ifm.primary] : []

  const copyAddr = async (ip: string) => {
    try {
      await navigator.clipboard.writeText(ip)
      setCopied(ip)
      window.setTimeout(() => setCopied((cur) => (cur === ip ? '' : cur)), 1400)
    } catch {
      setError(`Could not copy ${ip}`)
    }
  }

  const setCamera = async (index: number) => {
    setBusy('cam')
    setError('')
    try {
      apply(await api.setCamera(index), true, false)
    } catch (e) {
      setError(String(e))
    } finally {
      setBusy('')
    }
  }

  const imageKind = showGen ? 'gen' : showOverlay && !points.length ? 'overlay' : 'source'
  const imageBust = showGen ? genBust : bust
  const weights = status?.weights ?? ZERO_WEIGHTS
  const feelVals = status?.feel ?? ZERO_FEEL
  const showFace = (feelVals.show_face ?? 1) >= 0.5
  const showSkeleton = (feelVals.show_skeleton ?? 1) >= 0.5
  const showHair = (feelVals.show_hair ?? 1) >= 0.5
  const showIds = (feelVals.show_ids ?? 0) >= 0.5
  const presets = status?.presets ?? DEFAULT_PRESETS
  const toScreen = (ix: number, iy: number) =>
    project(ix, iy, wellBox.w, wellBox.h, srcW, srcH, nat.w, nat.h, view)
  const showMesh =
    (points.length >= 28 ||
      (status?.hair?.length ?? 0) > 0 ||
      (status?.skeleton?.length ?? 0) > 0 ||
      (status?.iris?.length ?? 0) > 0) &&
    srcW > 0 &&
    nat.w > 0

  const movePoint = (slot: number, clientX: number, clientY: number) => {
    const el = well.current
    const st = statusRef.current
    const n = natRef.current
    const box = wellBoxRef.current
    if (!el || !st?.width || !n.w) return
    const local = localXY(el, clientX, clientY)
    const img = unproject(local.x, local.y, box.w, box.h, st.width, st.height, n.w, n.h, viewRef.current)
    const next = pointsRef.current.map((row, i) => (i === slot ? [img.x, img.y, row[2] ?? 1] : row.slice()))
    pointsRef.current = next
    setPoints(next)
  }

  const moveSkeletonPoint = (id: number, clientX: number, clientY: number) => {
    const el = well.current
    const st = statusRef.current
    const n = natRef.current
    const box = wellBoxRef.current
    if (!el || !st?.width || !n.w) return
    const local = localXY(el, clientX, clientY)
    const img = unproject(local.x, local.y, box.w, box.h, st.width, st.height, n.w, n.h, viewRef.current)
    skeletonDrag.current = { id, x: img.x, y: img.y }
    setStatus((current) =>
      current
        ? {
            ...current,
            skeleton: (current.skeleton ?? []).map((joint) =>
              joint.id === id ? { ...joint, x: img.x, y: img.y } : joint,
            ),
          }
        : current,
    )
  }

  useEffect(() => {
    if (!dragging) return
    const onMove = (e: PointerEvent) => {
      const slot = dragPt.current
      if (slot == null) return
      e.preventDefault()
      movePoint(slot, e.clientX, e.clientY)
    }
    const onUp = () => {
      const slot = dragPt.current
      const row = slot == null ? null : pointsRef.current[slot]
      setDragging(false)
      if (slot == null || !row) {
        dragPt.current = null
        return
      }
      void api
        .setPoint(slot, row[0], row[1])
        .then((next) => {
          dragPt.current = null
          afterPointCommit(next)
        })
        .catch((e) => {
          dragPt.current = null
          setError(String(e))
        })
    }
    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', onUp)
    window.addEventListener('pointercancel', onUp)
    return () => {
      window.removeEventListener('pointermove', onMove)
      window.removeEventListener('pointerup', onUp)
      window.removeEventListener('pointercancel', onUp)
    }
  }, [dragging])

  useEffect(() => {
    if (!draggingSkeleton) return
    const onMove = (e: PointerEvent) => {
      const point = skeletonDrag.current
      if (!point) return
      e.preventDefault()
      moveSkeletonPoint(point.id, e.clientX, e.clientY)
    }
    const onUp = () => {
      const point = skeletonDrag.current
      setDraggingSkeleton(false)
      if (!point) {
        skeletonDrag.current = null
        return
      }
      const tracking = Boolean(statusRef.current?.live)
      const save = tracking
        ? api.setPoint(point.id, point.x, point.y)
        : api.setSkeletonPoint(point.id, point.x, point.y)
      void save
        .then((next) => {
          skeletonDrag.current = null
          afterPointCommit(next)
        })
        .catch((e) => {
          skeletonDrag.current = null
          setError(String(e))
        })
    }
    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', onUp)
    window.addEventListener('pointercancel', onUp)
    return () => {
      window.removeEventListener('pointermove', onMove)
      window.removeEventListener('pointerup', onUp)
      window.removeEventListener('pointercancel', onUp)
    }
  }, [draggingSkeleton])

  const moveIrisPoint = (id: number, clientX: number, clientY: number) => {
    const el = well.current
    const st = statusRef.current
    const n = natRef.current
    const box = wellBoxRef.current
    if (!el || !st?.width || !n.w) return
    const local = localXY(el, clientX, clientY)
    const img = unproject(local.x, local.y, box.w, box.h, st.width, st.height, n.w, n.h, viewRef.current)
    irisDrag.current = { id, x: img.x, y: img.y }
    setStatus((current) =>
      current
        ? {
            ...current,
            iris: (current.iris ?? []).map((row) =>
              row.id === id ? { ...row, x: img.x, y: img.y } : row,
            ),
          }
        : current,
    )
  }

  useEffect(() => {
    if (!draggingIris) return
    const onMove = (e: PointerEvent) => {
      const point = irisDrag.current
      if (!point) return
      e.preventDefault()
      moveIrisPoint(point.id, e.clientX, e.clientY)
    }
    const onUp = () => {
      const point = irisDrag.current
      setDraggingIris(false)
      if (!point) {
        irisDrag.current = null
        return
      }
      void api
        .setPoint(point.id, point.x, point.y)
        .then((next) => {
          irisDrag.current = null
          afterPointCommit(next)
        })
        .catch((e) => {
          irisDrag.current = null
          setError(String(e))
        })
    }
    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', onUp)
    window.addEventListener('pointercancel', onUp)
    return () => {
      window.removeEventListener('pointermove', onMove)
      window.removeEventListener('pointerup', onUp)
      window.removeEventListener('pointercancel', onUp)
    }
  }, [draggingIris])

  const runCalibrate = async (id: string) => {
    setBusy('cal')
    setError('')
    try {
      apply(await api.calibrate(id), true, false)
    } catch (e) {
      setError(String(e))
    } finally {
      setBusy('')
    }
  }

  const putFeel = (key: keyof FeelSettings, value: number) => {
    const next = { ...feelVals, [key]: value }
    setStatus((s) => (s ? { ...s, feel: next } : s))
    void api.setFeel({ [key]: value }).catch((e) => setError(String(e)))
  }

  const putTravel = (patch: Partial<TravelBox>) => {
    setStatus((s) =>
      s
        ? {
            ...s,
            travel_box: { ...(s.travel_box ?? {}), ...patch } as TravelBox,
          }
        : s,
    )
    void api
      .setTravel(patch)
      .then((next) => apply(next, false, false))
      .catch((e) => setError(String(e)))
  }

  const putMirror = (on: boolean) => {
    setStatus((s) => (s ? { ...s, mirror: on } : s))
    void api.setMirror(on).then((next) => apply(next, false, false)).catch((e) => setError(String(e)))
  }

  const travelOn = status?.travel_box?.enabled !== false
  const travelRects = status?.travel_rects
  const limitRect = (rect: number[] | null | undefined, color: string, dashed: boolean, key: string) => {
    if (!rect || rect.length < 4) return null
    const a = toScreen(rect[0], rect[1])
    const b = toScreen(rect[2], rect[3])
    const x = Math.min(a.x, b.x)
    const y = Math.min(a.y, b.y)
    const w = Math.abs(b.x - a.x)
    const h = Math.abs(b.y - a.y)
    if (w < 1 || h < 1) return null
    return (
      <rect
        key={key}
        x={x}
        y={y}
        width={w}
        height={h}
        fill="none"
        stroke={color}
        strokeWidth={1.5}
        strokeDasharray={dashed ? '6 4' : undefined}
        pointerEvents="none"
      />
    )
  }

  return (
    <div className="bench">
      <header className="mast">
        <p className="eyebrow">track lab</p>
        <h1>Face bench</h1>
        <div className="mast-menu" ref={menuRef}>
          <button
            type="button"
            className="mast-menu-btn"
            aria-expanded={menuOpen}
            onClick={() => setMenuOpen((open) => !open)}
          >
            {panel === 'blend' ? 'Blend' : panel === 'limiters' ? 'Limiters' : source === 'ifm' ? 'iFacialMocap' : 'Camera'}
          </button>
          {menuOpen ? (
            <ul>
              <li>
                <button type="button" className={panel === 'desk' && source === 'camera' ? 'on' : ''} onClick={() => openPanel('camera')}>
                  Camera
                </button>
              </li>
              <li>
                <button type="button" className={panel === 'desk' && source === 'ifm' ? 'on' : ''} onClick={() => openPanel('ifm')}>
                  iFacialMocap
                </button>
              </li>
              <li>
                <button type="button" className={panel === 'limiters' ? 'on' : ''} onClick={() => openPanel('limiters')}>
                  Limiters
                </button>
              </li>
              <li>
                <button type="button" className={panel === 'blend' ? 'on' : ''} onClick={() => openPanel('blend')}>
                  Blend
                </button>
              </li>
            </ul>
          ) : null}
        </div>
      </header>

      <figure
        ref={well}
        className={`well${drag ? ' drag' : ''}${panning ? ' pan' : ''}`}
        onDragOver={(e) => {
          e.preventDefault()
          setDrag(true)
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDrag(false)
          void loadFile(e.dataTransfer.files[0])
        }}
        onClick={() => {
          if (!status?.has_source) picker.current?.click()
        }}
        onPointerDown={(e) => {
          if (!status?.has_source || e.button !== 1) return
          e.preventDefault()
          e.currentTarget.setPointerCapture(e.pointerId)
          pan.current = { mx: e.clientX, my: e.clientY, x: viewRef.current.x, y: viewRef.current.y }
          setPanning(true)
        }}
        onPointerMove={(e) => {
          const start = pan.current
          if (!start) return
          putView({
            scale: viewRef.current.scale,
            x: start.x + (e.clientX - start.mx),
            y: start.y + (e.clientY - start.my),
          })
        }}
        onPointerUp={(e) => {
          if (pan.current) {
            pan.current = null
            setPanning(false)
          }
          if (e.button === 1) e.preventDefault()
        }}
        onPointerCancel={() => {
          pan.current = null
          setPanning(false)
        }}
        onAuxClick={(e) => {
          if (e.button === 1) e.preventDefault()
        }}
      >
        <input
          ref={picker}
          className="pick"
          type="file"
          accept="image/png,image/jpeg,image/webp"
          onChange={(e) => {
            const file = e.target.files?.[0]
            e.target.value = ''
            void loadFile(file)
          }}
        />
        {status?.has_source ? (
          <>
            <img
              src={frameUrl(imageKind, imageBust)}
              alt={showGen ? 'Generated' : 'Tracking still'}
              draggable={false}
              onLoad={(e) => {
                if (imageKind === 'gen') return
                const next = { w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight }
                sourceNat.current = next
                setNat(next)
              }}
              style={{ transform: `translate(${view.x}px, ${view.y}px) scale(${view.scale})` }}
            />
            {genBust > 0 || points.length >= 28 ? (
              <div
                className="well-swaps"
                onClick={(e) => e.stopPropagation()}
                onPointerDown={(e) => e.stopPropagation()}
              >
                {genBust > 0 ? (
                  <button
                    type="button"
                    className={`id-map${showGen ? ' on' : ''}`}
                    onClick={() => {
                      setShowGen((on) => {
                        const next = !on
                        if (!next && sourceNat.current.w) setNat(sourceNat.current)
                        return next
                      })
                    }}
                  >
                    {showGen ? 'Still' : 'Gen'}
                  </button>
                ) : null}
                {points.length >= 28 ? (
                  <button
                    type="button"
                    className={`id-map${autoGen ? ' on' : ''}`}
                    title="Generate whenever a point is dropped"
                    onClick={() => putAuto(!autoGen)}
                  >
                    Auto
                  </button>
                ) : null}
              </div>
            ) : null}
            {live ? (
              <img
                className="pip"
                src={frameUrl('camera', camBust)}
                alt={source === 'ifm' ? 'iFacialMocap' : 'OSF camera'}
                draggable={false}
                title="Right-click to enlarge and zoom"
                onLoad={() => {
                  camBusy.current = false
                }}
                onError={() => {
                  camBusy.current = false
                }}
                onContextMenu={(e) => {
                  e.preventDefault()
                  e.stopPropagation()
                  putCamView({ scale: CAM_START, x: 0, y: 0 })
                  setCamZoom(true)
                }}
                onClick={(e) => e.stopPropagation()}
              />
            ) : null}
            {showMesh ? (
              <svg className={`mesh${live ? ' live' : ''}`} viewBox={`0 0 ${wellBox.w} ${wellBox.h}`}>
                {travelOn
                  ? [
                      travelFocus === null || travelFocus === 'head'
                        ? limitRect(travelRects?.head, '#f45b69', false, 'lim-head')
                        : null,
                      travelFocus === null || travelFocus === 'head'
                        ? limitRect(travelRects?.head_wall, '#f45b69', true, 'lim-head-wall')
                        : null,
                      travelFocus === null || travelFocus === 'body'
                        ? limitRect(travelRects?.body, '#5ba4f4', false, 'lim-body')
                        : null,
                      travelFocus === null || travelFocus === 'body'
                        ? limitRect(travelRects?.body_wall, '#5ba4f4', true, 'lim-body-wall')
                        : null,
                    ]
                  : null}
                {showHair
                  ? (status?.hair ?? []).map((part, i) => {
                      const d = part.polygon
                        .map((xy) => {
                          const p = toScreen(xy[0], xy[1])
                          return `${p.x},${p.y}`
                        })
                        .join(' ')
                      if (!d) return null
                      let label: JSX.Element | null = null
                      if (showIds && part.polygon.length) {
                        const cx = part.polygon.reduce((s, xy) => s + xy[0], 0) / part.polygon.length
                        const cy = part.polygon.reduce((s, xy) => s + xy[1], 0) / part.polygon.length
                        const p = toScreen(cx, cy)
                        const w = part.width ?? 1
                        const tag = part.side === 'mid' ? 'M' : part.side === 'l' ? 'L' : part.side === 'r' ? 'R' : ''
                        label = (
                          <text x={p.x} y={p.y} fill="#ffe14a" fontSize="10" textAnchor="middle" pointerEvents="none">
                            {`${tag} ×${w.toFixed(2)}`}
                          </text>
                        )
                      }
                      return (
                        <g key={`${part.class}-${i}`}>
                          <polygon
                            points={d}
                            fill={HAIR_FILL[part.class] ?? 'rgba(200,200,200,0.2)'}
                            stroke="none"
                            pointerEvents="none"
                          />
                          {label}
                        </g>
                      )
                    })
                  : null}
                {showSkeleton
                  ? SKELETON_BONES.map(([a, b]) => {
                      const ja = (status?.skeleton ?? []).find((j) => j.id === a)
                      const jb = (status?.skeleton ?? []).find((j) => j.id === b)
                      if (!ja || !jb) return null
                      const pa = toScreen(ja.x, ja.y)
                      const pb = toScreen(jb.x, jb.y)
                      return (
                        <line
                          key={`${a}-${b}`}
                          x1={pa.x}
                          y1={pa.y}
                          x2={pb.x}
                          y2={pb.y}
                          stroke="rgba(120, 220, 80, 0.9)"
                          strokeWidth={2}
                          pointerEvents="none"
                        />
                      )
                    })
                  : null}
                {showSkeleton
                  ? (status?.skeleton ?? [])
                      .filter((j) => j.id !== 30)
                      .map((j) => {
                        const p = toScreen(j.x, j.y)
                        return (
                          <g key={j.id} pointerEvents="auto">
                            <circle
                              cx={p.x}
                              cy={p.y}
                              r={live ? 5 : 6}
                              fill="rgba(180, 255, 80, 0.95)"
                              stroke="rgba(8, 16, 24, 0.9)"
                              strokeWidth={1.5}
                              style={{ cursor: 'grab' }}
                              onPointerDown={(e) => {
                                if (e.button !== 0) return
                                e.preventDefault()
                                e.stopPropagation()
                                e.currentTarget.setPointerCapture(e.pointerId)
                                skeletonDrag.current = { id: j.id, x: j.x, y: j.y }
                                setDraggingSkeleton(true)
                              }}
                            />
                            <text x={p.x + 5} y={p.y - 3} fill="#7dff50" fontSize="9">
                              {refOf(j.id)}
                            </text>
                          </g>
                        )
                      })
                  : null}
                {showFace
                  ? LINES.map((line) => {
                      const d = line.ids
                        .map((i) => {
                          const row = points[i]
                          if (!row || (row[2] ?? 1) < 0.05) return ''
                          const p = toScreen(row[0], row[1])
                          return `${p.x},${p.y}`
                        })
                        .filter(Boolean)
                        .join(' ')
                      if (!d) return null
                      return (
                        <polyline
                          key={line.ids.join('-')}
                          points={d}
                          fill="none"
                          stroke={line.color}
                          strokeWidth={line.ids[0] === 23 ? 2.2 : 1.8}
                        />
                      )
                    })
                  : null}
                {showFace
                  ? points.map((row, i) => {
                      if (i === 15 || (row[2] ?? 1) < 0.05) return null
                      const p = toScreen(row[0], row[1])
                      const mouth = MOUTH_SET.has(i)
                      return (
                        <g key={i}>
                          <circle
                            cx={p.x}
                            cy={p.y}
                            r={14}
                            fill="transparent"
                            className="hit"
                            onPointerDown={(e) => {
                              if (e.button !== 0) return
                              e.preventDefault()
                              e.stopPropagation()
                              dragPt.current = i
                              setDragging(true)
                            }}
                          />
                          <circle
                            cx={p.x}
                            cy={p.y}
                            r={mouth ? 4.2 : 4.6}
                            fill={mouth ? '#e24cff' : '#3ad8ff'}
                            stroke="#081018"
                            strokeWidth={1.6}
                            pointerEvents="none"
                          />
                          <text
                            x={p.x + 7}
                            y={p.y - 5}
                            fill={mouth ? '#ff4d7a' : '#f4fbff'}
                            fontSize="12"
                            fontWeight="700"
                            stroke="#081018"
                            strokeWidth={3}
                            paintOrder="stroke"
                            pointerEvents="none"
                          >
                            {refOf(i)}
                          </text>
                        </g>
                      )
                    })
                  : null}
                {showFace
                  ? (status?.iris ?? [])
                      .filter((row) => row.visible !== false)
                      .map((row) => {
                        const p = toScreen(row.x, row.y)
                        return (
                          <g key={`iris-${row.id}`}>
                            <circle
                              cx={p.x}
                              cy={p.y}
                              r={16}
                              fill="transparent"
                              className="hit"
                              onPointerDown={(e) => {
                                if (e.button !== 0) return
                                e.preventDefault()
                                e.stopPropagation()
                                irisDrag.current = { id: row.id, x: row.x, y: row.y }
                                setDraggingIris(true)
                              }}
                            />
                            <circle
                              cx={p.x}
                              cy={p.y}
                              r={8.5}
                              fill="#ffe14a"
                              stroke="#fff6b0"
                              strokeWidth={2.4}
                              pointerEvents="none"
                            />
                            <circle cx={p.x} cy={p.y} r={3.2} fill="#081018" pointerEvents="none" />
                            <text
                              x={p.x + 9}
                              y={p.y - 8}
                              fill="#ffe14a"
                              fontSize="13"
                              fontWeight="800"
                              stroke="#081018"
                              strokeWidth={3.5}
                              paintOrder="stroke"
                              pointerEvents="none"
                            >
                              {refOf(row.id)}
                            </text>
                          </g>
                        )
                      })
                  : null}
              </svg>
            ) : null}
          </>
        ) : (
          <figcaption className="empty">
            Drop a still here, or click to load
            <code>input/source.png</code>
          </figcaption>
        )}
      </figure>

      <Side
        live={live}
        busy={busy}
        panel={panel}
        source={source}
        selected={selected}
        status={status}
        presets={presets}
        feel={feelVals}
        travel={status?.travel_box ?? null}
        weights={weights}
        ifmPort={ifmPort}
        copied={copied}
        localIps={localIps}
        onSelectPreset={previewPreset}
        onApply={() => void commitPreset(selected)}
        onAddMid={(a, b, t) => void addMid(a, b, t)}
        onScrub={scrubPair}
        onMoveKey={(id, t) => void moveKey(id, t)}
        onResetKey={(id) => void resetKey(id)}
        onDeleteKey={(id) => void deleteKey(id)}
        onPresetMenu={(id, x, y) => setMenu({ id, x, y })}
        onCalibrate={(id) => void runCalibrate(id)}
        onFeel={putFeel}
        onTravel={putTravel}
        onTravelFocus={setTravelFocus}
        onMirror={putMirror}
        onResetPoints={() => {
          void api.resetPoints().then((next) => apply(next, false, false)).catch((e) => setError(String(e)))
        }}
        onCopyAddr={(ip) => void copyAddr(ip)}
        onIfmPort={setIfmPort}
        onSaveIfm={saveIfm}
        onSetCamera={(index) => void setCamera(index)}
      />

      <footer className="bar">
        <button className="ghost" type="button" disabled={busy !== ''} onClick={() => picker.current?.click()}>
          {busy === 'load' ? 'Loading…' : 'Load'}
        </button>
        <button className="act" type="button" disabled={busy !== ''} onClick={() => void run('track')}>
          {busy === 'track' ? 'Overlay…' : 'Overlay'}
        </button>
        <button
          className="act"
          type="button"
          disabled={busy !== '' || points.length < 28}
          title="Run the model on the current overlay"
          onClick={() => void runGen()}
        >
          {busy === 'gen' ? 'Generating…' : 'Gen'}
        </button>
        <span
          className="osf-pick"
          onContextMenu={(e) => {
            e.preventDefault()
            setSourceMenu(true)
          }}
        >
          <button
            className={live ? 'act' : 'ghost'}
            type="button"
            disabled={busy !== '' || (!live && !status?.ready)}
            title="Right-click to choose Camera or iFacialMocap"
            onClick={() => void runOsf()}
            onContextMenu={(e) => {
              e.preventDefault()
              setSourceMenu(true)
            }}
          >
            {busy === 'osf'
              ? source === 'ifm'
                ? 'Listen…'
                : 'Stake…'
              : live
                ? source === 'ifm'
                  ? 'Stop'
                  : 'Stop Stake'
                : source === 'ifm'
                  ? 'Listen'
                  : 'Stake'}
          </button>
          {sourceMenu ? (
            <div
              className="menu osf-menu"
              role="menu"
              aria-label="Tracking source"
              onClick={(e) => e.stopPropagation()}
              onContextMenu={(e) => e.preventDefault()}
            >
              <button
                type="button"
                role="menuitemradio"
                aria-checked={source === 'camera'}
                className={source === 'camera' ? 'on' : ''}
                onClick={() => {
                  setSourceMenu(false)
                  void setInput('camera')
                }}
              >
                Camera
              </button>
              <button
                type="button"
                role="menuitemradio"
                aria-checked={source === 'ifm'}
                className={source === 'ifm' ? 'on' : ''}
                onClick={() => {
                  setSourceMenu(false)
                  void setInput('ifm')
                }}
              >
                iFacialMocap
              </button>
            </div>
          ) : null}
        </span>
        <button className="ghost" type="button" disabled={busy !== ''} onClick={() => void run('reset')}>
          {busy === 'reset' ? 'Resetting…' : 'Reset'}
        </button>
        <button
          className={status?.recording ? 'act' : 'ghost'}
          type="button"
          disabled={busy !== '' || (!status?.recording && (!live || !status?.has_source))}
          title={
            status?.record_path
              ? `Saved ${status.record_frames ?? 0} frames to ${status.record_path}`
              : 'Record the tracked character for the benchmark'
          }
          onClick={() => void runRecord()}
        >
          {status?.recording
            ? `Stop ${Math.max(0, status.record_seconds ?? 0).toFixed(1)}s`
            : 'Record movement'}
        </button>
      </footer>

      {error ? <p className="err">{error}</p> : null}
      {!status?.recording && status?.record_path ? (
        <p className="hint">Saved {status.record_frames ?? 0} frames · {status.record_path}</p>
      ) : null}

      {menu ? (
        <div className="menu" style={{ left: menu.x, top: menu.y }} onClick={(e) => e.stopPropagation()}>
          <button type="button" disabled={!pointsFor(menu.id)} onClick={() => copyPreset(menu.id)}>
            Copy
          </button>
          <button type="button" disabled={!clip || busy !== ''} onClick={() => void pastePreset(menu.id)}>
            Paste
          </button>
        </div>
      ) : null}

      {camZoom && live ? (
        <div
          className="cam-zoom"
          onClick={() => {
            if (!camDragged.current) setCamZoom(false)
            camDragged.current = false
          }}
          onContextMenu={(e) => {
            e.preventDefault()
            setCamZoom(false)
          }}
        >
          <div
            ref={camStage}
            className="cam-zoom-stage"
            onClick={(e) => e.stopPropagation()}
            onPointerDown={(e) => {
              if (e.button !== 0) return
              camDragged.current = false
              camPan.current = { mx: e.clientX, my: e.clientY, x: camViewRef.current.x, y: camViewRef.current.y }
              e.currentTarget.setPointerCapture(e.pointerId)
            }}
            onPointerMove={(e) => {
              const start = camPan.current
              if (!start) return
              const dx = e.clientX - start.mx
              const dy = e.clientY - start.my
              if (Math.hypot(dx, dy) > 3) camDragged.current = true
              putCamView({ scale: camViewRef.current.scale, x: start.x + dx, y: start.y + dy })
            }}
            onPointerUp={() => {
              camPan.current = null
            }}
            onPointerCancel={() => {
              camPan.current = null
            }}
          >
            <button
              type="button"
              className={`id-map${showIds ? ' on' : ''}`}
              onClick={(e) => {
                e.stopPropagation()
                putFeel('show_ids', showIds ? 0 : 1)
              }}
              onPointerDown={(e) => e.stopPropagation()}
            >
              {showIds ? 'Hide numbers' : 'Show numbers'}
            </button>
            <img
              src={frameUrl('camera', camBust)}
              alt={source === 'ifm' ? 'iFacialMocap' : 'OSF camera large'}
              draggable={false}
              style={{ transform: `translate(${camView.x}px, ${camView.y}px) scale(${camView.scale})` }}
              onLoad={() => {
                camBusy.current = false
              }}
              onError={() => {
                camBusy.current = false
              }}
            />
          </div>
          <aside
            className="lip-roster"
            onClick={(e) => e.stopPropagation()}
            onPointerDown={(e) => e.stopPropagation()}
            onContextMenu={(e) => {
              e.preventDefault()
              e.stopPropagation()
            }}
          >
            <p className="eyebrow">Eyes</p>
            <p className="lip-hint">OSF → overlay. L/R is the person.</p>
            <ul>
              {(status?.eye_points ?? []).map((row) => {
                const dest = Number(eyeTo[row.id] ?? row.to ?? -1)
                return (
                <li key={row.id} className={row.on ? 'on' : ''}>
                  <button
                    type="button"
                    className={row.on ? 'on' : ''}
                    title={camRefOf(row.id)}
                    onClick={() => void putEyePoint(row.id, { on: !row.on })}
                  >
                    <b>{camShortOf(row.id)}{row.artificial ? '*' : ''}</b>
                    <em>{row.artificial ? 'mid' : row.on ? 'On' : 'Off'}</em>
                  </button>
                  <span aria-hidden="true">→</span>
                  <label className="lip-dest">
                    <input
                      inputMode="numeric"
                      maxLength={2}
                      placeholder="—"
                      title={dest >= 0 ? refOf(dest) : ''}
                      value={eyeTo[row.id] ?? (row.to == null ? '' : String(row.to))}
                      onChange={(e) => {
                        setEyeTo((cur) => ({ ...cur, [row.id]: e.target.value.replace(/\D/g, '').slice(0, 2) }))
                      }}
                      onBlur={(e) => commitMap(row.id, e.target.value, setEyeTo, (id, patch) => void putEyePoint(id, patch))}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') e.currentTarget.blur()
                      }}
                      aria-label={`Overlay slot for ${camRefOf(row.id)}`}
                    />
                    <small>{dest >= 0 ? refOf(dest) : ''}</small>
                  </label>
                </li>
                )
              })}
            </ul>
            <p className="eyebrow">Lips</p>
            <p className="lip-hint">On draws. Slot is the overlay ref.</p>
            <ul>
              {(status?.mouth_points ?? []).map((row) => {
                const dest = Number(lipTo[row.id] ?? row.to ?? -1)
                return (
                <li key={row.id} className={row.on ? 'on' : ''}>
                  <button
                    type="button"
                    className={row.on ? 'on' : ''}
                    title={camRefOf(row.id)}
                    onClick={() => void putMouthPoint(row.id, { on: !row.on })}
                  >
                    <b>{camShortOf(row.id)}</b>
                    <em>{row.on ? 'On' : 'Off'}</em>
                  </button>
                  <span aria-hidden="true">→</span>
                  <label className="lip-dest">
                    <input
                      inputMode="numeric"
                      maxLength={2}
                      placeholder="—"
                      title={dest >= 0 ? refOf(dest) : ''}
                      value={lipTo[row.id] ?? (row.to == null ? '' : String(row.to))}
                      onChange={(e) => {
                        setLipTo((cur) => ({ ...cur, [row.id]: e.target.value.replace(/\D/g, '').slice(0, 2) }))
                      }}
                      onBlur={(e) => commitMap(row.id, e.target.value, setLipTo, (id, patch) => void putMouthPoint(id, patch))}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') e.currentTarget.blur()
                      }}
                      aria-label={`Overlay slot for ${camRefOf(row.id)}`}
                    />
                    <small>{dest >= 0 ? refOf(dest) : ''}</small>
                  </label>
                </li>
                )
              })}
            </ul>
          </aside>
          <p>Scroll to zoom · drag to pan · Esc to close</p>
        </div>
      ) : null}
    </div>
  )
}
