import { forwardRef, useEffect, useImperativeHandle, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react'
import { api, type CharacterFit as FitView, type FitBox, type TravelBox } from '../api'
import { useI18n, type MessageKey } from '../i18n'

const ROOM_MAX = 1.2
const BRUSH_MIN = 6
const BRUSH_MAX = 48
const ZOOM_MIN = 1
const ZOOM_MAX = 8

const HAIR_PARTS = [
  { id: 'hair_middle', label: 'fit.middle', color: 'rgba(255, 200, 0, 0.42)' },
  { id: 'hair_left', label: 'fit.left', color: 'rgba(0, 180, 255, 0.42)' },
  { id: 'hair_right', label: 'fit.right', color: 'rgba(255, 80, 160, 0.42)' },
] as const satisfies readonly { id: string; label: MessageKey; color: string }[]

const MODE_LABELS: Record<Mode, MessageKey> = {
  hair: 'fit.hair',
  points: 'fit.points',
  limiters: 'fit.limiters',
}

type Mode = 'hair' | 'points' | 'limiters'
type Mark = FitView['points'][number]
type Edge =
  | 'body-left'
  | 'body-right'
  | 'body-up'
  | 'body-down'
  | 'head-left'
  | 'head-right'
  | 'head-up'
  | 'head-down'

type HairStroke = {
  part: (typeof HAIR_PARTS)[number]['id']
  points: number[][]
  erase: boolean
  radius: number
}

type Snap = {
  pending: HairStroke[]
  marks: Mark[]
  draft: TravelBox | null
}

type Props = {
  stillUrl: string
  travel?: TravelBox | null
  onTravel: (box: TravelBox) => void
  onNotice: (message: string) => void
}

export type CharacterFitHandle = {
  commit: () => Promise<void>
}

function clamp(n: number, lo: number, hi: number) {
  if (!Number.isFinite(n)) return lo
  return Math.max(lo, Math.min(hi, n))
}

function normX(px: number, width: number) {
  return (px / Math.max(width, 1)) * 2 - 1
}

function normY(py: number, height: number) {
  return (py / Math.max(height, 1)) * 2 - 1
}

function pixX(n: number, width: number) {
  return ((n + 1) * 0.5) * width
}

function pixY(n: number, height: number) {
  return ((n + 1) * 0.5) * height
}

function expandBox(
  tight: FitBox,
  faceHeight: number,
  width: number,
  height: number,
  left: number,
  right: number,
  up: number,
  down: number,
): FitBox {
  if (!tight) return null
  const x0 = normX(tight[0], width) - left * faceHeight
  const x1 = normX(tight[2], width) + right * faceHeight
  const y0 = normY(tight[1], height) - up * faceHeight
  const y1 = normY(tight[3], height) + down * faceHeight
  return [pixX(x0, width), pixY(y0, height), pixX(x1, width), pixY(y1, height)]
}

function boxPath(box: FitBox) {
  if (!box) return ''
  const [x0, y0, x1, y1] = box
  return `${x0},${y0} ${x1},${y0} ${x1},${y1} ${x0},${y1} ${x0},${y0}`
}

export const CharacterFit = forwardRef<CharacterFitHandle, Props>(function CharacterFit(props, ref) {
  const { t, tr } = useI18n()
  const [fit, setFit] = useState<FitView | null>(null)
  const [mode, setMode] = useState<Mode>('hair')
  const [part, setPart] = useState<(typeof HAIR_PARTS)[number]['id']>('hair_middle')
  const [erase, setErase] = useState(false)
  const [radius, setRadius] = useState(18)
  const [busy, setBusy] = useState(false)
  const [pastCount, setPastCount] = useState(0)
  const [futureCount, setFutureCount] = useState(0)
  const [cursor, setCursor] = useState<{ x: number; y: number } | null>(null)
  const [stroke, setStroke] = useState<number[][]>([])
  const [pending, setPending] = useState<HairStroke[]>([])
  const [marks, setMarks] = useState<Mark[]>([])
  const [grabbed, setGrabbed] = useState<number | null>(null)
  const [draft, setDraft] = useState<TravelBox | null>(null)
  const strokeRef = useRef<number[][]>([])
  const pendingRef = useRef<HairStroke[]>([])
  const marksRef = useRef<Mark[]>([])
  const fitRef = useRef<FitView | null>(null)
  const draftRef = useRef<TravelBox | null>(null)
  const pastRef = useRef<Snap[]>([])
  const futureRef = useRef<Snap[]>([])
  const beforeDrag = useRef<Snap | null>(null)
  const radiusRef = useRef(18)
  const stageRef = useRef<HTMLDivElement>(null)
  const zoomRef = useRef(1)
  const panRef = useRef({ x: 0, y: 0 })
  const [zoom, setZoom] = useState(1)
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const panning = useRef(false)
  const panOrigin = useRef({ x: 0, y: 0, panX: 0, panY: 0 })
  const drag = useRef<
    | { kind: 'hair' }
    | { kind: 'point'; id: number }
    | { kind: 'edge'; edge: Edge }
    | null
  >(null)
  const travel = draft ?? props.travel ?? null

  function writeDraft(box: TravelBox | null) {
    draftRef.current = box
    setDraft(box)
  }

  function writeStroke(points: number[][]) {
    strokeRef.current = points
    setStroke(points)
  }

  function writePending(strokes: HairStroke[]) {
    pendingRef.current = strokes
    setPending(strokes)
  }

  function writeMarks(rows: Mark[]) {
    marksRef.current = rows
    setMarks(rows)
  }

  function snapshot(): Snap {
    return {
      pending: pendingRef.current.map((row) => ({
        ...row,
        points: row.points.map((point) => [point[0], point[1]]),
      })),
      marks: marksRef.current.map((row) => ({ ...row })),
      draft: draftRef.current ? { ...draftRef.current } : null,
    }
  }

  function restore(snap: Snap) {
    writePending(snap.pending)
    writeMarks(snap.marks)
    writeDraft(snap.draft)
  }

  function remember(before: Snap | null) {
    if (!before) return
    pastRef.current.push(before)
    futureRef.current = []
    setPastCount(pastRef.current.length)
    setFutureCount(0)
  }

  function undo() {
    const before = pastRef.current.pop()
    if (!before) return
    futureRef.current.push(snapshot())
    setPastCount(pastRef.current.length)
    setFutureCount(futureRef.current.length)
    restore(before)
  }

  function redo() {
    const next = futureRef.current.pop()
    if (!next) return
    pastRef.current.push(snapshot())
    setPastCount(pastRef.current.length)
    setFutureCount(futureRef.current.length)
    restore(next)
  }

  useEffect(() => {
    let cancel = false
    api
      .characterFit()
      .then((view) => {
        if (cancel) return
        fitRef.current = view
        setFit(view)
        writeMarks(view.points ?? [])
      })
      .catch((err: unknown) => {
        if (!cancel) props.onNotice(err instanceof Error ? err.message : String(err))
      })
    return () => {
      cancel = true
    }
  }, [])

  useEffect(() => {
    radiusRef.current = radius
  }, [radius])

  useEffect(() => {
    const stage = stageRef.current
    if (!stage) return
    const view = stage
    function onWheel(event: WheelEvent) {
      event.preventDefault()
      const delta = Math.abs(event.deltaY) >= Math.abs(event.deltaX) ? event.deltaY : event.deltaX
      if (!delta) return
      const rect = view.getBoundingClientRect()
      const px = event.clientX - rect.left
      const py = event.clientY - rect.top
      const cx = rect.width / 2
      const cy = rect.height / 2
      const prev = zoomRef.current
      const next = clamp(prev * (delta < 0 ? 1.12 : 1 / 1.12), ZOOM_MIN, ZOOM_MAX)
      const scale = next / prev
      const origin = panRef.current
      const nextPan =
        next <= ZOOM_MIN + 1e-3
          ? { x: 0, y: 0 }
          : {
              x: px - cx - (px - cx - origin.x) * scale,
              y: py - cy - (py - cy - origin.y) * scale,
            }
      zoomRef.current = next
      panRef.current = nextPan
      setZoom(next)
      setPan(nextPan)
    }
    stage.addEventListener('wheel', onWheel, { passive: false })
    return () => stage.removeEventListener('wheel', onWheel)
  }, [])

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      const target = event.target
      if (target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement) return
      const key = event.key.toLowerCase()
      if (!(event.ctrlKey || event.metaKey)) return
      if (key !== 'z' && key !== 'y') return
      event.preventDefault()
      if (key === 'y' || event.shiftKey) redo()
      else undo()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const width = fit?.width ?? 1
  const height = fit?.height ?? 1
  const faceHeight = fit?.face_height || 0.001
  const bodyWall = expandBox(
    fit?.boxes.body_tight ?? null,
    faceHeight,
    width,
    height,
    travel?.body_left ?? 0,
    travel?.body_right ?? 0,
    travel?.body_up ?? 0,
    travel?.body_down ?? 0,
  )
  const headWall = expandBox(
    fit?.boxes.head_tight ?? null,
    faceHeight,
    width,
    height,
    travel?.left ?? 0,
    travel?.right ?? 0,
    travel?.up ?? 0,
    travel?.down ?? 0,
  )

  function toImage(event: ReactPointerEvent<SVGSVGElement>) {
    const rect = event.currentTarget.getBoundingClientRect()
    if (rect.width <= 0 || rect.height <= 0 || !fit) return null
    return {
      x: ((event.clientX - rect.left) / rect.width) * fit.width,
      y: ((event.clientY - rect.top) / rect.height) * fit.height,
    }
  }

  function slop() {
    return Math.max(10, width * 0.014)
  }

  function hitMark(x: number, y: number) {
    // Face points sit close together; the nearest one inside reach wins.
    const reach = Math.max(12, width * 0.018) / zoomRef.current
    let best: number | null = null
    let bestD = reach
    for (const mark of marks) {
      const d = Math.hypot(mark.x - x, mark.y - y)
      if (d <= bestD) {
        best = mark.id
        bestD = d
      }
    }
    return best
  }

  function hitEdge(x: number, y: number): Edge | null {
    const pad = slop()
    const boxes: [Edge, FitBox, 'x0' | 'x1' | 'y0' | 'y1'][] = [
      ['body-left', bodyWall, 'x0'],
      ['body-right', bodyWall, 'x1'],
      ['body-up', bodyWall, 'y0'],
      ['body-down', bodyWall, 'y1'],
      ['head-left', headWall, 'x0'],
      ['head-right', headWall, 'x1'],
      ['head-up', headWall, 'y0'],
      ['head-down', headWall, 'y1'],
    ]
    let best: Edge | null = null
    let bestD = pad
    for (const [edge, box, side] of boxes) {
      if (!box) continue
      const [x0, y0, x1, y1] = box
      const onX = x >= x0 - pad && x <= x1 + pad
      const onY = y >= y0 - pad && y <= y1 + pad
      let d = pad + 1
      if (side === 'x0' && onY) d = Math.abs(x - x0)
      if (side === 'x1' && onY) d = Math.abs(x - x1)
      if (side === 'y0' && onX) d = Math.abs(y - y0)
      if (side === 'y1' && onX) d = Math.abs(y - y1)
      if (d <= bestD) {
        best = edge
        bestD = d
      }
    }
    return best
  }

  function applyEdge(edge: Edge, x: number, y: number, base: TravelBox): TravelBox {
    if (!fit) return base
    const next: TravelBox = { ...base, enabled: true }
    const fh = fit.face_height || 0.001
    const body = fit.boxes.body_tight
    const head = fit.boxes.head_tight
    if (edge === 'body-left' && body) {
      next.body_left = clamp((normX(body[0], fit.width) - normX(x, fit.width)) / fh, 0, ROOM_MAX)
    } else if (edge === 'body-right' && body) {
      next.body_right = clamp((normX(x, fit.width) - normX(body[2], fit.width)) / fh, 0, ROOM_MAX)
    } else if (edge === 'body-up' && body) {
      next.body_up = clamp((normY(body[1], fit.height) - normY(y, fit.height)) / fh, 0, ROOM_MAX)
    } else if (edge === 'body-down' && body) {
      next.body_down = clamp((normY(y, fit.height) - normY(body[3], fit.height)) / fh, 0, ROOM_MAX)
    } else if (edge === 'head-left' && head) {
      next.left = clamp((normX(head[0], fit.width) - normX(x, fit.width)) / fh, 0, ROOM_MAX)
    } else if (edge === 'head-right' && head) {
      next.right = clamp((normX(x, fit.width) - normX(head[2], fit.width)) / fh, 0, ROOM_MAX)
    } else if (edge === 'head-up' && head) {
      next.up = clamp((normY(head[1], fit.height) - normY(y, fit.height)) / fh, 0, ROOM_MAX)
    } else if (edge === 'head-down' && head) {
      next.down = clamp((normY(y, fit.height) - normY(head[3], fit.height)) / fh, 0, ROOM_MAX)
    }
    return next
  }

  function onDown(event: ReactPointerEvent<SVGSVGElement>) {
    if (event.button !== 0 || !fit || busy) return
    event.preventDefault()
    event.currentTarget.setPointerCapture(event.pointerId)
    if (event.shiftKey) {
      panning.current = true
      panOrigin.current = {
        x: event.clientX,
        y: event.clientY,
        panX: panRef.current.x,
        panY: panRef.current.y,
      }
      return
    }
    const xy = toImage(event)
    if (!xy) return
    beforeDrag.current = snapshot()
    if (mode === 'hair') {
      drag.current = { kind: 'hair' }
      writeStroke([[xy.x, xy.y]])
      setCursor(xy)
      return
    }
    if (mode === 'points') {
      const id = hitMark(xy.x, xy.y)
      if (id == null) return
      drag.current = { kind: 'point', id }
      setGrabbed(id)
      writeMarks(marksRef.current.map((row) => (row.id === id ? { ...row, x: xy.x, y: xy.y } : row)))
      return
    }
    const edge = hitEdge(xy.x, xy.y)
    if (!edge || !props.travel) return
    drag.current = { kind: 'edge', edge }
    writeDraft(applyEdge(edge, xy.x, xy.y, draftRef.current ?? props.travel))
  }

  function onMove(event: ReactPointerEvent<SVGSVGElement>) {
    if (panning.current) {
      const origin = panOrigin.current
      const next = {
        x: origin.panX + (event.clientX - origin.x),
        y: origin.panY + (event.clientY - origin.y),
      }
      panRef.current = next
      setPan(next)
      return
    }
    const xy = toImage(event)
    if (!xy) return
    if (mode === 'hair') setCursor(xy)
    const active = drag.current
    if (!active) return
    if (active.kind === 'hair') {
      const pts = strokeRef.current
      const last = pts[pts.length - 1]
      if (last && Math.hypot(last[0] - xy.x, last[1] - xy.y) < 2) return
      writeStroke([...pts, [xy.x, xy.y]])
      return
    }
    if (active.kind === 'point') {
      writeMarks(marksRef.current.map((row) => (row.id === active.id ? { ...row, x: xy.x, y: xy.y } : row)))
      return
    }
    const base = draftRef.current ?? props.travel
    if (base) writeDraft(applyEdge(active.edge, xy.x, xy.y, base))
  }

  function finish() {
    if (panning.current) {
      panning.current = false
      drag.current = null
      return
    }
    const active = drag.current
    drag.current = null
    setGrabbed(null)
    if (!active) return
    if (active.kind === 'hair') {
      const points = strokeRef.current
      const before = beforeDrag.current
      writeStroke([])
      if (points.length < 1) return
      remember(before)
      writePending([...pendingRef.current, { part, points, erase, radius: radiusRef.current }])
      return
    }
    const before = beforeDrag.current
    beforeDrag.current = null
    if (!before) return
    const changed =
      active.kind === 'point'
        ? marksRef.current.some((mark) => {
            const orig = before.marks.find((row) => row.id === mark.id)
            return !orig || Math.abs(orig.x - mark.x) > 0.5 || Math.abs(orig.y - mark.y) > 0.5
          })
        : JSON.stringify(draftRef.current) !== JSON.stringify(before.draft)
    if (changed) remember(before)
  }

  async function commit() {
    const strokes = pendingRef.current
    const saved = fitRef.current
    const moved = marksRef.current.filter((mark) => {
      const orig = saved?.points?.find((row) => row.id === mark.id)
      return !orig || Math.abs(orig.x - mark.x) > 0.5 || Math.abs(orig.y - mark.y) > 0.5
    })
    const box = draftRef.current
    if (!strokes.length && !moved.length && !box) return
    setBusy(true)
    try {
      let view = saved
      for (const stroke of strokes) {
        const res = await api.fitHair(stroke)
        view = res.fit
      }
      for (const mark of moved) {
        const res = await api.fitPoint(mark.id, mark.x, mark.y)
        view = res.fit
      }
      if (box) props.onTravel(box)
      fitRef.current = view
      setFit(view)
      if (view) writeMarks(view.points ?? [])
      writePending([])
      writeDraft(null)
      pastRef.current = []
      futureRef.current = []
      setPastCount(0)
      setFutureCount(0)
      props.onNotice('')
    } catch (err: unknown) {
      props.onNotice(err instanceof Error ? err.message : String(err))
      throw err
    } finally {
      setBusy(false)
    }
  }

  useImperativeHandle(ref, () => ({ commit }))

  const dirty =
    pending.length > 0 ||
    draft != null ||
    marks.some((mark) => {
      const orig = fit?.points?.find((row) => row.id === mark.id)
      return !orig || Math.abs(orig.x - mark.x) > 0.5 || Math.abs(orig.y - mark.y) > 0.5
    })

  return (
    <div className="char-fit">
      <div ref={stageRef} className={`fit-stage${zoom > 1.01 ? ' is-zoomed' : ''}`}>
        <div
          className="fit-frame"
          style={{
            aspectRatio: `${width} / ${height}`,
            transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
          }}
        >
          {props.stillUrl ? <img src={props.stillUrl} alt="" draggable={false} /> : null}
          <svg
            className={`fit-overlay is-${mode}`}
            viewBox={`0 0 ${width} ${height}`}
            onPointerDown={onDown}
            onPointerMove={onMove}
            onPointerUp={finish}
            onPointerCancel={finish}
          >
            {mode === 'limiters' ? (
              <>
                <polyline className="fit-head" points={boxPath(headWall)} />
                <polyline className="fit-body" points={boxPath(bodyWall)} />
              </>
            ) : null}
            {mode === 'hair'
              ? (fit?.hair ?? []).map((seg, index) => {
                  const color = HAIR_PARTS.find((row) => row.id === seg.class)?.color
                  return (
                    <polygon
                      key={`${seg.class}-${index}`}
                      points={seg.polygon.map((p) => p.join(',')).join(' ')}
                      fill={color ?? 'rgba(255,255,255,0.3)'}
                      stroke={color ?? '#fff'}
                      strokeWidth={1.5}
                    />
                  )
                })
              : null}
            {mode === 'hair'
              ? pending.map((row, index) => (
                  <polyline
                    key={`pending-${index}`}
                    points={row.points.map((p) => p.join(',')).join(' ')}
                    fill="none"
                    stroke={row.erase ? '#f2f2f2' : HAIR_PARTS.find((part) => part.id === row.part)?.color}
                    strokeWidth={row.radius * 2}
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                ))
              : null}
            {mode === 'hair' && stroke.length > 0 ? (
              <polyline
                points={stroke.map((p) => p.join(',')).join(' ')}
                fill="none"
                stroke={erase ? '#f2f2f2' : HAIR_PARTS.find((row) => row.id === part)?.color}
                strokeWidth={radius * 2}
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            ) : null}
            {mode === 'hair' && cursor ? (
              <circle cx={cursor.x} cy={cursor.y} r={radius} className="fit-brush" />
            ) : null}
            {mode === 'points'
              ? marks.map((mark) => (
                  <circle
                    key={mark.id}
                    cx={mark.x}
                    cy={mark.y}
                    r={(mark.group === 'body' ? Math.max(7, width * 0.012) : Math.max(4, width * 0.007)) / zoom}
                    className={`fit-joint is-${mark.group}${grabbed === mark.id ? ' is-grabbed' : ''}`}
                  >
                    <title>{tr(mark.label)}</title>
                  </circle>
                ))
              : null}
            {mode === 'points' && grabbed != null
              ? marks
                  .filter((mark) => mark.id === grabbed)
                  .map((mark) => (
                    <text
                      key="grabbed-label"
                      x={mark.x}
                      y={mark.y - Math.max(10, width * 0.016) / zoom}
                      className="fit-point-label"
                      fontSize={Math.max(11, width * 0.016) / zoom}
                    >
                      {tr(mark.label)}
                    </text>
                  ))
              : null}
          </svg>
        </div>
      </div>
      <div className="fit-tools">
        <div className="fit-modes" role="tablist" aria-label={t('fit.tools')}>
          {(['hair', 'points', 'limiters'] as Mode[]).map((id) => (
            <button
              key={id}
              type="button"
              role="tab"
              aria-selected={mode === id}
              className={`btn fit-mode${mode === id ? ' is-on' : ''}`}
              onClick={() => setMode(id)}
            >
              {t(MODE_LABELS[id])}
            </button>
          ))}
        </div>
        {mode === 'hair' ? (
          <>
            <div className="fit-parts">
              {HAIR_PARTS.map((row) => (
                <button
                  key={row.id}
                  type="button"
                  className={`btn fit-part${part === row.id ? ' is-on' : ''}`}
                  onClick={() => {
                    setPart(row.id)
                    setErase(false)
                  }}
                >
                  <i style={{ background: row.color }} />
                  {t(row.label)}
                </button>
              ))}
            </div>
            <div className="fit-actions">
              <button
                type="button"
                className={`btn${erase ? '' : ' primary'}`}
                onClick={() => setErase(false)}
              >
                {t('fit.draw')}
              </button>
              <button
                type="button"
                className={`btn${erase ? ' primary' : ''}`}
                onClick={() => setErase(true)}
              >
                {t('fit.erase')}
              </button>
            </div>
            <ul className="lab-sliders">
              <li>
                <span>{t('fit.brush')}</span>
                <input
                  type="range"
                  min={BRUSH_MIN}
                  max={BRUSH_MAX}
                  step={1}
                  value={radius}
                  onChange={(event) => setRadius(Number(event.target.value))}
                />
                <em className="mono">{radius}</em>
              </li>
            </ul>
          </>
        ) : null}
        <div className="fit-actions">
          <button type="button" className="btn" disabled={pastCount < 1 || busy} onClick={undo}>
            {t('common.undo')}
          </button>
          <button type="button" className="btn" disabled={futureCount < 1 || busy} onClick={redo}>
            {t('common.redo')}
          </button>
        </div>
        <button type="button" className="btn primary" disabled={!dirty || busy} onClick={() => void commit()}>
          {t('common.apply')}
        </button>
      </div>
    </div>
  )
})
