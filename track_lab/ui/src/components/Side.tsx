import { useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react'
import type { FeelSettings, HairPart, IrisCamHit, LabStatus, MixWeights, MouthPreset, TravelBox as TravelBoxValue } from '../api'
import { FEEL, METERS, OVERLAY, ZERO_FEEL, type Busy } from '../constants'
import { MOUTH_ENDS, MOUTH_LABEL, keysOn, pairId } from '../points'
import { TravelBox, type TravelFocus } from './TravelBox'

export type SideProps = {
  live: boolean
  busy: Busy
  panel: 'desk' | 'limiters' | 'blend'
  source: 'camera' | 'ifm'
  selected: string
  status: LabStatus | null
  presets: MouthPreset[]
  feel: FeelSettings
  travel: TravelBoxValue | null
  weights: MixWeights
  ifmPort: string
  copied: string
  localIps: string[]
  onSelectPreset: (id: string) => void
  onApply: () => void
  onAddMid: (a: string, b: string, t: number) => void
  onScrub: (a: string, b: string, t: number) => void
  onMoveKey: (id: string, t: number) => Promise<void> | void
  onResetKey: (id: string) => void
  onDeleteKey: (id: string) => void
  onPresetMenu: (id: string, x: number, y: number) => void
  onCalibrate: (id: string) => void
  onFeel: (key: keyof FeelSettings, value: number) => void
  onTravel: (patch: Partial<TravelBoxValue>) => void
  onTravelFocus: (focus: TravelFocus | null) => void
  onMirror: (on: boolean) => void
  onResetPoints: () => void
  onCopyAddr: (ip: string) => void
  onIfmPort: (value: string) => void
  onSaveIfm: () => void
  onSetCamera: (index: number) => void
  onRefreshCameras: () => void
}

export function Side({
  live,
  busy,
  panel,
  source,
  selected,
  status,
  presets,
  feel,
  travel,
  weights,
  ifmPort,
  copied,
  localIps,
  onSelectPreset,
  onApply,
  onAddMid,
  onScrub,
  onMoveKey,
  onResetKey,
  onDeleteKey,
  onCalibrate,
  onFeel,
  onTravel,
  onTravelFocus,
  onMirror,
  onResetPoints,
  onCopyAddr,
  onIfmPort,
  onSaveIfm,
  onSetCamera,
  onRefreshCameras,
}: SideProps) {
  return (
    <aside className="side">
      {panel === 'limiters' ? (
        <TravelBox value={travel} disabled={busy !== ''} onChange={onTravel} onFocus={onTravelFocus} />
      ) : panel === 'blend' ? (
        <MouthSection
          live={live}
          busy={busy}
          selected={selected}
          status={status}
          presets={presets}
          onSelectPreset={onSelectPreset}
          onApply={onApply}
          onAddMid={onAddMid}
          onScrub={onScrub}
          onMoveKey={onMoveKey}
          onResetKey={onResetKey}
          onDeleteKey={onDeleteKey}
        />
      ) : (
      <>
      {source === 'ifm' ? (
        <IfmSection
          live={live}
          busy={busy}
          status={status}
          ifmPort={ifmPort}
          copied={copied}
          localIps={localIps}
          onCopyAddr={onCopyAddr}
          onIfmPort={onIfmPort}
          onSaveIfm={onSaveIfm}
        />
      ) : (
        <CameraSection
          live={live}
          busy={busy}
          status={status}
          onSetCamera={onSetCamera}
          onRefreshCameras={onRefreshCameras}
        />
      )}
      <TrackingSection
        live={live}
        busy={busy}
        status={status}
        onCalibrate={onCalibrate}
      />
      <SettingsSection
        feel={feel}
        mirrored={Boolean(status?.mirror)}
        nudged={(status?.point_offsets?.length ?? 0) > 0}
        onFeel={onFeel}
        onMirror={onMirror}
        onResetPoints={onResetPoints}
      />
      <ActivitySection weights={weights} blink={status?.blink} hair={status?.hair} />
      </>
      )}
    </aside>
  )
}

const KEY_INSET = 16
const KEY_GAP = 0.04

function homeCursor(saved: { t: number }[]) {
  const taken = (spot: number) =>
    saved.some((key) => Math.abs(key.t - spot) < KEY_GAP) || spot <= KEY_GAP || spot >= 1 - KEY_GAP
  if (!taken(0.5)) return 0.5
  return [0.32, 0.68, 0.2, 0.8].find((spot) => !taken(spot)) ?? 0.5
}

const MOUTH_PAIRS: [string, string][] = MOUTH_ENDS.flatMap((left, index) =>
  MOUTH_ENDS.slice(index + 1).map((right) => [left, right] as [string, string]),
)

type KeyDrag =
  | { kind: 'cursor'; origin: number; moved: boolean }
  | { kind: 'key'; id: string; origin: number; moved: boolean }

function MouthSection({
  live,
  busy,
  selected,
  status,
  presets,
  onSelectPreset,
  onApply,
  onAddMid,
  onScrub,
  onMoveKey,
  onResetKey,
  onDeleteKey,
}: Pick<
  SideProps,
  | 'live'
  | 'busy'
  | 'selected'
  | 'status'
  | 'presets'
  | 'onSelectPreset'
  | 'onApply'
  | 'onAddMid'
  | 'onScrub'
  | 'onMoveKey'
  | 'onResetKey'
  | 'onDeleteKey'
>) {
  const locked = live || busy !== '' || (!status?.ready && !status?.points?.length)
  const shapeIds = [...(status?.mids ?? []), ...Object.keys(status?.shapes ?? {})]
  const [livePair, setLivePair] = useState<string | null>(null)
  const [page, setPage] = useState<'blend' | 'shapes'>('blend')
  const editing = page === 'shapes'
  return (
    <>
      <div className="mouth-head">
        <p className="side-label">{editing ? 'Shapes' : 'Mouth'}</p>
        <button
          type="button"
          className="mouth-arrow"
          aria-label={editing ? 'Back to blends' : 'Edit original shapes'}
          title={editing ? 'Back to blends' : 'Edit original shapes'}
          onClick={() => setPage(editing ? 'blend' : 'shapes')}
        >
          {editing ? '‹' : '›'}
        </button>
      </div>
      {editing ? (
        <ul className="presets">
          {presets.filter((preset) => preset.id !== 'O').map((preset) => (
            <li key={preset.id}>
              <label className={selected === preset.id ? 'on' : ''}>
                <input
                  type="radio"
                  name="mouth-shape"
                  checked={selected === preset.id}
                  disabled={locked}
                  onChange={() => onSelectPreset(preset.id)}
                />
                <span className="mark" />
                <span>{preset.label}</span>
                <em>{preset.ready ? 'saved' : ''}</em>
              </label>
            </li>
          ))}
        </ul>
      ) : (
      <div className="keybar-list">
        {MOUTH_PAIRS.map(([left, right]) => (
          <PairBar
            key={pairId(left, right)}
            left={left}
            right={right}
            locked={locked}
            selected={selected}
            shapeIds={shapeIds}
            onSelectPreset={onSelectPreset}
            onAddMid={onAddMid}
            onScrub={onScrub}
            onMoveKey={onMoveKey}
            onResetKey={onResetKey}
            onDeleteKey={onDeleteKey}
            live={livePair === pairId(left, right)}
            onLive={() => setLivePair(pairId(left, right))}
          />
        ))}
      </div>
      )}
      <button
        className="act"
        type="button"
        disabled={live || busy !== '' || !status?.ready || !selected}
        onClick={onApply}
      >
        {busy === 'apply' ? 'Applying…' : 'Apply'}
      </button>
    </>
  )
}

function PairBar({
  left,
  right,
  locked,
  selected,
  shapeIds,
  onSelectPreset,
  onAddMid,
  onScrub,
  onMoveKey,
  onResetKey,
  onDeleteKey,
  live: liveBar,
  onLive,
}: {
  left: string
  right: string
  locked: boolean
  selected: string
  shapeIds: string[]
  onSelectPreset: (id: string) => void
  onAddMid: (a: string, b: string, t: number) => void
  onScrub: (a: string, b: string, t: number) => void
  onMoveKey: (id: string, t: number) => Promise<void> | void
  onResetKey: (id: string) => void
  onDeleteKey: (id: string) => void
  live: boolean
  onLive: () => void
}) {
  const [cursor, setCursor] = useState(0.5)
  const [hold, setHold] = useState<{ id: string; t: number } | null>(null)
  const [spot, setSpot] = useState<
    { kind: 'add'; x: number; y: number; t: number } | { kind: 'point'; x: number; y: number; id: string } | null
  >(null)
  const trackRef = useRef<HTMLDivElement>(null)
  const drag = useRef<KeyDrag | null>(null)
  const placed = useRef(false)
  const saved = keysOn(shapeIds, left, right)
  const savedRef = useRef(saved)
  savedRef.current = saved
  const keys = saved.map((key) => (hold?.id === key.id ? { ...key, t: hold.t } : key))
  const savedKey = saved.map((key) => `${key.id}@${key.t}`).join('|')

  function freeAt(t: number) {
    return t > KEY_GAP && t < 1 - KEY_GAP && !keys.some((key) => Math.abs(key.t - t) < KEY_GAP)
  }

  useEffect(() => {
    if (!spot) return
    const close = () => setSpot(null)
    window.addEventListener('pointerdown', close)
    window.addEventListener('scroll', close, true)
    return () => {
      window.removeEventListener('pointerdown', close)
      window.removeEventListener('scroll', close, true)
    }
  }, [spot])

  useEffect(() => {
    if (placed.current) return
    setCursor((cur) => {
      const home = homeCursor(saved)
      const taken = (spot: number) =>
        saved.some((key) => Math.abs(key.t - spot) < KEY_GAP) || spot <= KEY_GAP || spot >= 1 - KEY_GAP
      if (!taken(cur)) return cur
      return home
    })
  }, [savedKey, saved])

  useEffect(() => {
    if (liveBar || !placed.current) return
    placed.current = false
    setCursor(homeCursor(savedRef.current))
  }, [liveBar])

  function tAt(clientX: number) {
    const el = trackRef.current
    if (!el) return cursor
    const box = el.getBoundingClientRect()
    const inner = box.width - KEY_INSET * 2
    if (inner <= 1) return 0
    return Math.min(1, Math.max(0, (clientX - box.left - KEY_INSET) / inner))
  }

  function bounds(id: string) {
    const sorted = [...saved].sort((a, b) => a.t - b.t)
    const index = sorted.findIndex((key) => key.id === id)
    const lo = index <= 0 ? KEY_GAP : sorted[index - 1].t + KEY_GAP
    const hi = index < 0 || index >= sorted.length - 1 ? 1 - KEY_GAP : sorted[index + 1].t - KEY_GAP
    return [Math.min(lo, hi), Math.max(lo, hi)] as const
  }

  function placeCursor(clientX: number) {
    placed.current = true
    onLive()
    const t = tAt(clientX)
    setCursor(t)
    onScrub(left, right, t)
  }

  function menuAt(clientX: number, clientY: number, tall: boolean) {
    return {
      x: Math.min(clientX, window.innerWidth - 148),
      y: Math.min(clientY, window.innerHeight - (tall ? 88 : 52)),
    }
  }

  function onTrackMenu(e: React.MouseEvent) {
    e.preventDefault()
    e.stopPropagation()
    if (locked) return
    const point = (e.target as HTMLElement).closest('.key.stop')
    const place = menuAt(e.clientX, e.clientY, Boolean(point))
    if (point) {
      const id = point.getAttribute('data-key')
      if (id) setSpot({ kind: 'point', ...place, id })
      return
    }
    setSpot({ kind: 'add', ...place, t: tAt(e.clientX) })
  }

  function onTrackDown(e: ReactPointerEvent) {
    if (e.button !== 0 || locked || (e.target as HTMLElement).closest('.key')) return
    e.preventDefault()
    trackRef.current?.setPointerCapture(e.pointerId)
    drag.current = { kind: 'cursor', origin: e.clientX, moved: true }
    placeCursor(e.clientX)
  }

  function onCursorDown(e: ReactPointerEvent) {
    if (e.button !== 0 || locked) return
    e.stopPropagation()
    e.preventDefault()
    trackRef.current?.setPointerCapture(e.pointerId)
    drag.current = { kind: 'cursor', origin: e.clientX, moved: false }
  }

  function onKeyDown(e: ReactPointerEvent, id: string) {
    if (e.button !== 0 || locked) return
    e.stopPropagation()
    e.preventDefault()
    trackRef.current?.setPointerCapture(e.pointerId)
    drag.current = { kind: 'key', id, origin: e.clientX, moved: false }
  }

  function onTrackMove(e: ReactPointerEvent) {
    const active = drag.current
    if (!active) return
    if (Math.abs(e.clientX - active.origin) > 3) active.moved = true
    if (active.kind === 'cursor') {
      placeCursor(e.clientX)
      return
    }
    const [lo, hi] = bounds(active.id)
    const t = Math.min(hi, Math.max(lo, tAt(e.clientX)))
    setHold({ id: active.id, t })
  }

  function onTrackUp(e: ReactPointerEvent) {
    const active = drag.current
    drag.current = null
    if (!active) return
    if (Math.abs(e.clientX - active.origin) > 3) active.moved = true
    if (active.kind === 'cursor') {
      placeCursor(e.clientX)
      return
    }
    if (!active.moved) {
      setHold(null)
      onSelectPreset(active.id)
      return
    }
    const [lo, hi] = bounds(active.id)
    const t = Math.min(hi, Math.max(lo, tAt(e.clientX)))
    setHold({ id: active.id, t })
    void Promise.resolve(onMoveKey(active.id, t)).finally(() => setHold(null))
  }

  function leftOf(t: number) {
    return `calc(${KEY_INSET}px + ${t} * (100% - ${KEY_INSET * 2}px))`
  }

  const leftLabel = MOUTH_LABEL[left as keyof typeof MOUTH_LABEL]
  const rightLabel = MOUTH_LABEL[right as keyof typeof MOUTH_LABEL]

  return (
    <div className="keybar">
      <div
        className="keybar-track"
        ref={trackRef}
        onPointerDown={onTrackDown}
        onPointerMove={onTrackMove}
        onPointerUp={onTrackUp}
        onContextMenu={onTrackMenu}
      >
        <div className="keybar-line" />
        <button
          type="button"
          className={`key end${selected === left ? ' on' : ''}`}
          style={{ left: leftOf(0) }}
          disabled={locked}
          onClick={() => onSelectPreset(left)}
        >
          <i />
          <span>{leftLabel}</span>
        </button>
        {keys.map((key) => (
          <button
            key={key.id}
            type="button"
            className={`key stop${selected === key.id ? ' on' : ''}`}
            style={{ left: leftOf(key.t) }}
            disabled={locked}
            aria-label={`${leftLabel} ${rightLabel} point ${Math.round(key.t * 100)}`}
            data-key={key.id}
            onPointerDown={(e) => onKeyDown(e, key.id)}
          >
            <i />
          </button>
        ))}
        <button
          type="button"
          className={`key end${selected === right ? ' on' : ''}`}
          style={{ left: leftOf(1) }}
          disabled={locked}
          onClick={() => onSelectPreset(right)}
        >
          <i />
          <span>{rightLabel}</span>
        </button>
        <button
          type="button"
          className="key cursor"
          style={{ left: leftOf(cursor) }}
          disabled={locked}
          aria-label={`${leftLabel} ${rightLabel} in-between`}
          onPointerDown={onCursorDown}
        >
          <i />
        </button>
      </div>
      {spot ? (
        <div
          className="menu"
          style={{ left: spot.x, top: spot.y }}
          onPointerDown={(e) => e.stopPropagation()}
        >
          {spot.kind === 'point' ? (
            <>
              <button
                type="button"
                onClick={() => {
                  onResetKey(spot.id)
                  setSpot(null)
                }}
              >
                Reset
              </button>
              <button
                type="button"
                onClick={() => {
                  onDeleteKey(spot.id)
                  setSpot(null)
                }}
              >
                Delete
              </button>
            </>
          ) : (
            <button
              type="button"
              disabled={!freeAt(spot.t)}
              onClick={() => {
                const t = spot.t
                onAddMid(left, right, t)
                setSpot(null)
                if (Math.abs(cursor - t) < KEY_GAP) {
                  const step = 0.15
                  const next = [t + step, t - step, t + step * 2, t - step * 2].find((place) => freeAt(place))
                  if (next != null) {
                    placed.current = true
                    setCursor(next)
                  }
                }
              }}
            >
              Add point
            </button>
          )}
        </div>
      ) : null}
    </div>
  )
}

function IfmSection({
  live,
  busy,
  status,
  ifmPort,
  copied,
  localIps,
  onCopyAddr,
  onIfmPort,
  onSaveIfm,
}: Pick<
  SideProps,
  | 'live'
  | 'busy'
  | 'status'
  | 'ifmPort'
  | 'copied'
  | 'localIps'
  | 'onCopyAddr'
  | 'onIfmPort'
  | 'onSaveIfm'
>) {
  const ifm = status?.ifm
  return (
    <>
      <p className="side-label mix">iFacialMocap</p>
      <p className={`ifm-link${ifm?.receiving ? ' on' : live ? ' wait' : ''}`}>
        {ifm?.receiving
          ? `Live · ${Math.round(ifm.fps)} fps`
          : live
            ? 'Listening'
            : 'Idle'}
      </p>
      {localIps.map((ip) => (
        <button
          key={ip}
          type="button"
          className={`ifm-addr${ip === ifm?.primary ? ' primary' : ''}`}
          onClick={() => onCopyAddr(ip)}
        >
          <span>{ip}</span>
          <em>{copied === ip ? 'copied' : 'copy'}</em>
        </button>
      ))}
      <p className="hint">
        {ifm?.hint ||
          (live
            ? `Listening on UDP ${ifm?.port ?? 49983}`
            : 'Listen. On the iPhone, destination is this PC.')}
      </p>
      {ifm?.last_peer || ifm?.peer ? (
        <p className="hint">Phone {ifm.peer || ifm.last_peer}</p>
      ) : null}
      <input
        className="cam"
        type="text"
        inputMode="numeric"
        placeholder="UDP 49983"
        value={ifmPort}
        disabled={live || busy !== ''}
        onChange={(e) => onIfmPort(e.target.value)}
        onBlur={onSaveIfm}
      />
    </>
  )
}

function CameraSection({
  live,
  busy,
  status,
  onSetCamera,
  onRefreshCameras,
}: Pick<SideProps, 'live' | 'busy' | 'status' | 'onSetCamera' | 'onRefreshCameras'>) {
  const index = status?.camera_index ?? 0
  const listed = status?.cameras ?? []
  const current = listed.find((cam) => cam.index === index)
  // A value with no matching option makes the select show the first camera
  // while another one is chosen, and picking that first one fires no change.
  const cameras = current
    ? listed
    : [{ index, name: listed.length ? `Camera ${index} (not found)` : `Camera ${index}` }, ...listed]
  const name = current?.name ?? cameras[0].name
  return (
    <>
      <p className="side-label mix">Camera</p>
      <select
        className="cam"
        value={index}
        title={live ? `${name} (stop tracking to change camera)` : name}
        disabled={live || busy !== ''}
        onFocus={onRefreshCameras}
        onChange={(e) => onSetCamera(Number(e.target.value))}
      >
        {cameras.map((cam) => (
          <option key={cam.index} value={cam.index} disabled={listed.length > 0 && !listed.includes(cam)}>
            {cam.name}
          </option>
        ))}
      </select>
    </>
  )
}

function fmt(value: number) {
  return Number.isFinite(value) ? value.toFixed(3) : '—'
}

function IrisRaw({ hits, look }: { hits: IrisCamHit[]; look?: { x: number; y: number } | null }) {
  if (!hits.length && !look) {
    return <pre className="iris-raw">no look</pre>
  }
  return (
    <pre className="iris-raw">
      {look ? `look  x ${fmt(look.x)}  y ${fmt(look.y)}\n` : null}
      {hits.map((hit) => {
        const side = hit.side || '?'
        return `${side}  x ${fmt(hit.x)}  y ${fmt(hit.y)}  score ${fmt(hit.score)}  vis ${hit.visible ? 1 : 0}  ${hit.method || 'none'}\n`
      })}
    </pre>
  )
}

function TrackingSection({
  live,
  busy,
  status,
  onCalibrate,
}: Pick<SideProps, 'live' | 'busy' | 'status' | 'onCalibrate'>) {
  const [irisDebug, setIrisDebug] = useState(false)
  const calib = status?.calib
  const method = String(status?.iris_method || '')
  const showMethod = live && method && method !== 'none'
  return (
    <>
      <p className="side-label mix">
        Tracking
        {showMethod ? <em className="iris-pill">{method}</em> : null}
      </p>
      <div className="calib">
        <button
          type="button"
          className={`${calib?.rest ? 'saved' : ''}${calib?.capturing === 'rest' ? ' hot' : ''}`}
          disabled={!live || busy !== '' || Boolean(calib?.capturing)}
          onClick={() => onCalibrate('rest')}
        >
          {calib?.capturing === 'rest'
            ? `Setting Rest ${Math.round((calib?.progress ?? 0) * 100)}%`
            : calib?.rest
              ? 'Set Rest Again'
              : 'Set Rest'}
        </button>
        <button
          type="button"
          className={irisDebug ? 'saved' : ''}
          onClick={() => setIrisDebug((on) => !on)}
        >
          {irisDebug ? 'Hide iris' : 'Debug iris'}
        </button>
      </div>
      {irisDebug ? <IrisRaw hits={status?.iris_cam ?? []} look={status?.look} /> : null}
    </>
  )
}

const SETTINGS_OPEN_KEY = 'track-lab-settings-open'

function SettingsSection({
  feel,
  nudged,
  mirrored,
  onFeel,
  onMirror,
  onResetPoints,
}: Pick<SideProps, 'feel' | 'onFeel' | 'onMirror' | 'onResetPoints'> & {
  nudged: boolean
  mirrored: boolean
}) {
  const [open, setOpen] = useState(() => {
    try {
      return localStorage.getItem(SETTINGS_OPEN_KEY) !== '0'
    } catch {
      return true
    }
  })

  function toggle() {
    setOpen((cur) => {
      const next = !cur
      try {
        localStorage.setItem(SETTINGS_OPEN_KEY, next ? '1' : '0')
      } catch {
        /* ignore */
      }
      return next
    })
  }

  return (
    <div className={`settings-box${open ? '' : ' is-closed'}`}>
      <button
        type="button"
        className="side-label settings-toggle"
        aria-expanded={open}
        onClick={toggle}
      >
        Settings
        <i className="fold" aria-hidden="true" />
      </button>
      {open ? (
        <>
      <ul className="feel">
        {FEEL.map((row) => (
          <li key={row.key}>
            <span>{row.label}</span>
            <input
              type="range"
              min={0}
              max={row.max ?? 1}
              step={0.01}
              value={feel[row.key] ?? ZERO_FEEL[row.key]}
              onChange={(e) => onFeel(row.key, Number(e.target.value))}
            />
            <em>{(feel[row.key] ?? ZERO_FEEL[row.key] ?? 0).toFixed(2)}</em>
          </li>
        ))}
      </ul>
      <ul className="overlay-toggles">
        {OVERLAY.map((row) => (
          <li key={row.key}>
            <label>
              <input
                type="checkbox"
                checked={(feel[row.key] ?? ZERO_FEEL[row.key] ?? 0) >= 0.5}
                onChange={(e) => onFeel(row.key, e.target.checked ? 1 : 0)}
              />
              <span title={row.title}>{row.label}</span>
            </label>
          </li>
        ))}
        <li>
          <label>
            <input
              type="checkbox"
              checked={mirrored}
              onChange={(e) => onMirror(e.target.checked)}
            />
            <span title="Off: the character is your reflection (turn left, it turns to screen-left). On: it copies you like a person facing you.">Mirror</span>
          </label>
        </li>
      </ul>
      <p className="hint">Drag any overlay point to nudge it. Tracking still follows.</p>
      <button
        type="button"
        className={nudged ? 'saved' : ''}
        disabled={!nudged}
        onClick={onResetPoints}
      >
        Reset points
      </button>
        </>
      ) : null}
    </div>
  )
}

const ACTIVITY_OPEN_KEY = 'track-lab-activity-open'

function ActivitySection({
  weights,
  blink,
  hair,
}: Pick<SideProps, 'weights'> & {
  blink?: { l?: number; r?: number }
  hair?: HairPart[]
}) {
  const [open, setOpen] = useState(() => {
    try {
      return localStorage.getItem(ACTIVITY_OPEN_KEY) !== '0'
    } catch {
      return true
    }
  })

  function toggle() {
    setOpen((cur) => {
      const next = !cur
      try {
        localStorage.setItem(ACTIVITY_OPEN_KEY, next ? '1' : '0')
      } catch {
        /* ignore */
      }
      return next
    })
  }

  const rows: [string, number][] = [
    ['blink L', blink?.l ?? 0],
    ['blink R', blink?.r ?? 0],
    ...METERS.map((name) => [name, weights[name] ?? 0] as [string, number]),
  ]
  const widthOf = (side: 'l' | 'r' | 'mid') => {
    const part = (hair ?? []).find((p) => p.side === side && typeof p.width === 'number')
    return part?.width
  }
  const hairRows: [string, number | undefined][] = [
    ['hair L', widthOf('l')],
    ['hair M', widthOf('mid')],
    ['hair R', widthOf('r')],
  ]
  return (
    <div className={`settings-box${open ? '' : ' is-closed'}`}>
      <button
        type="button"
        className="side-label settings-toggle"
        aria-expanded={open}
        onClick={toggle}
      >
        Activity
        <i className="fold" aria-hidden="true" />
      </button>
      {open ? (
        <>
      <ul className="meters">
        {rows.map(([name, value]) => (
          <li key={name}>
            <span>{name}</span>
            <i>
              <b style={{ width: `${Math.round(Math.min(1, Math.max(0, value)) * 100)}%` }} />
            </i>
            <em>{value.toFixed(2)}</em>
          </li>
        ))}
      </ul>
      <p className="side-label mix">Hair width</p>
      <ul className="meters hair-width">
        {hairRows.map(([name, value]) => {
          const w = value ?? 1
          // 1.0 sits at the center; bar grows right when wider, left when narrower.
          const pct = Math.round(Math.min(1, Math.max(0, (w - 0.5) / 1.0)) * 100)
          return (
            <li key={name} className={value === undefined ? 'off' : ''}>
              <span>{name}</span>
              <i className="mid">
                <b style={{ width: `${pct}%` }} />
              </i>
              <em>{value === undefined ? '—' : `×${w.toFixed(2)}`}</em>
            </li>
          )
        })}
      </ul>
        </>
      ) : null}
    </div>
  )
}
