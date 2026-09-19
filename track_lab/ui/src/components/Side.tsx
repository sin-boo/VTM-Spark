import { useState } from 'react'
import type { FeelSettings, IrisCamHit, LabStatus, MixWeights, MouthPreset } from '../api'
import { FEEL, METERS, OVERLAY, ZERO_FEEL, type Busy } from '../constants'

export type SideProps = {
  live: boolean
  busy: Busy
  source: 'camera' | 'ifm'
  selected: string
  status: LabStatus | null
  presets: MouthPreset[]
  feel: FeelSettings
  weights: MixWeights
  ifmPort: string
  copied: string
  localIps: string[]
  onSelectPreset: (id: string) => void
  onApply: () => void
  onPresetMenu: (id: string, x: number, y: number) => void
  onCalibrate: (id: string) => void
  onFeel: (key: keyof FeelSettings, value: number) => void
  onMirror: (on: boolean) => void
  onResetPoints: () => void
  onCopyAddr: (ip: string) => void
  onIfmPort: (value: string) => void
  onSaveIfm: () => void
  onSetCamera: (index: number) => void
}

export function Side({
  live,
  busy,
  source,
  selected,
  status,
  presets,
  feel,
  weights,
  ifmPort,
  copied,
  localIps,
  onSelectPreset,
  onApply,
  onPresetMenu,
  onCalibrate,
  onFeel,
  onMirror,
  onResetPoints,
  onCopyAddr,
  onIfmPort,
  onSaveIfm,
  onSetCamera,
}: SideProps) {
  return (
    <aside className="side">
      <MouthSection
        live={live}
        busy={busy}
        selected={selected}
        status={status}
        presets={presets}
        onSelectPreset={onSelectPreset}
        onApply={onApply}
        onPresetMenu={onPresetMenu}
      />
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
      <ActivitySection weights={weights} blink={status?.blink} />
    </aside>
  )
}

function MouthSection({
  live,
  busy,
  selected,
  status,
  presets,
  onSelectPreset,
  onApply,
  onPresetMenu,
}: Pick<
  SideProps,
  | 'live'
  | 'busy'
  | 'selected'
  | 'status'
  | 'presets'
  | 'onSelectPreset'
  | 'onApply'
  | 'onPresetMenu'
>) {
  const locked = live || busy !== '' || (!status?.ready && !status?.points?.length)
  return (
    <>
      <p className="side-label">Mouth</p>
      <ul className="presets">
        {presets.map((preset) => (
          <li key={preset.id}>
            <label
              className={preset.id === selected ? 'on' : ''}
              onClick={(e) => {
                e.preventDefault()
                if (locked) return
                onSelectPreset(preset.id)
              }}
              onContextMenu={(e) => {
                e.preventDefault()
                if (locked) return
                onPresetMenu(preset.id, e.clientX, e.clientY)
              }}
            >
              <input
                type="radio"
                name="mouth-preset"
                checked={selected === preset.id}
                onChange={() => onSelectPreset(preset.id)}
                disabled={locked}
                tabIndex={locked ? -1 : 0}
              />
              <span className="mark" />
              <span>{preset.label}</span>
              {preset.ready ? <em>saved</em> : null}
            </label>
          </li>
        ))}
      </ul>
      <button
        className="act"
        type="button"
        disabled={live || busy !== '' || !status?.ready}
        onClick={onApply}
      >
        {busy === 'apply' ? 'Applying…' : 'Apply'}
      </button>
    </>
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
}: Pick<SideProps, 'live' | 'busy' | 'status' | 'onSetCamera'>) {
  const cameras = status?.cameras?.length
    ? status.cameras
    : [{ index: status?.camera_index ?? 0, name: `Camera ${status?.camera_index ?? 0}` }]
  return (
    <>
      <p className="side-label mix">Camera</p>
      <select
        className="cam"
        value={status?.camera_index ?? 0}
        disabled={live || busy !== ''}
        onChange={(e) => onSetCamera(Number(e.target.value))}
      >
        {cameras.map((cam) => (
          <option key={cam.index} value={cam.index}>
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
            <em>{(feel[row.key] ?? ZERO_FEEL[row.key]).toFixed(2)}</em>
          </li>
        ))}
      </ul>
      <ul className="overlay-toggles">
        {OVERLAY.map((row) => (
          <li key={row.key}>
            <label>
              <input
                type="checkbox"
                checked={(feel[row.key] ?? ZERO_FEEL[row.key]) >= 0.5}
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

function ActivitySection({
  weights,
  blink,
}: Pick<SideProps, 'weights'> & { blink?: { l?: number; r?: number } }) {
  const rows: [string, number][] = [
    ['blink L', blink?.l ?? 0],
    ['blink R', blink?.r ?? 0],
    ...METERS.map((name) => [name, weights[name] ?? 0] as [string, number]),
  ]
  return (
    <>
      <p className="side-label mix">Activity</p>
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
    </>
  )
}
