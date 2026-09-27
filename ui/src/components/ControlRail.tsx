import { useEffect, useRef, useState } from 'react'
import {
  labBannerError,
  labSourceOf,
  type AppStatus,
  type CameraInfo,
  type CatalogOffer,
  type CharacterCard,
  type CharacterExportResult,
  type CharacterInfo,
  type CharacterMeta,
  type Checkpoint,
  type LabFeel as LabFeelValues,
  type LabIfm,
  type CharacterLoadResult,
  type LabStatus,
} from '../api'
import { DEVELOPER } from '../developer'
import { CharacterLibrary } from './CharacterLibrary'
import { LabFeel, MixMeters } from './LabFeel'
import { TravelBox } from './TravelBox'
import { Lamp, ProgressMeter, Toggle } from './widgets'

type Props = {
  status: AppStatus | null
  checkpoints: Checkpoint[]
  catalogOffers?: CatalogOffer[]
  cameras: CameraInfo[]
  refPath: string
  error: string
  onRefPath: (v: string) => void
  onCheckpoint: (path: string) => void
  onBrowseCheckpoint: () => void
  onUploadRef: (file: File) => void
  onApplyRef: () => void
  characters: CharacterCard[]
  charactersCreating: boolean
  createStillUrl?: string
  onRefreshCharacters: () => void
  onCreateCharacter: (file: File) => Promise<CharacterCard | void | undefined>
  onLoadCharacter: (
    id: string,
    opts?: { repair?: boolean },
  ) => Promise<CharacterLoadResult | void | undefined>
  onRemoveCharacter: (id: string) => void
  onRenameCharacter: (id: string, name: string) => void
  onImportCharacter: (file: File) => Promise<CharacterCard>
  onExportCharacter: (id: string, name: string) => Promise<CharacterExportResult>
  onRevealCharacter: (id: string) => Promise<unknown>
  onCharacterInfo: (id: string) => Promise<CharacterInfo>
  onCharacterMeta: (meta: CharacterMeta) => Promise<CharacterCard>
  onSettings: (patch: Partial<AppStatus>) => void
  onToggleTracking: () => void
  onCalibrate: () => void
  onGenerate: () => void
  onToggleStream: () => void
  onTogglePause: () => void
  onToggleVirtualCam: () => void
  lab: LabStatus | null
  onLabFeel: (patch: Partial<LabFeelValues>) => void
  onLabCalibrate: (id: string) => void
  onLabSource: (source: 'camera' | 'ifm') => void
  onCamera: (index: number) => void
  onRefreshCameras?: () => void
  onRefreshCheckpoints?: () => void
  onDownloadModel?: (name: string) => void
  onLabIfmPort: (port: number) => void
  onReloadBackend: () => void
}

function clampSteps(raw: string): number | null {
  const n = Number(raw)
  if (!Number.isFinite(n)) return null
  return Math.max(1, Math.min(50, Math.round(n)))
}

/** How the character looks: the model's own knobs. */
const TUNE_DEFAULTS = {
  steps: 1,
  pose_cfg: 1,
  id_cfg: 1,
  frame_blend: 0.58,
  hold_last: true,
}

/** How fast it runs and how much GPU it takes. Compile is left alone: turning it on recompiles. */
const PERF_DEFAULTS = {
  max_fps: 0,
  interpolate: true,
  inbetweens: 1,
}

const STREAM_DEFAULTS = { ...TUNE_DEFAULTS, ...PERF_DEFAULTS }

const HOLD_LAST_TITLE =
  'Start each gen from a light mix of the last picture. Small moves stay consistent; a turn or new character drops the mix so the old face does not stick. Off = every frame is a fresh still from noise.'

const INBETWEENS_TITLE =
  'How many extra pictures to print between generated keys. 0 = keys only. 1 = one mid frame. Ignored when Interpolate is off.'

const MAX_FPS_TITLE =
  'Cap on generated frames per second. The GPU idles between frames, so a cap leaves headroom for games or OBS. Off = as fast as the GPU goes.'

const INTERPOLATE_TITLE =
  'Print optical-flow frames between DiT keys so motion looks smoother. Off = keys only. Runs off the generate thread so it does not steal Generate FPS.'

const MODEL_SLIDERS: {
  key: 'pose_cfg' | 'id_cfg' | 'frame_blend'
  label: string
  min: number
  max: number
  title: string
}[] = [
  {
    key: 'pose_cfg',
    label: 'Pose follow',
    min: 0,
    max: 6,
    title: 'This 1–2 step model already mixes pose in the joint pass. Fast keeps this at 1.0 — raising it splits pose off the still and melts the face.',
  },
  {
    key: 'id_cfg',
    label: 'Reference lock',
    min: 0,
    max: 6,
    title: 'This 1–2 step model already mixes the still in the joint pass. Fast keeps this at 1.0.',
  },
  {
    key: 'frame_blend',
    label: 'Snap',
    min: 0.05,
    max: 1,
    title: `How much of the new frame is shown. Default ${STREAM_DEFAULTS.frame_blend.toFixed(2)}. 1 = no leftover. Lower blends the last picture in, so hair can linger after a turn.`,
  },
]

function ifmListenLine(ifm?: LabIfm): string {
  if (!ifm) return ''
  if (ifm.receiving && ifm.peer) return `Live · ${ifm.peer}`
  if (ifm.listening) return 'Waiting for the phone'
  return ''
}

export function ControlRail(props: Props) {
  const s = props.status
  const busy = Boolean(s?.busy)
  const streaming = Boolean(s?.streaming)
  const paused = Boolean(s?.paused)
  const interpolate = s?.interpolate !== false
  const tracking = Boolean(s?.tracking) || Boolean(props.lab?.live)
  const virtualCam = Boolean(s?.virtual_cam)
  const progress = Number(s?.progress ?? 0)
  const showProgress =
    busy && Boolean(s?.progress_kind) && (progress > 0 || Boolean(s?.progress_label))

  // Draft while typing — committing every keystroke was snapping the field back.
  const [stepsDraft, setStepsDraft] = useState(String(s?.steps ?? STREAM_DEFAULTS.steps))
  const [cfgDraft, setCfgDraft] = useState({
    pose_cfg: s?.pose_cfg ?? STREAM_DEFAULTS.pose_cfg,
    id_cfg: s?.id_cfg ?? STREAM_DEFAULTS.id_cfg,
    frame_blend: s?.frame_blend ?? STREAM_DEFAULTS.frame_blend,
  })
  const cfgDrag = useRef(false)
  const cfgTimer = useRef<number | null>(null)
  const [railPane, setRailPane] = useState<'desk' | 'settings'>('desk')
  const [ifmPort, setIfmPort] = useState(String(props.lab?.ifm?.port || 49983))
  const [copiedDest, setCopiedDest] = useState(false)
  const copyTimer = useRef<number | null>(null)
  useEffect(() => {
    setStepsDraft(String(s?.steps ?? STREAM_DEFAULTS.steps))
  }, [s?.steps])
  useEffect(() => {
    if (cfgDrag.current) return
    setCfgDraft({
      pose_cfg: s?.pose_cfg ?? STREAM_DEFAULTS.pose_cfg,
      id_cfg: s?.id_cfg ?? STREAM_DEFAULTS.id_cfg,
      frame_blend: s?.frame_blend ?? STREAM_DEFAULTS.frame_blend,
    })
  }, [s?.pose_cfg, s?.id_cfg, s?.frame_blend])
  useEffect(() => {
    return () => {
      if (cfgTimer.current != null) window.clearTimeout(cfgTimer.current)
      if (copyTimer.current != null) window.clearTimeout(copyTimer.current)
    }
  }, [])
  useEffect(() => {
    if (props.lab?.ifm?.port != null) setIfmPort(String(props.lab.ifm.port))
  }, [props.lab?.ifm?.port])
  const overlayParts = {
    show_outline: s?.show_outline === true,
    show_brows: s?.show_brows === true,
    show_eyes: s?.show_eyes === true,
    show_nose: s?.show_nose === true,
    show_mouth: s?.show_mouth === true,
    show_iris_overlay: s?.show_iris_overlay === true,
    show_skeleton: s?.show_skeleton === true,
    show_hair: s?.show_hair === true,
  }
  const overlayOn = Object.values(overlayParts).some(Boolean)
  const labOnline = Boolean(props.lab?.online)
  const labLive = Boolean(props.lab?.live)
  const labError = labBannerError(props.lab)
  const restCalib = props.lab?.calib
  const restProgress = Number(restCalib?.progress ?? 0)
  const capturingRest = Boolean(restCalib?.capturing) && restProgress < 1
  const restLabel = capturingRest
    ? `Calibrating ${Math.round(restProgress * 100)}%`
    : 'Calibrate'
  const canCalibrate = tracking || labLive
  const trackSource = labSourceOf(props.lab)
  const trackCameras: CameraInfo[] =
    labOnline && props.lab?.cameras?.length ? props.lab.cameras : props.cameras
  const trackCameraIndex = labOnline
    ? (props.lab?.camera_index ?? s?.camera_index ?? 0)
    : (s?.camera_index ?? 0)
  const deviceLocked = tracking || (labLive && trackSource === 'camera')
  const ifm = props.lab?.ifm
  const destIp = ifm?.primary || ''
  const ifmLine = ifmListenLine(ifm)

  function copyDest() {
    if (!destIp) return
    void navigator.clipboard.writeText(destIp).then(
      () => {
        setCopiedDest(true)
        if (copyTimer.current != null) window.clearTimeout(copyTimer.current)
        copyTimer.current = window.setTimeout(() => setCopiedDest(false), 1200)
      },
      () => {},
    )
  }

  function commitSteps() {
    const next = clampSteps(stepsDraft)
    if (next == null) {
      setStepsDraft(String(s?.steps ?? STREAM_DEFAULTS.steps))
      return
    }
    setStepsDraft(String(next))
    if (next !== (s?.steps ?? STREAM_DEFAULTS.steps)) {
      props.onSettings({ steps: next })
    }
  }

  function resetPerfDefaults() {
    props.onSettings({ ...PERF_DEFAULTS })
  }

  function resetStreamDefaults() {
    if (cfgTimer.current != null) window.clearTimeout(cfgTimer.current)
    cfgDrag.current = false
    setStepsDraft(String(STREAM_DEFAULTS.steps))
    setCfgDraft({
      pose_cfg: STREAM_DEFAULTS.pose_cfg,
      id_cfg: STREAM_DEFAULTS.id_cfg,
      frame_blend: STREAM_DEFAULTS.frame_blend,
    })
    props.onSettings({ ...TUNE_DEFAULTS })
  }

  function commitCfg(key: 'pose_cfg' | 'id_cfg' | 'frame_blend', value: number) {
    setCfgDraft((cur) => ({ ...cur, [key]: value }))
    if (cfgTimer.current != null) window.clearTimeout(cfgTimer.current)
    cfgDrag.current = true
    cfgTimer.current = window.setTimeout(() => {
      cfgDrag.current = false
      props.onSettings({ [key]: value })
    }, 80)
  }

  const modelProgress =
    showProgress && (s?.progress_kind === 'model' || s?.progress_kind === 'download') ? (
      <ProgressMeter
        label={
          s.progress_label ||
          (s.progress_kind === 'download' ? 'Downloading model' : 'Loading model')
        }
        value={progress}
      />
    ) : null

  const listed = props.characters.find((c) => c.id === s?.character_id)
  const charName = listed?.name || (s?.character_id ? s.character_name : '') || ''

  return (
    <aside className="rail">
      <div className="rail-board">
        <div className="rail-panes" role="tablist" aria-label="Rail">
          <button
            type="button"
            role="tab"
            aria-selected={railPane === 'desk'}
            className={railPane === 'desk' ? 'on' : ''}
            onClick={() => setRailPane('desk')}
          >
            Desk
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={railPane === 'settings'}
            className={railPane === 'settings' ? 'on' : ''}
            onClick={() => setRailPane('settings')}
          >
            Settings
          </button>
        </div>

        {railPane === 'desk' ? (
          <div className="rail-desk">
        <div className="rail-controls">
          {modelProgress}
          <section className="group">
            {DEVELOPER ? (
              <div className="row">
                <button
                  type="button"
                  className="btn"
                  onClick={props.onBrowseCheckpoint}
                  disabled={busy || streaming}
                >
                  Browse model…
                </button>
              </div>
            ) : null}
            <label className="field">
              <span>Model</span>
              <select
                value={
                  props.checkpoints.find(
                    (c) => c.label === s?.checkpoint || c.path === s?.checkpoint,
                  )?.path ||
                  props.checkpoints[0]?.path ||
                  ''
                }
                onChange={(e) => props.onCheckpoint(e.target.value)}
                onFocus={() => props.onRefreshCheckpoints?.()}
                disabled={busy || streaming || !props.checkpoints.length}
              >
                {props.checkpoints.length === 0 && <option value="">No models</option>}
                {props.checkpoints.map((c) => (
                  <option key={c.path} value={c.path}>
                    {c.source === 'local' ? `${c.label} (local)` : c.label}
                  </option>
                ))}
              </select>
            </label>
            {(props.catalogOffers ?? []).length ? (
              <ul className="model-offers">
                {(props.catalogOffers ?? []).map((offer) => (
                  <li key={offer.name}>
                    <button
                      type="button"
                      className={`model-offer${offer.is_new ? ' is-new' : ''}`}
                      disabled={busy || streaming}
                      onClick={() => props.onDownloadModel?.(offer.name)}
                      title={
                        offer.is_new
                          ? 'New on the hub (under 30 days). Download into models/dit.'
                          : 'On the hub and not on disk yet.'
                      }
                    >
                      <span className="model-offer-badge">{offer.badge}</span>
                      <span className="model-offer-name">{offer.label}</span>
                    </button>
                  </li>
                ))}
              </ul>
            ) : null}
            {DEVELOPER ? (
              <>
                <label className="field">
                  <span>Image path</span>
                  <input
                    value={props.refPath}
                    onChange={(e) => props.onRefPath(e.target.value)}
                    placeholder="Path to reference still"
                    disabled={busy || streaming}
                  />
                </label>
                <div className="row">
                  <label
                    className={`btn ghost file-btn${busy || streaming ? ' disabled' : ''}`}
                    aria-disabled={busy || streaming}
                  >
                    Browse…
                    <input
                      type="file"
                      accept="image/*"
                      hidden
                      disabled={busy || streaming}
                      onChange={(e) => {
                        const f = e.target.files?.[0]
                        if (f) props.onUploadRef(f)
                        e.target.value = ''
                      }}
                    />
                  </label>
                  <button
                    type="button"
                    className="btn"
                    onClick={props.onApplyRef}
                    disabled={busy || streaming || !props.refPath.trim()}
                  >
                    Apply ref
                  </button>
                </div>
              </>
            ) : null}
          </section>
          <section className="group">
            <LabFeel
              lab={props.lab}
              busy={Boolean(s?.fast_warming) || Boolean(s?.track_busy)}
              onFeel={props.onLabFeel}
              headerExtra={
                <Toggle
                  className="track-mirror"
                  label="Mirror"
                  checked={Boolean(s?.mirror)}
                  onChange={(v) => props.onSettings({ mirror: v })}
                  title="Mirror look and head turn"
                />
              }
              actions={
                <>
                  <div className="row track-run">
                    <button
                      type="button"
                      className={tracking ? 'btn is-on' : 'btn primary'}
                      onClick={props.onToggleTracking}
                      disabled={Boolean(s?.track_busy)}
                    >
                      {tracking ? 'Stop tracking' : 'Start tracking'}
                    </button>
                    <button
                      type="button"
                      className="btn"
                      onClick={() =>
                        tracking || labOnline
                          ? props.onLabCalibrate('rest')
                          : props.onCalibrate()
                      }
                      disabled={
                        Boolean(s?.track_busy) || capturingRest || !canCalibrate
                      }
                      title="Capture rest for tracking"
                    >
                      {restLabel}
                    </button>
                  </div>
                  {restCalib?.hint ? <p className="hint">{restCalib.hint}</p> : null}
                  {DEVELOPER && s?.body_label ? <p className="hint">{s.body_label}</p> : null}
                </>
              }
            >
              <div className="track-cam-row">
                <div className="lab-tabs" role="tablist" aria-label="Tracking input">
                  <button
                    type="button"
                    className={trackSource === 'camera' ? 'on' : ''}
                    aria-pressed={trackSource === 'camera'}
                    onClick={() => props.onLabSource('camera')}
                  >
                    Camera
                  </button>
                  <button
                    type="button"
                    className={trackSource === 'ifm' ? 'on brand' : 'brand'}
                    aria-pressed={trackSource === 'ifm'}
                    onClick={() => props.onLabSource('ifm')}
                  >
                    iFacialMocap
                  </button>
                </div>
                {trackSource === 'ifm' || labOnline ? (
                  labError ? (
                    <p className="status-error">{labError}</p>
                  ) : !labOnline && trackSource === 'ifm' ? (
                    <p className="hint">Start Track Lab to listen for iFacialMocap.</p>
                  ) : null
                ) : null}
                {trackSource === 'ifm' ? (
                  <div className="ifm-listen">
                    {destIp ? (
                      <button
                        type="button"
                        className="ifm-dest"
                        title="Copy this PC’s address for iFacialMocap"
                        onClick={copyDest}
                      >
                        <span>{copiedDest ? 'Copied' : 'This PC'}</span>
                        <em className="mono">{destIp}</em>
                      </button>
                    ) : (
                      <p className="hint">Same Wi-Fi as the iPhone.</p>
                    )}
                    <label className="field inline">
                      <span>Port</span>
                      <input
                        className="mono"
                        value={ifmPort}
                        disabled={labLive}
                        onChange={(e) => setIfmPort(e.target.value.replace(/[^\d]/g, ''))}
                        onBlur={() => {
                          const port = Number(ifmPort)
                          if (port > 0 && port !== Number(props.lab?.ifm?.port)) {
                            props.onLabIfmPort(port)
                          }
                        }}
                      />
                    </label>
                    {ifmLine ? (
                      <p className={ifm?.receiving ? 'ifm-status is-live' : 'ifm-status'}>
                        {ifmLine}
                      </p>
                    ) : null}
                  </div>
                ) : (
                  <label className="field track-cam-field">
                    <select
                      className="camera-select"
                      aria-label="Camera"
                      value={trackCameraIndex}
                      onChange={(e) => props.onCamera(Number(e.target.value))}
                      onFocus={() => {
                        if (!labOnline || trackCameras.length === 0) {
                          props.onRefreshCameras?.()
                        }
                      }}
                      disabled={deviceLocked}
                    >
                      {trackCameras.length === 0 && <option value={0}>Camera 0</option>}
                      {trackCameras.map((cam) => (
                        <option key={cam.index} value={cam.index}>
                          {cam.name}
                        </option>
                      ))}
                    </select>
                  </label>
                )}
              </div>
            </LabFeel>
          </section>
        </div>

          <section className="char-stage" aria-label="Toon">
            <div className="char-stage-bar">
              <h2 className="group-title">Toon</h2>
              <span className="char-stage-name">{charName}</span>
            </div>
            {showProgress &&
            !props.charactersCreating &&
            (s?.progress_kind === 'character' || s?.progress_kind === 'reference') ? (
              <ProgressMeter
                label={s.progress_label || 'Loading character'}
                value={progress}
              />
            ) : null}
            <CharacterLibrary
              currentId={s?.character_id || ''}
              characters={props.characters}
              creating={props.charactersCreating}
              createProgress={
                props.charactersCreating
                  ? Math.max(progress, 0.04)
                  : s?.progress_kind === 'character' || s?.progress_kind === 'reference'
                    ? progress
                    : 0
              }
              createLabel={
                props.charactersCreating
                  ? s?.progress_label || 'Creating character…'
                  : s?.progress_kind === 'character' || s?.progress_kind === 'reference'
                    ? s.progress_label || 'Creating character…'
                    : 'Creating character…'
              }
              createStillUrl={props.createStillUrl}
              busy={busy || streaming}
              error={props.error || s?.error || ''}
              onLoad={props.onLoadCharacter}
              onRemove={props.onRemoveCharacter}
              onRename={props.onRenameCharacter}
              onCreate={props.onCreateCharacter}
              onImport={props.onImportCharacter}
              onExport={props.onExportCharacter}
              onReveal={props.onRevealCharacter}
              onInfo={props.onCharacterInfo}
              onMeta={props.onCharacterMeta}
              onRefresh={props.onRefreshCharacters}
              travel={s?.travel_box}
              onTravel={(travel_box) => props.onSettings({ travel_box })}
            />
          </section>

          <MixMeters lab={props.lab} />

          <section className="group stream-panel">
            <div className="group-head">
              <h2 className="group-title">Stream</h2>
              <Lamp
                on={streaming || virtualCam}
                pending={paused}
                label={
                  paused
                    ? 'Stream paused'
                    : streaming
                      ? 'Streaming'
                      : virtualCam
                        ? 'Virtual camera on'
                        : 'Stream idle'
                }
              />
            </div>
            {showProgress &&
            (s?.progress_kind === 'warmup' || s?.progress_kind === 'compile') ? (
              <ProgressMeter
                label={s.progress_label || 'Preparing stream'}
                value={progress}
              />
            ) : null}
            <div className="row stream-run">
              <button
                type="button"
                className={streaming ? 'btn is-on' : 'btn primary'}
                onClick={props.onToggleStream}
                disabled={(busy && !streaming) || Boolean(s?.fast_warming)}
              >
                {streaming ? 'Stop stream' : 'Start stream'}
              </button>
              <button
                type="button"
                className={paused ? 'btn primary' : 'btn'}
                onClick={props.onTogglePause}
                disabled={!streaming || Boolean(s?.fast_warming)}
                title={paused ? 'Resume generating frames' : 'Hold the last picture'}
              >
                {paused ? 'Resume' : 'Pause'}
              </button>
              <button
                type="button"
                className={virtualCam ? 'btn is-on' : 'btn'}
                onClick={props.onToggleVirtualCam}
                disabled={busy && !virtualCam}
                title="Send avatar frames to a virtual camera for OBS"
              >
                {virtualCam ? 'Stop cam' : 'Start cam'}
              </button>
            </div>
            <div className="row">
              <button
                type="button"
                className="btn ghost"
                onClick={() => props.onGenerate()}
                disabled={busy || streaming || Boolean(s?.fast_warming)}
              >
                Generate once
              </button>
            </div>
            {s?.virtual_cam_error ? (
              <p className="status-error">{s.virtual_cam_error}</p>
            ) : null}
          </section>
          </div>
        ) : (
          <div className="rail-controls">
            <section className="group stream-tune">
              <div className="group-head">
                <h2 className="group-title">Tune</h2>
                <button
                  type="button"
                  className="btn ghost compact"
                  onClick={resetStreamDefaults}
                  title="Steps 1, Pose follow 1.0, Reference lock 1.0, Snap 0.58, Hold last on"
                >
                  Defaults
                </button>
              </div>
              <ul className="lab-sliders stream-sliders">
                <li className="stream-steps">
                  <span title="Denoise passes per frame. This model is meant for 1 or 2. Each pass sees the pose and the original still together.">
                    Steps
                  </span>
                  <input
                    className="mono"
                    type="number"
                    min={1}
                    max={50}
                    value={stepsDraft}
                    onChange={(e) => setStepsDraft(e.target.value)}
                    onBlur={() => commitSteps()}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') {
                        e.currentTarget.blur()
                      }
                    }}
                    disabled={streaming}
                    aria-label="Steps"
                  />
                </li>
                {MODEL_SLIDERS.map((row) => (
                  <li key={row.key}>
                    <span title={row.title}>{row.label}</span>
                    <input
                      type="range"
                      min={row.min}
                      max={row.max}
                      step={0.05}
                      value={cfgDraft[row.key]}
                      title={row.title}
                      onChange={(e) => commitCfg(row.key, Number(e.target.value))}
                    />
                    <em className="mono">{cfgDraft[row.key].toFixed(2)}</em>
                  </li>
                ))}
              </ul>
              <div className="stream-compile">
                <Toggle
                  label="Hold last"
                  checked={s?.hold_last !== false}
                  onChange={(v) => props.onSettings({ hold_last: v })}
                  title={HOLD_LAST_TITLE}
                />
              </div>
            </section>
            <section className="group stream-perf">
              <div className="group-head">
                <h2 className="group-title">Performance</h2>
                <button
                  type="button"
                  className="btn ghost compact"
                  onClick={resetPerfDefaults}
                  title="Max FPS off, Interpolate on, Inbetweens 1"
                >
                  Defaults
                </button>
              </div>
              <ul className="lab-sliders stream-sliders">
                <li>
                  <span title={MAX_FPS_TITLE}>Max FPS</span>
                  <input
                    type="range"
                    min={0}
                    max={60}
                    step={1}
                    value={s?.max_fps ?? STREAM_DEFAULTS.max_fps}
                    title={MAX_FPS_TITLE}
                    onChange={(e) => props.onSettings({ max_fps: Number(e.target.value) })}
                  />
                  <em className="mono">
                    {(s?.max_fps ?? STREAM_DEFAULTS.max_fps) > 0
                      ? s?.max_fps ?? STREAM_DEFAULTS.max_fps
                      : 'Off'}
                  </em>
                </li>
                <li>
                  <span title={INBETWEENS_TITLE}>Inbetweens</span>
                  <input
                    type="range"
                    min={0}
                    max={3}
                    step={1}
                    value={s?.inbetweens ?? STREAM_DEFAULTS.inbetweens}
                    title={INBETWEENS_TITLE}
                    disabled={!interpolate}
                    onChange={(e) =>
                      props.onSettings({ inbetweens: Number(e.target.value) })
                    }
                  />
                  <em className="mono">{s?.inbetweens ?? STREAM_DEFAULTS.inbetweens}</em>
                </li>
              </ul>
              <div className="stream-compile">
                <Toggle
                  label="Interpolate"
                  checked={interpolate}
                  onChange={(v) => props.onSettings({ interpolate: v })}
                  title={INTERPOLATE_TITLE}
                />
                <Toggle
                  label="Compile"
                  checked={Boolean(s?.compile_model)}
                  onChange={(v) => props.onSettings({ compile_model: v })}
                  title="Speed boost: builds a version of the model tuned for your GPU when the stream starts. The first build can take a minute; off runs at normal speed."
                  light={
                    !s?.compile_model
                      ? 'off'
                      : s?.fast_warming || s?.compile_status === 'pending'
                        ? 'pending'
                        : s?.compile_on
                          ? 'on'
                          : s?.compile_status === 'fail' || s?.compile_status === 'skip'
                            ? 'fail'
                            : 'off'
                  }
                  lightTitle={
                    s?.compile_detail ||
                    (s?.compile_on
                      ? 'Speed boost on'
                      : s?.fast_warming
                        ? 'Building the speed boost'
                        : s?.compile_status === 'fail'
                          ? 'Speed boost unavailable — running at normal speed'
                          : s?.compile_status === 'skip'
                            ? 'Speed boost needs an NVIDIA GPU'
                            : s?.compile_model
                              ? 'Speed boost builds when the stream starts'
                              : 'Speed boost off')
                  }
                />
              </div>
            </section>
            <section className="group">
              <TravelBox
                value={s?.travel_box}
                disabled={busy}
                show={s?.show_limiters === true}
                onShow={(show_limiters) => props.onSettings({ show_limiters })}
                onChange={(travel_box) => props.onSettings({ travel_box })}
              />
            </section>
            <section className="group">
              <div className="group-head">
                <h2 className="group-title">Overlay</h2>
                <Toggle
                  className="overlay-master"
                  label="Show"
                  checked={overlayOn}
                  onChange={(v) =>
                    props.onSettings({
                      show_mesh: v,
                      show_hair: v,
                      show_outline: v,
                      show_brows: v,
                      show_eyes: v,
                      show_nose: v,
                      show_mouth: v,
                      show_iris_overlay: v,
                      show_skeleton: v,
                    })
                  }
                />
              </div>
              {overlayOn ? (
                <div className="overlay-controls">
                  <div className="overlay-parts">
                    {(
                      [
                        ['show_outline', 'Outline'],
                        ['show_brows', 'Brows'],
                        ['show_eyes', 'Eyes'],
                        ['show_nose', 'Nose'],
                        ['show_mouth', 'Mouth'],
                        ['show_iris_overlay', 'Iris'],
                        ['show_skeleton', 'Skeleton'],
                        ['show_hair', 'Hair'],
                      ] as const
                    ).map(([key, label]) => (
                      <Toggle
                        key={key}
                        label={label}
                        checked={overlayParts[key]}
                        onChange={(v) => props.onSettings({ [key]: v })}
                      />
                    ))}
                  </div>
                </div>
              ) : null}
            </section>
            <section className="group">
              <h2 className="group-title">App</h2>
              <div className="row">
                <button
                  type="button"
                  className="btn"
                  onClick={props.onReloadBackend}
                  title="Restart Python so code changes load. A small hold window stays up until the desk comes back. Track Lab stays running."
                >
                  Reload backend
                </button>
              </div>
            </section>
            {DEVELOPER ? (
              <section className="group">
                <div className="group-head">
                  <h2 className="group-title">Developer</h2>
                </div>
                <label className="field inline">
                  <span>Track FPS</span>
                  <input
                    className="mono"
                    type="number"
                    min={0.5}
                    max={30}
                    step={0.5}
                    value={s?.track_fps ?? 2}
                    onChange={(e) => props.onSettings({ track_fps: Number(e.target.value) })}
                    disabled={streaming && Boolean(s?.auto_sync_track)}
                  />
                </label>
                <div className="toggles">
                  <Toggle
                    label="Fast"
                    checked={Boolean(s?.fast_mode)}
                    onChange={(v) => props.onSettings({ fast_mode: v })}
                    disabled={streaming || Boolean(s?.fast_warming)}
                    light={!s?.fast_mode ? 'off' : s?.fast_warming ? 'pending' : 'on'}
                    lightTitle={s?.fast_mode ? 'Fast path on' : 'Fast off'}
                  />
                  <Toggle
                    label="Batch ×2"
                    checked={Boolean(s?.batch2)}
                    onChange={(v) => props.onSettings({ batch2: v })}
                    disabled={streaming || Boolean(s?.fast_warming)}
                  />
                  <Toggle
                    label="Auto sync track"
                    checked={Boolean(s?.auto_sync_track)}
                    onChange={(v) => props.onSettings({ auto_sync_track: v })}
                  />
                  <Toggle
                    label="Iris"
                    checked={Boolean(s?.use_iris)}
                    onChange={(v) => props.onSettings({ use_iris: v })}
                  />
                  <Toggle
                    label="Body"
                    checked={Boolean(s?.use_body)}
                    onChange={(v) => props.onSettings({ use_body: v })}
                  />
                  <Toggle
                    label="Drive pose"
                    checked={Boolean(s?.drive_pose)}
                    onChange={(v) => props.onSettings({ drive_pose: v })}
                  />
                </div>
              </section>
            ) : null}
          </div>
        )}
      </div>
      {props.error || s?.error ? (
        <footer className="rail-status">
          <p className="status-error">{props.error || s?.error}</p>
        </footer>
      ) : null}
    </aside>
  )
}
