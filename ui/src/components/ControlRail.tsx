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
import { LANGUAGES, useI18n, type I18n, type Lang, type MessageKey } from '../i18n'
import { CharacterLibrary } from './CharacterLibrary'
import { GpuPicker } from './GpuPicker'
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
  /** Resolves true once saved; undefined if it failed (the error is already shown). */
  onSettings: (patch: Partial<AppStatus>) => Promise<boolean | undefined>
  onToggleTracking: () => void
  onCalibrate: () => void
  onFitLimiters: () => void
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
  onLanguage: (lang: Lang) => void
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
  batch: 0,
  interpolate: true,
  inbetweens: -1,
}

const STREAM_DEFAULTS = { ...TUNE_DEFAULTS, ...PERF_DEFAULTS }

/** Batch choices: 0 = Auto, then poses per model call (engine max is 4). */
const BATCH_CHOICES = [0, 1, 2, 3, 4]
/** In-between choices: -1 = Auto (fill the display at this PC's key rate). */
const INBETWEEN_CHOICES = [-1, 0, 1, 2, 3]

const MODEL_SLIDERS: {
  key: 'pose_cfg' | 'id_cfg' | 'frame_blend'
  label: MessageKey
  min: number
  max: number
  title: MessageKey
}[] = [
  { key: 'pose_cfg', label: 'tune.poseFollow', min: 0, max: 6, title: 'tune.poseFollowTitle' },
  { key: 'id_cfg', label: 'tune.refLock', min: 0, max: 6, title: 'tune.refLockTitle' },
  { key: 'frame_blend', label: 'tune.snap', min: 0.05, max: 1, title: 'tune.snapTitle' },
]

const OVERLAY_PARTS = [
  ['show_outline', 'overlay.outline'],
  ['show_brows', 'overlay.brows'],
  ['show_eyes', 'overlay.eyes'],
  ['show_nose', 'overlay.nose'],
  ['show_mouth', 'overlay.mouth'],
  ['show_iris_overlay', 'overlay.iris'],
  ['show_skeleton', 'overlay.skeleton'],
  ['show_hair', 'overlay.hair'],
] as const satisfies readonly (readonly [string, MessageKey])[]

function ifmListenLine(t: I18n['t'], ifm?: LabIfm): string {
  if (!ifm) return ''
  if (ifm.receiving && ifm.peer) return t('track.ifmLive', { peer: ifm.peer })
  if (ifm.listening) return t('track.ifmWaiting')
  return ''
}

export function ControlRail(props: Props) {
  const { t, tr, lang } = useI18n()
  const s = props.status
  const busy = Boolean(s?.busy)
  const streaming = Boolean(s?.streaming)
  const paused = Boolean(s?.paused)
  // A model picked but not loaded: Start stream loads it first.
  const pendingModel = s?.pending_checkpoint || ''
  const shownModel = pendingModel || s?.checkpoint
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
  const labError = tr(labBannerError(props.lab))
  const restCalib = props.lab?.calib
  const restProgress = Number(restCalib?.progress ?? 0)
  const capturingRest = Boolean(restCalib?.capturing) && restProgress < 1
  const restLabel = capturingRest
    ? t('track.calibrating', { pct: Math.round(restProgress * 100) })
    : t('track.calibrate')
  const canCalibrate = tracking || labLive
  const trackSource = labSourceOf(props.lab)
  const trackCameras: CameraInfo[] =
    labOnline && props.lab?.cameras?.length ? props.lab.cameras : props.cameras
  const trackCameraIndex = labOnline
    ? (props.lab?.camera_index ?? s?.camera_index ?? 0)
    : (s?.camera_index ?? 0)
  const deviceLocked = tracking || (labLive && trackSource === 'camera')
  // Each batch size is its own compiled graph; it only changes between streams.
  const batchLocked = streaming || Boolean(s?.fast_warming)
  const batchSetting = s?.batch ?? STREAM_DEFAULTS.batch
  const inbetweenSetting = s?.inbetweens ?? STREAM_DEFAULTS.inbetweens
  // This PC's own speeds per batch size (measured, or ≈ predicted).
  const batchRates = BATCH_CHOICES.filter((n) => n > 0 && s?.batch_rates?.[String(n)]).map((n) => {
    const rate = s!.batch_rates![String(n)]
    return t(rate.measured ? 'perf.batchRate' : 'perf.batchRateGuess', {
      n,
      fps: rate.fps.toFixed(1),
    })
  })
  const batchTitle = batchRates.length
    ? `${t('perf.batchTitle')}\n${t('perf.batchRates', { list: batchRates.join(' · ') })}`
    : t('perf.batchTitle')
  // Which decoder is really running (read-only; shown on the Compile light).
  const decoderNote =
    s?.speed_mode_active === 'ultra'
      ? t('perf.decoderUltra')
      : s?.speed_mode_active === 'normal'
        ? t('perf.decoderNormal')
        : s?.speed_mode_active === 'eager'
          ? t('perf.decoderEager')
          : ''
  const ifm = props.lab?.ifm
  const destIp = ifm?.primary || ''
  const ifmLine = ifmListenLine(t, ifm)

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
        label={tr(
          s.progress_label ||
            (s.progress_kind === 'download' ? 'Downloading model' : 'Loading model'),
        )}
        value={progress}
      />
    ) : null

  // Only a bar from this job may drive the Create window; boot or an earlier
  // job can leave another kind behind. A model load inside Create is its own phase.
  const progressKind = s?.progress_kind || ''
  const characterBar = progressKind === 'character' || progressKind === 'reference'
  const modelBar = progressKind === 'model' || progressKind === 'download' || progressKind === 'warmup'
  const createOwnsBar = busy && (characterBar || modelBar)

  const listed = props.characters.find((c) => c.id === s?.character_id)
  const charName = listed?.name || (s?.character_id ? s.character_name : '') || ''

  return (
    <aside className="rail">
      <div className="rail-board">
        <div className="rail-panes" role="tablist" aria-label={t('rail.label')}>
          <button
            type="button"
            role="tab"
            aria-selected={railPane === 'desk'}
            className={railPane === 'desk' ? 'on' : ''}
            onClick={() => setRailPane('desk')}
          >
            {t('rail.desk')}
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={railPane === 'settings'}
            className={railPane === 'settings' ? 'on' : ''}
            onClick={() => setRailPane('settings')}
          >
            {t('rail.settings')}
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
                  {t('model.browse')}
                </button>
              </div>
            ) : null}
            <label className="field">
              <span>{t('model.label')}</span>
              <select
                value={
                  props.checkpoints.find(
                    (c) => c.label === shownModel || c.path === shownModel,
                  )?.path ||
                  props.checkpoints[0]?.path ||
                  ''
                }
                onChange={(e) => props.onCheckpoint(e.target.value)}
                onFocus={() => props.onRefreshCheckpoints?.()}
                disabled={busy || streaming || !props.checkpoints.length}
              >
                {props.checkpoints.length === 0 && <option value="">{t('model.none')}</option>}
                {props.checkpoints.map((c) => (
                  <option key={c.path} value={c.path}>
                    {c.source === 'local' ? t('model.local', { name: c.label }) : c.label}
                  </option>
                ))}
              </select>
            </label>
            {pendingModel && !streaming ? (
              <p className="hint">{t('model.notLoaded')}</p>
            ) : null}
            {(props.catalogOffers ?? []).length ? (
              <ul className="model-offers">
                {(props.catalogOffers ?? []).map((offer) => (
                  <li key={offer.name}>
                    <button
                      type="button"
                      className={`model-offer${offer.is_new ? ' is-new' : ''}`}
                      disabled={busy || streaming}
                      onClick={() => props.onDownloadModel?.(offer.name)}
                      title={offer.is_new ? t('model.offerNew') : t('model.offerAvailable')}
                    >
                      <span className="model-offer-badge">{tr(offer.badge)}</span>
                      <span className="model-offer-name">{offer.label}</span>
                    </button>
                  </li>
                ))}
              </ul>
            ) : null}
            {DEVELOPER ? (
              <>
                <label className="field">
                  <span>{t('ref.imagePath')}</span>
                  <input
                    value={props.refPath}
                    onChange={(e) => props.onRefPath(e.target.value)}
                    placeholder={t('ref.placeholder')}
                    disabled={busy || streaming}
                  />
                </label>
                <div className="row">
                  <label
                    className={`btn ghost file-btn${busy || streaming ? ' disabled' : ''}`}
                    aria-disabled={busy || streaming}
                  >
                    {t('ref.browse')}
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
                    {t('ref.apply')}
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
                  label={t('track.mirror')}
                  checked={Boolean(s?.mirror)}
                  onChange={(v) => props.onSettings({ mirror: v })}
                  title={t('track.mirrorTitle')}
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
                      {tracking ? t('track.stop') : t('track.start')}
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
                      title={t('track.calibrateTitle')}
                    >
                      {restLabel}
                    </button>
                  </div>
                  {restCalib?.hint ? <p className="hint">{tr(restCalib.hint)}</p> : null}
                  {DEVELOPER && s?.body_label ? <p className="hint">{s.body_label}</p> : null}
                </>
              }
            >
              <div className="track-cam-row">
                <div className="lab-tabs" role="tablist" aria-label={t('track.input')}>
                  <button
                    type="button"
                    className={trackSource === 'camera' ? 'on' : ''}
                    aria-pressed={trackSource === 'camera'}
                    onClick={() => props.onLabSource('camera')}
                  >
                    {t('track.camera')}
                  </button>
                  <button
                    type="button"
                    className={trackSource === 'ifm' ? 'on brand' : 'brand'}
                    aria-pressed={trackSource === 'ifm'}
                    title={t('track.ifmTitle')}
                    onClick={() => props.onLabSource('ifm')}
                  >
                    {t('track.ifm')}
                  </button>
                </div>
                {trackSource === 'ifm' || labOnline ? (
                  labError ? (
                    <p className="status-error">{labError}</p>
                  ) : !labOnline && trackSource === 'ifm' ? (
                    <p className="hint">{t('track.startLab')}</p>
                  ) : null
                ) : null}
                {trackSource === 'ifm' ? (
                  <div className="ifm-listen">
                    {destIp ? (
                      <button
                        type="button"
                        className="ifm-dest"
                        title={t('track.copyTitle')}
                        onClick={copyDest}
                      >
                        <span>{copiedDest ? t('track.copied') : t('track.thisPc')}</span>
                        <em className="mono">{destIp}</em>
                      </button>
                    ) : (
                      <p className="hint">{t('track.sameWifi')}</p>
                    )}
                    <label className="field inline">
                      <span>{t('track.port')}</span>
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
                      aria-label={t('track.camera')}
                      value={trackCameraIndex}
                      onChange={(e) => props.onCamera(Number(e.target.value))}
                      onFocus={() => {
                        if (!labOnline || trackCameras.length === 0) {
                          props.onRefreshCameras?.()
                        }
                      }}
                      disabled={deviceLocked}
                    >
                      {trackCameras.length === 0 && <option value={0}>{t('track.cameraN', { n: 0 })}</option>}
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

          <section className="char-stage" aria-label={t('toon.title')}>
            <div className="char-stage-bar">
              <h2 className="group-title">{t('toon.title')}</h2>
              <span className="char-stage-name">{charName}</span>
            </div>
            {showProgress &&
            !props.charactersCreating &&
            (s?.progress_kind === 'character' || s?.progress_kind === 'reference') ? (
              <ProgressMeter
                label={tr(s.progress_label || 'Loading character')}
                value={progress}
              />
            ) : null}
            <CharacterLibrary
              currentId={s?.character_id || ''}
              characters={props.characters}
              creating={props.charactersCreating}
              createProgress={
                props.charactersCreating
                  ? Math.max(createOwnsBar ? progress : 0, 0.04)
                  : characterBar
                    ? progress
                    : 0
              }
              createLabel={
                (props.charactersCreating ? createOwnsBar : characterBar)
                  ? s?.progress_label || 'Creating character…'
                  : 'Creating character…'
              }
              createPhase={modelBar ? 'model' : 'character'}
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
              <h2 className="group-title">{t('stream.title')}</h2>
              <Lamp
                on={streaming || virtualCam}
                pending={paused}
                label={t(
                  paused
                    ? 'stream.paused'
                    : streaming
                      ? 'stream.streaming'
                      : virtualCam
                        ? 'stream.vcamOn'
                        : 'stream.idle',
                )}
              />
            </div>
            {showProgress &&
            (s?.progress_kind === 'warmup' || s?.progress_kind === 'compile') ? (
              <ProgressMeter
                label={tr(s.progress_label || 'Preparing stream')}
                value={progress}
              />
            ) : null}
            <div className="row stream-run">
              <button
                type="button"
                className={
                  streaming ? 'btn is-on' : pendingModel ? 'btn is-pending' : 'btn primary'
                }
                onClick={props.onToggleStream}
                disabled={(busy && !streaming) || Boolean(s?.fast_warming)}
                title={
                  !streaming && pendingModel
                    ? t('stream.loadThenStart', { model: pendingModel })
                    : undefined
                }
              >
                {streaming ? t('stream.stop') : pendingModel ? t('stream.loadNew') : t('stream.start')}
              </button>
              <button
                type="button"
                className={paused ? 'btn primary' : 'btn'}
                onClick={props.onTogglePause}
                disabled={!streaming || Boolean(s?.fast_warming)}
                title={paused ? t('stream.resumeTitle') : t('stream.pauseTitle')}
              >
                {paused ? t('stream.resume') : t('stream.pause')}
              </button>
              <button
                type="button"
                className={virtualCam ? 'btn is-on' : 'btn'}
                onClick={props.onToggleVirtualCam}
                disabled={busy && !virtualCam}
                title={t('stream.camTitle')}
              >
                {virtualCam ? t('stream.stopCam') : t('stream.startCam')}
              </button>
            </div>
            <div className="row">
              <button
                type="button"
                className="btn ghost"
                onClick={() => props.onGenerate()}
                disabled={busy || streaming || Boolean(s?.fast_warming)}
              >
                {t('stream.generateOnce')}
              </button>
            </div>
            {virtualCam && s?.virtual_cam_width && s?.virtual_cam_height ? (
              // The camera driver offers 1920x1080 by default and pads our square
              // frame with black; only the receiving app can ask for our size.
              <p className="hint">
                {t('stream.camSizeHint', {
                  size: `${s.virtual_cam_width}x${s.virtual_cam_height}`,
                })}
              </p>
            ) : null}
            {s?.virtual_cam_error ? (
              <p className="status-error">{tr(s.virtual_cam_error)}</p>
            ) : null}
          </section>
          </div>
        ) : (
          <div className="rail-controls">
            <section className="group">
              <h2 className="group-title">{t('lang.title')}</h2>
              <div className="lab-tabs" role="radiogroup" aria-label={t('lang.title')}>
                {LANGUAGES.map((row) => (
                  <button
                    key={row.id}
                    type="button"
                    role="radio"
                    lang={row.id}
                    aria-checked={lang === row.id}
                    className={lang === row.id ? 'on brand' : 'brand'}
                    onClick={() => {
                      if (lang !== row.id) props.onLanguage(row.id)
                    }}
                  >
                    {row.label}
                  </button>
                ))}
              </div>
            </section>
            <section className="group stream-tune">
              <div className="group-head">
                <h2 className="group-title">{t('tune.title')}</h2>
                <button
                  type="button"
                  className="btn ghost compact"
                  onClick={resetStreamDefaults}
                  title={t('tune.defaultsTitle')}
                >
                  {t('common.defaults')}
                </button>
              </div>
              <ul className="lab-sliders stream-sliders">
                <li className="stream-steps">
                  <span title={t('tune.stepsTitle')}>{t('tune.steps')}</span>
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
                    aria-label={t('tune.steps')}
                  />
                </li>
                {MODEL_SLIDERS.map((row) => {
                  const title = t(row.title, { value: STREAM_DEFAULTS.frame_blend.toFixed(2) })
                  return (
                  <li key={row.key}>
                    <span title={title}>{t(row.label)}</span>
                    <input
                      type="range"
                      min={row.min}
                      max={row.max}
                      step={0.05}
                      value={cfgDraft[row.key]}
                      title={title}
                      onChange={(e) => commitCfg(row.key, Number(e.target.value))}
                    />
                    <em className="mono">{cfgDraft[row.key].toFixed(2)}</em>
                  </li>
                  )
                })}
              </ul>
              <div className="stream-compile">
                <Toggle
                  label={t('tune.holdLast')}
                  checked={s?.hold_last !== false}
                  onChange={(v) => props.onSettings({ hold_last: v })}
                  title={t('tune.holdLastTitle')}
                />
              </div>
            </section>
            <section className="group stream-perf">
              <div className="group-head">
                <h2 className="group-title">{t('perf.title')}</h2>
                <button
                  type="button"
                  className="btn ghost compact"
                  onClick={resetPerfDefaults}
                  title={t('perf.defaultsTitle')}
                >
                  {t('common.defaults')}
                </button>
              </div>
              <ul className="lab-sliders stream-sliders">
                <li>
                  <span title={t('perf.maxFpsTitle')}>{t('perf.maxFps')}</span>
                  <input
                    type="range"
                    min={0}
                    max={100}
                    step={1}
                    value={s?.max_fps ?? STREAM_DEFAULTS.max_fps}
                    title={t('perf.maxFpsTitle')}
                    onChange={(e) => props.onSettings({ max_fps: Number(e.target.value) })}
                  />
                  <em className="mono">
                    {(s?.max_fps ?? STREAM_DEFAULTS.max_fps) > 0
                      ? s?.max_fps ?? STREAM_DEFAULTS.max_fps
                      : t('perf.auto', { n: Math.round((s?.gen_cap ?? 10) * 10) / 10 })}
                  </em>
                </li>
                <li className="stream-batch">
                  <span title={t('perf.inbetweensTitle')}>{t('perf.inbetweens')}</span>
                  <div
                    className="lab-tabs"
                    role="radiogroup"
                    aria-label={t('perf.inbetweens')}
                    title={t('perf.inbetweensTitle')}
                  >
                    {INBETWEEN_CHOICES.map((n) => (
                      <button
                        key={n}
                        type="button"
                        role="radio"
                        aria-checked={inbetweenSetting === n}
                        className={inbetweenSetting === n ? 'on' : ''}
                        disabled={!interpolate}
                        onClick={() => {
                          if (inbetweenSetting !== n) props.onSettings({ inbetweens: n })
                        }}
                      >
                        {n < 0 ? t('perf.inbetweensAuto') : n}
                      </button>
                    ))}
                  </div>
                  <em className="mono" title={t('perf.inbetweensTitle')}>
                    {!interpolate
                      ? 0
                      : streaming && s?.inbetweens_live != null
                        ? s.inbetweens_live
                        : inbetweenSetting < 0
                          ? '—'
                          : inbetweenSetting}
                  </em>
                </li>
                <li className="stream-batch">
                  <span title={batchTitle}>{t('perf.batch')}</span>
                  <div
                    className="lab-tabs"
                    role="radiogroup"
                    aria-label={t('perf.batch')}
                    title={batchLocked ? t('perf.batchLocked') : batchTitle}
                  >
                    {BATCH_CHOICES.map((n) => (
                      <button
                        key={n}
                        type="button"
                        role="radio"
                        aria-checked={batchSetting === n}
                        className={batchSetting === n ? 'on' : ''}
                        disabled={batchLocked}
                        onClick={() => {
                          if (batchSetting !== n) props.onSettings({ batch: n })
                        }}
                      >
                        {n === 0 ? t('perf.batchAuto') : n}
                      </button>
                    ))}
                  </div>
                  <em className="mono" title={batchTitle}>
                    ×{s?.batch_size ?? 1}
                  </em>
                </li>
              </ul>
              <div className="stream-compile">
                <Toggle
                  label={t('perf.interpolate')}
                  checked={interpolate}
                  onChange={(v) => props.onSettings({ interpolate: v })}
                  title={t('perf.interpolateTitle')}
                />
                <Toggle
                  label={t('perf.compile')}
                  checked={Boolean(s?.compile_model)}
                  onChange={(v) => props.onSettings({ compile_model: v })}
                  title={t('perf.compileTitle')}
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
                    (s?.compile_detail
                      ? tr(s.compile_detail)
                      : s?.fast_warming && !s?.compile_on
                        ? t('perf.boostBuilding')
                        : tr(
                            s?.compile_on
                              ? 'Speed boost on'
                              : s?.compile_status === 'fail'
                                ? 'Speed boost unavailable — running at normal speed'
                                : s?.compile_status === 'skip'
                                  ? 'Speed boost needs an NVIDIA GPU'
                                  : s?.compile_model
                                    ? 'Speed boost builds when the stream starts'
                                    : 'Speed boost off',
                          )) + (decoderNote ? ` · ${decoderNote}` : '')
                  }
                />
              </div>
            </section>
            <GpuPicker busy={busy} onRestart={props.onReloadBackend} />
            <section className="group">
              <TravelBox
                value={s?.travel_box}
                disabled={busy}
                show={s?.show_limiters === true}
                onShow={(show_limiters) => props.onSettings({ show_limiters })}
                onChange={(travel_box) => props.onSettings({ travel_box })}
                onFit={props.onFitLimiters}
              />
            </section>
            <section className="group">
              <div className="group-head">
                <h2 className="group-title">{t('overlay.title')}</h2>
                <Toggle
                  className="overlay-master"
                  label={t('common.show')}
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
                    {OVERLAY_PARTS.map(([key, label]) => (
                      <Toggle
                        key={key}
                        label={t(label)}
                        checked={overlayParts[key]}
                        onChange={(v) => props.onSettings({ [key]: v })}
                      />
                    ))}
                  </div>
                </div>
              ) : null}
            </section>
            <section className="group">
              <h2 className="group-title">{t('app.title')}</h2>
              <div className="row">
                <button
                  type="button"
                  className="btn"
                  onClick={props.onReloadBackend}
                  title={t('app.reloadTitle')}
                >
                  {t('app.reload')}
                </button>
              </div>
            </section>
            {DEVELOPER ? (
              <section className="group">
                <div className="group-head">
                  <h2 className="group-title">{t('dev.title')}</h2>
                </div>
                <label className="field inline">
                  <span>{t('dev.trackFps')}</span>
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
                    label={t('dev.fast')}
                    checked={Boolean(s?.fast_mode)}
                    onChange={(v) => props.onSettings({ fast_mode: v })}
                    disabled={streaming || Boolean(s?.fast_warming)}
                    light={!s?.fast_mode ? 'off' : s?.fast_warming ? 'pending' : 'on'}
                    lightTitle={s?.fast_mode ? t('dev.fastOn') : t('dev.fastOff')}
                  />
                  <Toggle
                    label={t('dev.autoSync')}
                    checked={Boolean(s?.auto_sync_track)}
                    onChange={(v) => props.onSettings({ auto_sync_track: v })}
                  />
                  <Toggle
                    label={t('dev.iris')}
                    checked={Boolean(s?.use_iris)}
                    onChange={(v) => props.onSettings({ use_iris: v })}
                  />
                  <Toggle
                    label={t('dev.body')}
                    checked={Boolean(s?.use_body)}
                    onChange={(v) => props.onSettings({ use_body: v })}
                  />
                  <Toggle
                    label={t('dev.drivePose')}
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
          <p className="status-error">{props.error || tr(s?.error)}</p>
        </footer>
      ) : null}
    </aside>
  )
}
