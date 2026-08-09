import { useEffect, useState } from 'react'
import type { AppStatus, CameraInfo, Checkpoint } from '../api'
import { DEVELOPER } from '../developer'

type Props = {
  status: AppStatus | null
  checkpoints: Checkpoint[]
  cameras: CameraInfo[]
  refPath: string
  error: string
  onRefPath: (v: string) => void
  onCheckpoint: (path: string) => void
  onBrowseCheckpoint: () => void
  onUploadRef: (file: File) => void
  onApplyRef: () => void
  onSettings: (patch: Partial<AppStatus>) => void
  onToggleTracking: () => void
  onCalibrate: () => void
  onGenerate: () => void
  onToggleStream: () => void
  onToggleVirtualCam: () => void
  onRefreshCameras: () => void
}

function clampSteps(raw: string): number | null {
  const n = Number(raw)
  if (!Number.isFinite(n)) return null
  return Math.max(1, Math.min(50, Math.round(n)))
}

export function ControlRail(props: Props) {
  const s = props.status
  const busy = Boolean(s?.busy)
  const streaming = Boolean(s?.streaming)
  const tracking = Boolean(s?.tracking)
  const virtualCam = Boolean(s?.virtual_cam)
  const progress = Number(s?.progress ?? 0)
  const showProgress =
    busy && Boolean(s?.progress_kind) && (progress > 0 || Boolean(s?.progress_label))

  // Draft while typing — committing every keystroke was snapping the field back.
  const [stepsDraft, setStepsDraft] = useState(String(s?.steps ?? 2))
  useEffect(() => {
    setStepsDraft(String(s?.steps ?? 2))
  }, [s?.steps])

  function commitSteps() {
    const next = clampSteps(stepsDraft)
    if (next == null) {
      setStepsDraft(String(s?.steps ?? 2))
      return
    }
    setStepsDraft(String(next))
    if (next !== (s?.steps ?? 2)) {
      props.onSettings({ steps: next })
    }
  }

  return (
    <aside className="rail">
      <header className="rail-brand">
        <p className="rail-eyebrow">Operator desk</p>
        <h1 className="rail-title">VTM Noble</h1>
      </header>

      <section className="group">
        <h2 className="group-title">Model</h2>
        <div className="row">
          <button
            type="button"
            className="btn primary"
            onClick={props.onBrowseCheckpoint}
            disabled={busy || streaming}
          >
            Browse…
          </button>
        </div>
        {s?.checkpoint ? <p className="hint mono">{s.checkpoint}</p> : null}
        {DEVELOPER ? (
          <label className="field">
            <span>Checkpoint</span>
            <select
              value={
                props.checkpoints.find((c) => c.label === s?.checkpoint)?.path ||
                props.checkpoints[0]?.path ||
                ''
              }
              onChange={(e) => props.onCheckpoint(e.target.value)}
              disabled={busy || streaming || !props.checkpoints.length}
            >
              {props.checkpoints.length === 0 && <option value="">No checkpoints</option>}
              {props.checkpoints.map((c) => (
                <option key={c.path} value={c.path}>
                  {c.label}
                </option>
              ))}
            </select>
          </label>
        ) : null}
        {showProgress &&
        (s?.progress_kind === 'model' || s?.progress_kind === 'download') ? (
          <ProgressMeter
            label={
              s.progress_label ||
              (s.progress_kind === 'download' ? 'Downloading model' : 'Loading model')
            }
            value={progress}
          />
        ) : null}
      </section>

      <section className="group">
        <h2 className="group-title">Reference</h2>
        {DEVELOPER ? (
          <label className="field">
            <span>Image path</span>
            <input
              value={props.refPath}
              onChange={(e) => props.onRefPath(e.target.value)}
              placeholder="Path to reference still"
              disabled={busy || streaming}
            />
          </label>
        ) : null}
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
          {DEVELOPER ? (
            <button
              type="button"
              className="btn"
              onClick={props.onApplyRef}
              disabled={busy || streaming || !props.refPath.trim()}
            >
              Apply ref
            </button>
          ) : null}
        </div>
        {!DEVELOPER && s?.reference_name ? (
          <p className="hint mono">{s.reference_name}</p>
        ) : null}
        {showProgress && s?.progress_kind === 'reference' ? (
          <ProgressMeter
            label={s.progress_label || 'Encoding reference'}
            value={progress}
          />
        ) : null}
      </section>

      <section className="group">
        <div className="group-head">
          <h2 className="group-title">Tracking</h2>
          <span
            className={`lamp ${tracking ? 'is-on' : 'is-off'}`}
            title={tracking ? 'Tracking on' : 'Tracking off'}
            aria-label={tracking ? 'Tracking on' : 'Tracking off'}
          />
        </div>

        <div className="track-cam-row">
          <Toggle
            label="Mirror"
            checked={Boolean(s?.mirror)}
            onChange={(v) => props.onSettings({ mirror: v })}
          />
          <label className="field track-cam-field">
            <span>Camera</span>
            <div className="track-cam-controls">
              <select
                className="camera-select"
                value={s?.camera_index ?? 0}
                onChange={(e) =>
                  props.onSettings({ camera_index: Number(e.target.value) })
                }
                disabled={tracking || busy}
              >
                {props.cameras.length === 0 && <option value={0}>Camera 0</option>}
                {props.cameras.map((c) => (
                  <option key={c.index} value={c.index}>
                    {c.name}
                  </option>
                ))}
              </select>
              <button
                type="button"
                className="btn ghost btn-compact"
                onClick={props.onRefreshCameras}
                disabled={tracking}
                title="Refresh cameras"
              >
                Refresh
              </button>
            </div>
          </label>
        </div>

        {DEVELOPER ? (
          <div className="toggles">
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
        ) : null}

        <div className="row">
          <button
            type="button"
            className={tracking ? 'btn danger' : 'btn primary'}
            onClick={props.onToggleTracking}
            disabled={busy || streaming}
          >
            {tracking ? 'Stop tracking' : 'Start tracking'}
          </button>
          <button
            type="button"
            className="btn"
            onClick={props.onCalibrate}
            disabled={!tracking || busy}
          >
            Calibrate
          </button>
        </div>
        {DEVELOPER && s?.body_label ? <p className="hint">{s.body_label}</p> : null}
      </section>

      <section className="group">
        <div className="group-head">
          <h2 className="group-title">Stream</h2>
          <span
            className={`lamp ${streaming || virtualCam ? 'is-on' : 'is-off'}`}
            title={
              streaming
                ? 'Streaming'
                : virtualCam
                  ? 'Virtual camera on'
                  : 'Stream idle'
            }
            aria-label={
              streaming
                ? 'Streaming'
                : virtualCam
                  ? 'Virtual camera on'
                  : 'Stream idle'
            }
          />
        </div>
        <label className="field inline">
          <span>Steps</span>
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
          />
        </label>
        {DEVELOPER ? (
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
        ) : null}
        <div className="toggles">
          <Toggle
            label="Fast"
            checked={Boolean(s?.fast_mode)}
            onChange={(v) => props.onSettings({ fast_mode: v })}
            disabled={streaming || Boolean(s?.fast_warming)}
            light={
              !s?.fast_mode
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
                ? 'torch.compile verified'
                : s?.fast_warming
                  ? 'Compiling / testing…'
                  : s?.compile_status === 'fail'
                    ? 'torch.compile unavailable — eager Fast'
                    : s?.compile_status === 'skip'
                      ? 'torch.compile skipped (CPU)'
                      : s?.fast_mode
                        ? 'torch.compile pending — apply a reference to test'
                        : 'Fast off')
            }
          />
          <Toggle
            label="Batch ×2"
            checked={Boolean(s?.batch2)}
            onChange={(v) => props.onSettings({ batch2: v })}
            disabled={streaming || Boolean(s?.fast_warming)}
          />
          <Toggle
            label="Show skeleton"
            checked={Boolean(s?.show_mesh)}
            onChange={(v) => props.onSettings({ show_mesh: v })}
          />
          {DEVELOPER ? (
            <Toggle
              label="Auto sync track"
              checked={Boolean(s?.auto_sync_track)}
              onChange={(v) => props.onSettings({ auto_sync_track: v })}
            />
          ) : null}
        </div>
        {showProgress &&
        (s?.progress_kind === 'warmup' || s?.progress_kind === 'compile') ? (
          <ProgressMeter
            label={s.progress_label || 'Compiling… please wait'}
            value={progress}
          />
        ) : null}
        <div className="row">
          <button
            type="button"
            className={streaming ? 'btn danger' : 'btn primary'}
            onClick={props.onToggleStream}
            disabled={(busy && !streaming) || Boolean(s?.fast_warming)}
          >
            {streaming ? 'Stop stream' : 'Start stream'}
          </button>
          <button
            type="button"
            className={virtualCam ? 'btn danger' : 'btn'}
            onClick={props.onToggleVirtualCam}
            disabled={busy && !virtualCam}
            title="Send avatar frames to a virtual camera for OBS"
          >
            {virtualCam ? 'Stop virtual cam' : 'Virtual camera'}
          </button>
        </div>
        <div className="row">
          <button
            type="button"
            className="btn ghost"
            onClick={props.onGenerate}
            disabled={busy || streaming || Boolean(s?.fast_warming)}
          >
            Generate once
          </button>
        </div>
        {virtualCam ? (
          <p className="hint">
            OBS → Video Capture Device →{' '}
            {s?.virtual_cam_device || 'VTM Noble Cam'}
          </p>
        ) : null}
        {s?.virtual_cam_error ? (
          <p className="status-error">{s.virtual_cam_error}</p>
        ) : null}
        {s?.fast_warming ? (
          <p className="hint">Please wait — torch.compile is still running.</p>
        ) : null}
      </section>

      <footer className="rail-status">
        <p className="status-line">{s?.message || 'Starting…'}</p>
        {props.error || s?.error ? (
          <p className="status-error">{props.error || s?.error}</p>
        ) : null}
        {s?.device ? <p className="hint mono">{s.device}</p> : null}
      </footer>
    </aside>
  )
}

function ProgressMeter({ label, value }: { label: string; value: number }) {
  const pct = Math.max(0, Math.min(100, Math.round(value * 100)))
  return (
    <div className="progress" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
      <div className="progress-meta">
        <span className="progress-label">{label}</span>
        <span className="progress-pct mono">{pct}%</span>
      </div>
      <div className="progress-track">
        <div className="progress-fill" style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}

function Toggle({
  label,
  checked,
  onChange,
  disabled,
  light,
  lightTitle,
}: {
  label: string
  checked: boolean
  onChange: (v: boolean) => void
  disabled?: boolean
  light?: 'on' | 'off' | 'pending' | 'fail'
  lightTitle?: string
}) {
  return (
    <label className={`toggle ${disabled ? 'is-disabled' : ''}`}>
      <input
        type="checkbox"
        checked={checked}
        disabled={disabled}
        onChange={(e) => onChange(e.target.checked)}
      />
      <span>{label}</span>
      {light ? (
        <span
          className={`signal-light is-${light}`}
          title={lightTitle || undefined}
          aria-label={lightTitle || `status ${light}`}
        />
      ) : null}
    </label>
  )
}
