import { useEffect, useRef } from 'react'
import type { BootStatus } from '../api'
import { dragIfPrimary } from '../nativeWindow'

const EMPTY: BootStatus = {
  ready: false,
  running: true,
  error: '',
  awaiting: '',
  suggested: '',
  progress: 0,
  progress_label: 'Loading resources…',
  stages: {
    model: { state: 'idle', progress: 0, label: 'Loading model' },
    character: { state: 'idle', progress: 0, label: 'Loading character' },
    lab: { state: 'idle', progress: 0, label: 'Connecting Track Lab' },
  },
}

const WEIGHTS = { model: 0.46, character: 0.34, lab: 0.2 } as const
const APP_VERSION = '0.1 alpha'

type Props = {
  boot: BootStatus | null
  error?: string
}

function overall(boot: BootStatus): { pct: number; label: string } {
  if (typeof boot.progress === 'number') {
    return {
      pct: Math.max(0, Math.min(100, boot.progress * 100)),
      label: boot.ready ? 'Ready' : boot.progress_label || 'Loading resources…',
    }
  }
  const stages = boot.stages ?? EMPTY.stages
  const pct =
    (Math.max(0, Math.min(1, stages.model?.progress ?? 0)) * WEIGHTS.model +
      Math.max(0, Math.min(1, stages.character?.progress ?? 0)) * WEIGHTS.character +
      Math.max(0, Math.min(1, stages.lab?.progress ?? 0)) * WEIGHTS.lab) *
    100
  const running = (['lab', 'character', 'model'] as const).find((key) => stages[key]?.state === 'run')
  const label = boot.ready ? 'Ready' : running ? stages[running]?.label : 'Loading resources…'
  return { pct: Math.max(0, Math.min(100, pct)), label: label || 'Loading resources…' }
}

export function Splash(props: Props) {
  const boot = props.boot ?? EMPTY
  const bar = overall(boot)
  const notice = props.error || boot.error
  const peak = useRef(0)
  const target = boot.ready ? 100 : bar.pct
  peak.current = Math.max(peak.current, target)
  const pct = peak.current

  useEffect(() => {
    const root = document.documentElement
    root.setAttribute('data-splash', '')
    return () => root.removeAttribute('data-splash')
  }, [])

  return (
    <div
      className="krita-splash"
      role="status"
      aria-live="polite"
      aria-busy={!boot.ready}
      onMouseDown={(e) => dragIfPrimary(e.button)}
    >
      <div className="krita-splash-art" aria-hidden="true" />
      <aside className="krita-splash-panel">
        <div className="krita-splash-plate">
          <header className="krita-splash-brand">
            <img className="krita-splash-mark" src="/splash-mark.png?alpha=1" width={72} height={72} alt="" />
            <p className="krita-splash-kicker">VTM</p>
            <h1>Noble</h1>
            <p className="krita-splash-ver">{APP_VERSION}</p>
          </header>
          <div className="krita-splash-status">
            <p className="krita-splash-line">{bar.label}</p>
            <div className="krita-splash-bar" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(pct)}>
              <b style={{ width: `${pct.toFixed(1)}%` }}><i /></b>
            </div>
            {notice ? <p className="status-error">{notice}</p> : null}
          </div>
        </div>
      </aside>
    </div>
  )
}
