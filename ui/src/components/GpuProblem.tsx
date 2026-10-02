import { useState } from 'react'
import { api, type GpuProblem as Problem, type GpuTried } from '../api'
import { useI18n, type MessageKey } from '../i18n'
import { nativeWindow } from '../nativeWindow'

const BUILD_LABELS: Record<string, string> = { cu128: 'CUDA 12.8', cu126: 'CUDA 12.6' }

type Props = { problem: Problem }

/** Splash panel when the graphics card cannot run the AI: what happened, what was tried, what is left. */
export function GpuProblem({ problem }: Props) {
  const { t } = useI18n()
  const [repair, setRepair] = useState<'idle' | 'busy' | 'started'>('idle')
  const [repairError, setRepairError] = useState('')
  const vars = {
    gpu: problem.gpu || 'GPU',
    cap: problem.cap || '?',
    driver: problem.driver || '?',
    build: problem.build || '?',
  }

  const triedLine = (row: GpuTried) => {
    if (row.what === 'start_check') return t('gpu.tried.start')
    const build = BUILD_LABELS[row.build ?? ''] ?? row.build ?? '?'
    return t(row.ok ? 'gpu.tried.buildOk' : 'gpu.tried.buildFail', { build })
  }
  const tried = [
    ...(problem.kind === 'no_nvidia' ? [t('gpu.tried.search')] : []),
    ...problem.tried.map(triedLine),
  ]
  const next: MessageKey =
    problem.kind === 'no_nvidia' || problem.kind === 'card_too_old' || problem.kind === 'driver_old'
      ? `gpu.next.${problem.kind}`
      : problem.repair
        ? 'gpu.next.repair'
        : 'gpu.next.other'

  const startRepair = async () => {
    setRepair('busy')
    setRepairError('')
    try {
      await api.gpuRepair()
      setRepair('started')
    } catch (err) {
      setRepair('idle')
      setRepairError(t('gpu.repairFailed', { error: err instanceof Error ? err.message : String(err) }))
    }
  }

  return (
    <div className="gpu-problem" role="alert" onMouseDown={(e) => e.stopPropagation()}>
      <h2>{t(`gpu.title.${problem.kind}`)}</h2>
      <section>
        <h3>{t('gpu.whatHeading')}</h3>
        <p>{t(`gpu.what.${problem.kind}`, vars)}</p>
        {problem.error ? <p className="gpu-problem-error">{t('gpu.details', { error: problem.error })}</p> : null}
      </section>
      {tried.length ? (
        <section>
          <h3>{t('gpu.triedHeading')}</h3>
          <ul>
            {tried.map((line, i) => (
              <li key={i}>{line}</li>
            ))}
          </ul>
        </section>
      ) : null}
      <section>
        <h3>{t('gpu.nextHeading')}</h3>
        <p>{repair === 'started' ? t('gpu.repairStarted') : t(next)}</p>
        {repairError ? <p className="status-error">{repairError}</p> : null}
      </section>
      <div className="gpu-problem-actions">
        {problem.repair && repair !== 'started' ? (
          <button type="button" className="btn primary" disabled={repair === 'busy'} onClick={() => void startRepair()}>
            {repair === 'busy' ? t('gpu.repairing') : t('gpu.repair')}
          </button>
        ) : null}
        <button type="button" className="btn" onClick={() => nativeWindow('close')}>
          {t('win.close')}
        </button>
      </div>
    </div>
  )
}
