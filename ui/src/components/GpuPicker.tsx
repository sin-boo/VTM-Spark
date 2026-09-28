import { useEffect, useState } from 'react'
import { api, type GpuState } from '../api'
import { useI18n } from '../i18n'

type Props = {
  /** Model load / warmup in flight: hold off on restarting under it. */
  busy: boolean
  onRestart: () => void
}

/** Which NVIDIA card runs the model. Saved now, used after a restart. */
export function GpuPicker({ busy, onRestart }: Props) {
  const { t } = useI18n()
  const [state, setState] = useState<GpuState | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    let alive = true
    api.gpus().then(
      (next) => alive && setState(next),
      // Older backend without /api/gpus: just leave the section empty.
      () => alive && setState(null),
    )
    return () => {
      alive = false
    }
  }, [])

  function pick(uuid: string) {
    setError('')
    api.setGpu(uuid).then(setState, (e: unknown) =>
      setError(e instanceof Error ? e.message : String(e)),
    )
  }

  if (!state) return null
  const gpus = state.gpus

  return (
    <section className="group">
      <h2 className="group-title" title={t('gpu.pickTitle')}>
        {t('gpu.title')}
      </h2>
      {gpus.length === 0 ? (
        <p className="hint">{t('gpu.none')}</p>
      ) : (
        <label className="field">
          <select
            className="camera-select"
            aria-label={t('gpu.title')}
            title={t('gpu.pickTitle')}
            value={state.selected}
            disabled={Boolean(state.external)}
            onChange={(e) => pick(e.target.value)}
          >
            <option value="">{t('gpu.auto')}</option>
            {gpus.map((gpu) => (
              <option key={gpu.uuid} value={gpu.uuid}>
                {t('gpu.option', {
                  n: gpu.index,
                  name: gpu.name,
                  gb: Math.round(gpu.memory_mb / 1024),
                })}
              </option>
            ))}
          </select>
        </label>
      )}
      {state.external ? (
        <p className="hint">{t('gpu.external', { value: state.external })}</p>
      ) : null}
      {state.in_use ? <p className="hint">{t('gpu.inUse', { name: state.in_use })}</p> : null}
      {state.restart_needed ? (
        <div className="row">
          <p className="hint">{t('gpu.restartHint')}</p>
          <button type="button" className="btn primary compact" onClick={onRestart} disabled={busy}>
            {t('gpu.restart')}
          </button>
        </div>
      ) : null}
      {error ? <p className="status-error">{error}</p> : null}
    </section>
  )
}
