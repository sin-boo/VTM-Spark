import { useState } from 'react'
import { api, type AppStatus, type PerfResult } from '../api'
import { useI18n } from '../i18n'

const DURATIONS = [15, 30, 60]

/** Change from the first run, e.g. "−32%"; blank for the first run itself. */
function versus(value: number, base: number): string {
  if (!(base > 0)) return ''
  const pct = Math.round(((value - base) / base) * 100)
  return pct === 0 ? '±0%' : `${pct > 0 ? '+' : '−'}${Math.abs(pct)}%`
}

function cell(value: number | null | undefined, unit = ''): string {
  return value == null || !Number.isFinite(value) ? '—' : `${value}${unit}`
}

/**
 * Developer: move the character on a fixed loop and time the stream. Run it once
 * alone and once with a game open; the second row shows what the game costs.
 */
export function PerfTest({ status }: { status: AppStatus | null }) {
  const { t } = useI18n()
  const [seconds, setSeconds] = useState(30)
  const [label, setLabel] = useState('')
  const [error, setError] = useState('')
  const state = status?.perf_test
  const running = Boolean(state?.running)
  const results: PerfResult[] = state?.results ?? []
  const base = results[0]

  const act = (call: () => Promise<unknown>) => {
    setError('')
    call().catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)))
  }

  return (
    <div className="perf-test">
      <p className="hint">{t('dev.perfHint')}</p>
      <div className="row">
        <select
          value={seconds}
          onChange={(e) => setSeconds(Number(e.target.value))}
          disabled={running}
          aria-label={t('dev.perfLength')}
        >
          {DURATIONS.map((n) => (
            <option key={n} value={n}>
              {t('dev.perfSeconds', { n })}
            </option>
          ))}
        </select>
        <input
          type="text"
          value={label}
          placeholder={t('dev.perfLabel', { n: results.length + 1 })}
          onChange={(e) => setLabel(e.target.value)}
          disabled={running}
          maxLength={40}
        />
        {running ? (
          <button type="button" className="btn" onClick={() => act(api.stopPerfTest)}>
            {t('dev.perfStop')}
          </button>
        ) : (
          <button
            type="button"
            className="btn primary"
            onClick={() =>
              act(async () => {
                await api.startPerfTest(seconds, label)
                setLabel('')
              })
            }
          >
            {t('dev.perfStart')}
          </button>
        )}
      </div>
      {running ? (
        <div className="perf-progress" title={state?.label}>
          <div style={{ width: `${Math.round((state?.progress ?? 0) * 100)}%` }} />
        </div>
      ) : null}
      {error ? <p className="status-error">{error}</p> : null}
      {results.length ? (
        <>
          <table className="perf-table mono">
            <thead>
              <tr>
                <th>{t('dev.perfRun')}</th>
                <th title={t('dev.perfFpsTitle')}>FPS</th>
                <th title={t('dev.perfLowTitle')}>1% low</th>
                <th title={t('dev.perfStallsTitle')}>{t('dev.perfStalls')}</th>
                <th title={t('dev.perfKeysTitle')}>{t('dev.perfKeys')}</th>
                <th title={t('dev.perfMsTitle')}>ms</th>
                <th>GPU</th>
                <th title={t('dev.perfVramTitle')}>VRAM</th>
                <th title={t('dev.perfCpuTitle')}>CPU</th>
              </tr>
            </thead>
            <tbody>
              {results.map((r, i) => (
                <tr key={`${r.label}-${i}`}>
                  <td title={t('dev.perfSecondsRan', { n: r.seconds })}>{r.label}</td>
                  <td>
                    {r.fps}
                    {i > 0 ? <small> {versus(r.fps, base.fps)}</small> : null}
                  </td>
                  <td>
                    {r.fps_low}
                    {i > 0 ? <small> {versus(r.fps_low, base.fps_low)}</small> : null}
                  </td>
                  <td>{r.stalls}</td>
                  <td>{r.keys_per_s}</td>
                  <td title={`p95 ${r.key_ms_p95} ms`}>{r.key_ms}</td>
                  <td>{cell(r.gpu, '%')}</td>
                  <td>{r.vram_mb == null ? '—' : `${(r.vram_mb / 1024).toFixed(1)}G`}</td>
                  <td>{cell(r.cpu, '%')}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="row">
            <button
              type="button"
              className="btn"
              onClick={() => act(api.clearPerfTests)}
              disabled={running}
            >
              {t('dev.perfClear')}
            </button>
          </div>
        </>
      ) : null}
    </div>
  )
}
