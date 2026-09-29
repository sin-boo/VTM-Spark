import { useI18n } from '../i18n'

type Props = {
  fps: number
  genFps?: number
  /** Whole-card GPU %, or null/undefined to hide. */
  gpuUtil?: number | null
  checkpoint: string
}

export function MetricStrip({ fps, genFps = 0, gpuUtil, checkpoint }: Props) {
  const { t } = useI18n()
  return (
    <div className="metrics">
      <div className="metric">
        <span className="metric-label" title={t('metric.fpsTitle')}>
          {t('metric.fps')}
        </span>
        <span className="metric-value mono">{fps > 0 ? fps.toFixed(1) : '—'}</span>
      </div>
      <div className="metric">
        <span className="metric-label" title={t('metric.genTitle')}>
          {t('metric.gen')}
        </span>
        <span className="metric-value mono">{genFps > 0 ? genFps.toFixed(1) : '—'}</span>
      </div>
      {gpuUtil != null ? (
        <div className="metric">
          <span className="metric-label" title={t('metric.gpuTitle')}>
            {t('metric.gpu')}
          </span>
          <span className="metric-value mono">{gpuUtil}%</span>
        </div>
      ) : null}
      <div className="metric">
        <span className="metric-label">{t('metric.model')}</span>
        <span className="metric-value mono">{checkpoint || ''}</span>
      </div>
    </div>
  )
}
