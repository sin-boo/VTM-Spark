type Props = {
  fps: number
  timing: string
  checkpoint: string
}

export function MetricStrip({ fps, timing, checkpoint }: Props) {
  return (
    <div className="metrics">
      <div className="metric">
        <span className="metric-label">Generate FPS</span>
        <span className="metric-value mono">{fps > 0 ? fps.toFixed(1) : '—'}</span>
      </div>
      <div className="metric">
        <span className="metric-label">Timing</span>
        <span className="metric-value mono">{timing || '—'}</span>
      </div>
      <div className="metric">
        <span className="metric-label">Model</span>
        <span className="metric-value mono">{checkpoint || ''}</span>
      </div>
    </div>
  )
}
