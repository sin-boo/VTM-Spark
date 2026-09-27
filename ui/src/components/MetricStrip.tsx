type Props = {
  fps: number
  genFps?: number
  checkpoint: string
}

export function MetricStrip({ fps, genFps = 0, checkpoint }: Props) {
  return (
    <div className="metrics">
      <div className="metric">
        <span className="metric-label" title="Frames shown on the preview per second, including in-betweens">
          FPS
        </span>
        <span className="metric-value mono">{fps > 0 ? fps.toFixed(1) : '—'}</span>
      </div>
      <div className="metric">
        <span className="metric-label" title="Frames the model generated per second. FPS should be at least this.">
          Gen
        </span>
        <span className="metric-value mono">{genFps > 0 ? genFps.toFixed(1) : '—'}</span>
      </div>
      <div className="metric">
        <span className="metric-label">Model</span>
        <span className="metric-value mono">{checkpoint || ''}</span>
      </div>
    </div>
  )
}
