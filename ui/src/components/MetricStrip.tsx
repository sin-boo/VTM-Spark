type Props = {
  fps: number
  genFps?: number
  timing: string
  checkpoint: string
}

export function MetricStrip({ fps, genFps = 0, timing, checkpoint }: Props) {
  return (
    <div className="metrics">
      <div className="metric">
        <span className="metric-label" title="Pictures that landed on the preview, including in-betweens">
          Shown
        </span>
        <span className="metric-value mono">{fps > 0 ? fps.toFixed(1) : '—'}</span>
      </div>
      <div className="metric">
        <span className="metric-label" title="Generated keys that landed on the preview. Shown should be at least this.">
          Generated
        </span>
        <span className="metric-value mono">{genFps > 0 ? genFps.toFixed(1) : '—'}</span>
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
