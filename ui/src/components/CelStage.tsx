import { useRef } from 'react'
import { api } from '../api'

type Props = {
  image: string | null
  live: boolean
}

export function CelStage({ image, live }: Props) {
  const imgRef = useRef<HTMLImageElement>(null)
  const dragging = useRef(false)

  function toImageXY(clientX: number, clientY: number) {
    const el = imgRef.current
    if (!el || !el.naturalWidth) return null
    const rect = el.getBoundingClientRect()
    const x = ((clientX - rect.left) / rect.width) * el.naturalWidth
    const y = ((clientY - rect.top) / rect.height) * el.naturalHeight
    return { x, y }
  }

  return (
    <section className="stage">
      <div className="stage-head">
        <h2 className="stage-title">Live cel</h2>
        {live ? <span className="live-pill">On air</span> : <span className="idle-pill">Idle</span>}
      </div>

      <div className={`cel ${live ? 'is-live' : ''}`}>
        <span className="tick tick-tl" aria-hidden />
        <span className="tick tick-tr" aria-hidden />
        <span className="tick tick-bl" aria-hidden />
        <span className="tick tick-br" aria-hidden />

        {image ? (
          <img
            ref={imgRef}
            className="cel-image"
            src={`data:image/jpeg;base64,${image}`}
            alt="Generated frame"
            draggable={false}
            onPointerDown={(e) => {
              const xy = toImageXY(e.clientX, e.clientY)
              if (!xy) return
              dragging.current = true
              ;(e.target as HTMLElement).setPointerCapture(e.pointerId)
              void api.meshPress(xy.x, xy.y)
            }}
            onPointerMove={(e) => {
              if (!dragging.current) return
              const xy = toImageXY(e.clientX, e.clientY)
              if (!xy) return
              void api.meshDrag(xy.x, xy.y)
            }}
            onPointerUp={() => {
              dragging.current = false
              void api.meshRelease()
            }}
            onDoubleClick={() => {
              void api.meshReset()
            }}
          />
        ) : (
          <div className="cel-empty">
            <p>No frame yet</p>
            <p className="hint">Apply a reference, then generate once or start stream.</p>
          </div>
        )}
      </div>
    </section>
  )
}
