import { useEffect, useRef, useState } from 'react'
import { api } from '../api'

type Props = {
  image: string | null
  live: boolean
  frozen?: boolean
  stillId?: string
}

const ZOOM_MIN = 1
const ZOOM_MAX = 8

export function CelStage({ image, live, frozen = false, stillId = '' }: Props) {
  const imgRef = useRef<HTMLImageElement>(null)
  const viewRef = useRef<HTMLDivElement>(null)
  const dragging = useRef(false)
  const panning = useRef(false)
  const dragChain = useRef(Promise.resolve())
  const lastXY = useRef<{ x: number; y: number } | null>(null)
  const raf = useRef(0)
  const panStart = useRef({ x: 0, y: 0, panX: 0, panY: 0 })
  const [zoom, setZoom] = useState(1)
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const [isPanning, setIsPanning] = useState(false)
  const zoomRef = useRef(1)

  useEffect(() => {
    zoomRef.current = zoom
  }, [zoom])

  useEffect(() => {
    zoomRef.current = 1
    setZoom(1)
    setPan({ x: 0, y: 0 })
    panning.current = false
    setIsPanning(false)
  }, [stillId])

  useEffect(() => {
    if (!frozen) {
      zoomRef.current = 1
      setZoom(1)
      setPan({ x: 0, y: 0 })
      panning.current = false
      setIsPanning(false)
    }
  }, [frozen])

  useEffect(() => {
    const view = viewRef.current
    if (!view || !frozen) return
    function onWheel(e: WheelEvent) {
      e.preventDefault()
      const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12
      const prev = zoomRef.current
      const next = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, prev * factor))
      zoomRef.current = next
      setZoom(next)
      if (next <= ZOOM_MIN + 1e-3) {
        setPan({ x: 0, y: 0 })
      }
    }
    view.addEventListener('wheel', onWheel, { passive: false })
    return () => view.removeEventListener('wheel', onWheel)
  }, [frozen])

  function toImageXY(clientX: number, clientY: number) {
    const el = imgRef.current
    if (!el || !el.naturalWidth) return null
    const rect = el.getBoundingClientRect()
    if (rect.width <= 0 || rect.height <= 0) return null
    const x = ((clientX - rect.left) / rect.width) * el.naturalWidth
    const y = ((clientY - rect.top) / rect.height) * el.naturalHeight
    return { x, y }
  }

  function queueDrag(xy: { x: number; y: number }) {
    lastXY.current = xy
    if (raf.current) return
    raf.current = requestAnimationFrame(() => {
      raf.current = 0
      const p = lastXY.current
      if (!p || !dragging.current) return
      dragChain.current = dragChain.current
        .then(() => api.meshDrag(p.x, p.y))
        .then(() => undefined)
        .catch(() => undefined)
    })
  }

  function endDrag() {
    dragging.current = false
    panning.current = false
    setIsPanning(false)
    lastXY.current = null
    if (raf.current) {
      cancelAnimationFrame(raf.current)
      raf.current = 0
    }
    dragChain.current = dragChain.current
      .then(() => api.meshRelease())
      .then(() => undefined)
      .catch(() => undefined)
  }

  return (
    <section className="stage">
      <div className="stage-head">
        <h2 className="stage-title">Cel</h2>
        <div className="stage-pills">
          {frozen ? <span className="stage-flag">Frozen</span> : null}
          {live && !frozen ? <span className="stage-flag is-live">Live</span> : null}
          {frozen && zoom > 1.01 ? <span className="stage-flag">{zoom.toFixed(1)}×</span> : null}
        </div>
      </div>

      <div
        ref={viewRef}
        className={`cel ${live ? 'is-live' : ''} ${frozen ? 'is-frozen' : ''} ${isPanning ? 'is-panning' : ''}`}
      >
        {image ? (
          <div
            className="cel-zoom"
            style={{
              transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
            }}
          >
            <img
              ref={imgRef}
              className="cel-image"
              src={`data:image/jpeg;base64,${image}`}
              alt=""
              draggable={false}
              onPointerDown={(e) => {
                if (e.button !== 0) return
                e.preventDefault()
                ;(e.target as HTMLElement).setPointerCapture(e.pointerId)
                if (frozen && e.shiftKey) {
                  panning.current = true
                  setIsPanning(true)
                  panStart.current = {
                    x: e.clientX,
                    y: e.clientY,
                    panX: pan.x,
                    panY: pan.y,
                  }
                  return
                }
                const xy = toImageXY(e.clientX, e.clientY)
                if (!xy) return
                dragging.current = true
                dragChain.current = dragChain.current
                  .then(() => api.meshPress(xy.x, xy.y))
                  .then(() => undefined)
                  .catch(() => undefined)
              }}
              onPointerMove={(e) => {
                if (panning.current) {
                  setPan({
                    x: panStart.current.panX + (e.clientX - panStart.current.x),
                    y: panStart.current.panY + (e.clientY - panStart.current.y),
                  })
                  return
                }
                if (!dragging.current) return
                if (e.buttons === 0) {
                  endDrag()
                  return
                }
                const xy = toImageXY(e.clientX, e.clientY)
                if (!xy) return
                queueDrag(xy)
              }}
              onPointerUp={endDrag}
              onPointerCancel={endDrag}
              onDoubleClick={() => {
                void api.meshReset()
              }}
            />
          </div>
        ) : null}
      </div>
      {frozen ? (
        <p className="hint">
          Frozen. Drag points, Shift-drag to pan, scroll to zoom. Double-click to undo.
        </p>
      ) : null}
    </section>
  )
}
