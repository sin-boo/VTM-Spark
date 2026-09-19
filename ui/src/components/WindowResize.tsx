import { useRef, type PointerEvent } from 'react'
import { beginResize, endResize, moveResize, type ResizeEdge } from '../nativeWindow'

const HITS: { edge: ResizeEdge; side: string }[] = [
  { edge: 'left', side: 'w' },
  { edge: 'right', side: 'e' },
  { edge: 'top', side: 'n' },
  { edge: 'bottom', side: 's' },
  { edge: 'topleft', side: 'nw' },
  { edge: 'topright', side: 'ne' },
  { edge: 'bottomleft', side: 'sw' },
  { edge: 'bottomright', side: 'se' },
]

export function WindowResize() {
  const active = useRef<ResizeEdge | null>(null)
  const frame = useRef(0)

  function pump() {
    frame.current = 0
    if (!active.current) return
    moveResize()
    frame.current = window.requestAnimationFrame(pump)
  }

  function onDown(edge: ResizeEdge, e: PointerEvent<HTMLDivElement>) {
    if (e.button !== 0) return
    e.preventDefault()
    e.stopPropagation()
    active.current = edge
    e.currentTarget.setPointerCapture(e.pointerId)
    void Promise.resolve(beginResize(edge)).finally(() => {
      if (!active.current) return
      if (!frame.current) frame.current = window.requestAnimationFrame(pump)
    })
  }

  function onUp(e: PointerEvent<HTMLDivElement>) {
    if (active.current == null) return
    active.current = null
    if (frame.current) window.cancelAnimationFrame(frame.current)
    frame.current = 0
    try {
      e.currentTarget.releasePointerCapture(e.pointerId)
    } catch {
      /* already released */
    }
    endResize()
  }

  return (
    <div className="win-resize" aria-hidden="true">
      {HITS.map(({ edge, side }) => (
        <div
          key={edge}
          className={`win-resize-hit is-${side}`}
          onPointerDown={(e) => onDown(edge, e)}
          onPointerUp={onUp}
          onPointerCancel={onUp}
        />
      ))}
    </div>
  )
}
