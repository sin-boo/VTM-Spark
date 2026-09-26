import { useEffect, useRef, useState, type ReactNode } from 'react'
import {
  ZERO_LAB_FEEL,
  ZERO_WEIGHTS,
  type LabFeel,
  type LabStatus,
  type MixWeights,
} from '../api'
import { Lamp } from './widgets'

type FeelSlider = { key: keyof LabFeel; label: string; max: number; title?: string }

const LIVE_FEEL: FeelSlider[] = [
  {
    key: 'smoothing',
    label: 'Smooth',
    max: 1,
    title: 'Ease each new pose toward the last one. Higher is smoother. Eyes use the same ease.',
  },
  { key: 'mouth', label: 'Mouth', max: 2 },
  { key: 'hair_pin', label: 'Hair pin', max: 1 },
]

const METERS: (keyof MixWeights)[] = ['smile', 'sad', 'A', 'I', 'U', 'E', 'O']

function feelPatch(key: keyof LabFeel, value: number): Partial<LabFeel> {
  if (key === 'smoothing') return { smoothing: value, gaze_smooth: value }
  return { [key]: value }
}

function mixRows(lab: LabStatus | null): [string, number][] {
  const weights = lab?.weights ?? ZERO_WEIGHTS
  const look = lab?.look
  return [
    ['blink L', lab?.blink?.l ?? 0],
    ['blink R', lab?.blink?.r ?? 0],
    ['look X', look?.x ?? 0],
    ['look Y', look?.y ?? 0],
    ...METERS.map((name) => [name, weights[name] ?? 0] as [string, number]),
  ]
}

function meterFill(name: string, value: number) {
  const n = name.startsWith('look') ? (value + 1) / 2 : value
  return `${Math.round(Math.min(1, Math.max(0, n)) * 100)}%`
}

// Status arrives every ~250 ms; ease the meters toward each sample per frame so
// they glide instead of stepping. ~80 ms time constant keeps them responsive.
const METER_TAU = 0.08

type MeterEls = { bar: HTMLElement | null; num: HTMLElement | null }

function useEasedMeters(rows: [string, number][]) {
  const els = useRef(new Map<string, MeterEls>())
  const shown = useRef(new Map<string, number>())
  const target = useRef(new Map<string, number>())
  const raf = useRef(0)
  const last = useRef(0)

  const paint = (name: string, value: number) => {
    const el = els.current.get(name)
    if (el?.bar) el.bar.style.width = meterFill(name, value)
    if (el?.num) el.num.textContent = value.toFixed(2)
  }

  const step = (now: number) => {
    const dt = last.current ? Math.min(0.1, (now - last.current) / 1000) : 1 / 60
    last.current = now
    const k = 1 - Math.exp(-dt / METER_TAU)
    let moving = false
    for (const [name, goal] of target.current) {
      const cur = shown.current.get(name) ?? goal
      let next = cur + (goal - cur) * k
      if (Math.abs(goal - next) < 0.001) next = goal
      else moving = true
      shown.current.set(name, next)
      paint(name, next)
    }
    raf.current = moving ? window.requestAnimationFrame(step) : 0
    if (!moving) last.current = 0
  }

  useEffect(() => {
    for (const [name, value] of rows) target.current.set(name, value)
    if (!raf.current) raf.current = window.requestAnimationFrame(step)
  })

  useEffect(() => () => window.cancelAnimationFrame(raf.current), [])

  const bind = (name: string, part: keyof MeterEls) => (node: HTMLElement | null) => {
    const el = els.current.get(name) ?? { bar: null, num: null }
    el[part] = node
    els.current.set(name, el)
  }
  const initial = (name: string, value: number) => shown.current.get(name) ?? value
  return { bind, initial }
}

function MixLane(props: {
  rows: [string, number][]
  meters: ReturnType<typeof useEasedMeters>
}) {
  const { rows, meters } = props
  return (
    <>
      {rows.map(([name, value]) => {
        const v = meters.initial(name, value)
        return (
          <li key={name} className={name.startsWith('look') ? 'is-look' : undefined}>
            <span>{name}</span>
            <i>
              <b ref={meters.bind(name, 'bar')} style={{ width: meterFill(name, v) }} />
            </i>
            <em ref={meters.bind(name, 'num')} className="mono">
              {v.toFixed(2)}
            </em>
          </li>
        )
      })}
    </>
  )
}

export function MixMeters({ lab }: { lab: LabStatus | null }) {
  const live = Boolean(lab?.live)
  const rows = mixRows(lab)
  const meters = useEasedMeters(rows)
  const eyes = rows.slice(0, 4)
  const mouth = rows.slice(4)
  return (
    <section className={`desk-mix${live ? ' is-live' : ''}`} aria-label="Live">
      <div className="char-stage-bar">
        <h2 className="group-title">Live</h2>
        <span className="char-stage-name">{live ? 'Tracking' : 'Waiting'}</span>
      </div>
      <ul className="lab-meters">
        <li className="meter-kicker">Eyes</li>
        <MixLane rows={eyes} meters={meters} />
        <li className="meter-kicker">Mouth</li>
        <MixLane rows={mouth} meters={meters} />
      </ul>
    </section>
  )
}

type SliderProps = {
  rows: FeelSlider[]
  feel: LabFeel
  online?: boolean
  busy?: boolean
  onFeel: (patch: Partial<LabFeel>) => void
}

function FeelSliders(props: SliderProps) {
  const { rows, feel, online, busy, onFeel } = props
  const [draft, setDraft] = useState<LabFeel>(feel)
  const drag = useRef(false)
  const timer = useRef<number | null>(null)

  useEffect(() => {
    if (drag.current) return
    setDraft(feel)
  }, [feel.response, feel.smoothing, feel.mouth, feel.hair_pin, feel.gaze_gain, feel.gaze_smooth])

  useEffect(() => {
    return () => {
      if (timer.current != null) window.clearTimeout(timer.current)
    }
  }, [])

  function commit(next: LabFeel, patch: Partial<LabFeel>) {
    setDraft(next)
    if (timer.current != null) window.clearTimeout(timer.current)
    drag.current = true
    timer.current = window.setTimeout(() => {
      drag.current = false
      onFeel(patch)
    }, 80)
  }

  return (
    <ul className="lab-sliders">
      {rows.map((row) => (
        <li key={row.key}>
          <span>{row.label}</span>
          <input
            type="range"
            min={0}
            max={row.max}
            step={0.01}
            value={draft[row.key] ?? ZERO_LAB_FEEL[row.key]}
            title={row.title}
            disabled={!online || busy}
            onChange={(e) => {
              const value = Number(e.target.value)
              const next = { ...draft, ...feelPatch(row.key, value) }
              commit(next, feelPatch(row.key, value))
            }}
          />
          <em className="mono">{(draft[row.key] ?? ZERO_LAB_FEEL[row.key]).toFixed(2)}</em>
        </li>
      ))}
    </ul>
  )
}

type Props = {
  lab: LabStatus | null
  busy?: boolean
  onFeel: (patch: Partial<LabFeel>) => void
  headerExtra?: ReactNode
  actions?: ReactNode
  children?: ReactNode
}

export function LabFeel(props: Props) {
  const online = Boolean(props.lab?.online)
  const live = Boolean(props.lab?.live)
  const feel = props.lab?.feel ?? ZERO_LAB_FEEL
  const lamp = live ? 'Track Lab live' : online ? 'Track Lab connected' : 'Track Lab offline'

  return (
    <div className="lab-feel">
      <div className="lab-feel-head">
        <h3 className="group-subtitle">Feel</h3>
        <div className="lab-feel-meta">
          {props.headerExtra}
          <Lamp on={live} idle={online} label={lamp} />
        </div>
      </div>
      {props.children}
      {!online ? (
        <p className="hint">Start Track Lab to edit overlay and rest.</p>
      ) : null}
      {props.actions}
      <div className={online ? undefined : 'is-offline'}>
        <FeelSliders
          rows={LIVE_FEEL}
          feel={feel}
          online={online}
          busy={props.busy}
          onFeel={props.onFeel}
        />
      </div>
    </div>
  )
}
