import { useEffect, useRef, useState, type ReactNode } from 'react'
import {
  ZERO_LAB_FEEL,
  ZERO_WEIGHTS,
  type LabFeel,
  type LabStatus,
  type MixWeights,
} from '../api'
import { useI18n, type MessageKey } from '../i18n'
import { Lamp } from './widgets'

type FeelSlider = { key: keyof LabFeel; label: MessageKey; max: number; title?: MessageKey }

const LIVE_FEEL: FeelSlider[] = [
  {
    key: 'smoothing',
    label: 'feel.smooth',
    max: 1,
    title: 'feel.smoothTitle',
  },
  { key: 'mouth', label: 'feel.mouth', max: 2 },
]

const METERS: (keyof MixWeights)[] = ['smile', 'sad', 'A', 'I', 'U', 'E']

function feelPatch(key: keyof LabFeel, value: number): Partial<LabFeel> {
  if (key === 'smoothing') return { smoothing: value, gaze_smooth: value }
  return { [key]: value }
}

// Head angles are degrees; the meter is full at this many either way.
const HEAD_RANGE = 45

function mixRows(lab: LabStatus | null): [string, number][] {
  const weights = lab?.weights ?? ZERO_WEIGHTS
  const look = lab?.look
  const head = lab?.head
  return [
    ['blink L', lab?.blink?.l ?? 0],
    ['blink R', lab?.blink?.r ?? 0],
    ['look X', look?.x ?? 0],
    ['look Y', look?.y ?? 0],
    ['yaw', head?.yaw ?? 0],
    ['pitch', head?.pitch ?? 0],
    ['roll', head?.roll ?? 0],
    ...METERS.map((name) => [name, weights[name] ?? 0] as [string, number]),
  ]
}

const HEAD_ROWS = new Set(['yaw', 'pitch', 'roll'])

// Rows are keyed by their English name (the eased meters track them by it).
const METER_LABELS: Record<string, MessageKey> = {
  'blink L': 'meter.blinkL',
  'blink R': 'meter.blinkR',
  'look X': 'meter.lookX',
  'look Y': 'meter.lookY',
  yaw: 'meter.yaw',
  pitch: 'meter.pitch',
  roll: 'meter.roll',
  smile: 'meter.smile',
  sad: 'meter.sad',
  A: 'meter.A',
  I: 'meter.I',
  U: 'meter.U',
  E: 'meter.E',
}

/** Centred meters: look is -1..1, head is degrees. */
function isCentred(name: string) {
  return name.startsWith('look') || HEAD_ROWS.has(name)
}

function meterFill(name: string, value: number) {
  const n = HEAD_ROWS.has(name)
    ? (value / HEAD_RANGE + 1) / 2
    : name.startsWith('look')
      ? (value + 1) / 2
      : value
  return `${Math.round(Math.min(1, Math.max(0, n)) * 100)}%`
}

function meterText(name: string, value: number) {
  return HEAD_ROWS.has(name) ? `${Math.round(value)}°` : value.toFixed(2)
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
    if (el?.num) el.num.textContent = meterText(name, value)
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
  const { t } = useI18n()
  return (
    <>
      {rows.map(([name, value]) => {
        const v = meters.initial(name, value)
        return (
          <li key={name} className={isCentred(name) ? 'is-look' : undefined}>
            <span>{METER_LABELS[name] ? t(METER_LABELS[name]) : name}</span>
            <i>
              <b ref={meters.bind(name, 'bar')} style={{ width: meterFill(name, v) }} />
            </i>
            <em ref={meters.bind(name, 'num')} className="mono">
              {meterText(name, v)}
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
  const head = rows.slice(4, 7)
  const mouth = rows.slice(7)
  const { t } = useI18n()
  return (
    <section className={`desk-mix${live ? ' is-live' : ''}`} aria-label={t('mix.title')}>
      <div className="char-stage-bar">
        <h2 className="group-title">{t('mix.title')}</h2>
        <span className="char-stage-name">{live ? t('mix.tracking') : t('mix.waiting')}</span>
      </div>
      <ul className="lab-meters">
        <li className="meter-kicker">{t('mix.eyes')}</li>
        <MixLane rows={eyes} meters={meters} />
        <li className="meter-kicker">{t('mix.head')}</li>
        <MixLane rows={head} meters={meters} />
        <li className="meter-kicker">{t('mix.mouth')}</li>
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
  const { t } = useI18n()
  const [draft, setDraft] = useState<LabFeel>(feel)
  const drag = useRef(false)
  const timer = useRef<number | null>(null)

  useEffect(() => {
    if (drag.current) return
    setDraft(feel)
  }, [feel.response, feel.smoothing, feel.mouth, feel.gaze_gain, feel.gaze_smooth])

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
          <span>{t(row.label)}</span>
          <input
            type="range"
            min={0}
            max={row.max}
            step={0.01}
            value={draft[row.key] ?? ZERO_LAB_FEEL[row.key]}
            title={row.title ? t(row.title) : undefined}
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
  const { t } = useI18n()
  const lamp = t(live ? 'lab.live' : online ? 'lab.connected' : 'lab.offline')

  return (
    <div className="lab-feel">
      <div className="lab-feel-head">
        <h3 className="group-subtitle">{t('feel.title')}</h3>
        <div className="lab-feel-meta">
          {props.headerExtra}
          <Lamp on={live} idle={online} label={lamp} />
        </div>
      </div>
      {props.children}
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
