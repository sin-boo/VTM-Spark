import { useEffect, useRef, useState, type ReactNode } from 'react'
import {
  ZERO_LAB_FEEL,
  ZERO_WEIGHTS,
  type LabFeel,
  type LabStatus,
  type MixWeights,
} from '../api'
import { Lamp } from './widgets'

type FeelSlider = { key: keyof LabFeel; label: string; max: number }

const LIVE_FEEL: FeelSlider[] = [
  { key: 'mouth', label: 'Mouth', max: 2 },
  { key: 'hair_pin', label: 'Hair pin', max: 1 },
]

const METERS: (keyof MixWeights)[] = ['smile', 'sad', 'A', 'I', 'U', 'E', 'O']

function feelPatch(key: keyof LabFeel, value: number): Partial<LabFeel> {
  if (key === 'smoothing') return { smoothing: value, gaze_smooth: value }
  return { [key]: value }
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
  const weights = props.lab?.weights ?? ZERO_WEIGHTS
  const look = props.lab?.look
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

      {live ? (
        <ul className="lab-meters">
          {(
            [
              ['blink L', props.lab?.blink?.l ?? 0],
              ['blink R', props.lab?.blink?.r ?? 0],
              ['look X', look?.x ?? 0],
              ['look Y', look?.y ?? 0],
              ...METERS.map((name) => [name, weights[name] ?? 0] as [string, number]),
            ] satisfies [string, number][]
          ).map(([name, value]) => (
            <li key={name}>
              <span>{name}</span>
              <i>
                <b
                  style={{
                    width: `${Math.round(Math.min(1, Math.max(0, name.startsWith('look') ? (value + 1) / 2 : value)) * 100)}%`,
                  }}
                />
              </i>
              <em className="mono">{value.toFixed(2)}</em>
            </li>
          ))}
        </ul>
      ) : null}
      </div>
    </div>
  )
}
