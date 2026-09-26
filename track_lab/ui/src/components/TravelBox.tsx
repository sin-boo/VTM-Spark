import { useEffect, useRef, useState } from 'react'
import type { TravelBox as TravelBoxValue } from '../api'

const ROOM_MAX = 1.2
const YAW_MAX = 80
const ROLL_MAX = 80
const PITCH_UP_MAX = 50
const PITCH_DOWN_MAX = 32
const SIZE_MAX = 0.7
const VERSION = 2

export const ZERO_TRAVEL: TravelBoxValue = {
  version: VERSION,
  enabled: true,
  left: 0.68,
  right: 0.54,
  up: 0.57,
  down: 0.07,
  body_left: 0.54,
  body_right: 0.48,
  body_up: 0.1,
  body_down: 0.12,
  turn_left: 12,
  turn_right: 22,
  tilt_left: 22,
  tilt_right: 12,
  pitch_up: 14,
  pitch_down: 3,
  eye: 0.56,
  size: 0,
}

export type TravelFocus = 'head' | 'body'

type RoomKey = 'left' | 'right' | 'up' | 'down' | 'body_left' | 'body_right' | 'body_up' | 'body_down'

const HEAD_ROOM: [RoomKey, string][] = [
  ['left', 'Left'],
  ['right', 'Right'],
  ['up', 'Up'],
  ['down', 'Down'],
]

type TurnKey = 'turn_left' | 'turn_right' | 'tilt_left' | 'tilt_right'

// Screen sides of the character: which way the face turns, or the crown tips.
const HEAD_TURN: [TurnKey, string, number][] = [
  ['turn_left', 'Turn left', YAW_MAX],
  ['turn_right', 'Turn right', YAW_MAX],
  ['tilt_left', 'Tilt left', ROLL_MAX],
  ['tilt_right', 'Tilt right', ROLL_MAX],
]

const BODY_ROOM: [RoomKey, string][] = [
  ['body_left', 'Left'],
  ['body_right', 'Right'],
  ['body_up', 'Up'],
  ['body_down', 'Down'],
]

type Props = {
  value?: TravelBoxValue | null
  disabled?: boolean
  onChange: (patch: Partial<TravelBoxValue>) => void
  onFocus?: (focus: TravelFocus | null) => void
}

function clampRoom(n: number) {
  if (!Number.isFinite(n)) return 0
  return Math.max(0, Math.min(ROOM_MAX, n))
}

function clampDeg(n: number, max: number) {
  if (!Number.isFinite(n)) return max
  return Math.max(0, Math.min(max, n))
}

function clampEye(n: number) {
  if (!Number.isFinite(n)) return ZERO_TRAVEL.eye
  return Math.max(0, Math.min(1, n))
}

function clampSize(n: number) {
  if (!Number.isFinite(n)) return ZERO_TRAVEL.size
  return Math.max(0, Math.min(SIZE_MAX, n))
}

function fill(raw: TravelBoxValue | null | undefined): TravelBoxValue {
  const fresh = raw?.version === VERSION
  const room = (key: RoomKey) => clampRoom(fresh ? (raw?.[key] ?? ZERO_TRAVEL[key]) : ZERO_TRAVEL[key])
  return {
    version: VERSION,
    enabled: raw?.enabled !== false,
    left: room('left'),
    right: room('right'),
    up: room('up'),
    down: room('down'),
    body_left: room('body_left'),
    body_right: room('body_right'),
    body_up: room('body_up'),
    body_down: room('body_down'),
    turn_left: clampDeg(raw?.turn_left ?? raw?.yaw ?? ZERO_TRAVEL.turn_left, YAW_MAX),
    turn_right: clampDeg(raw?.turn_right ?? raw?.yaw ?? ZERO_TRAVEL.turn_right, YAW_MAX),
    tilt_left: clampDeg(raw?.tilt_left ?? raw?.roll ?? ZERO_TRAVEL.tilt_left, ROLL_MAX),
    tilt_right: clampDeg(raw?.tilt_right ?? raw?.roll ?? ZERO_TRAVEL.tilt_right, ROLL_MAX),
    pitch_up: clampDeg(raw?.pitch_up ?? ZERO_TRAVEL.pitch_up, PITCH_UP_MAX),
    pitch_down: clampDeg(raw?.pitch_down ?? ZERO_TRAVEL.pitch_down, PITCH_DOWN_MAX),
    eye: clampEye(raw?.eye ?? ZERO_TRAVEL.eye),
    size: clampSize(raw?.size ?? ZERO_TRAVEL.size),
  }
}

export function TravelBox(props: Props) {
  const live = fill(props.value)
  const [draft, setDraft] = useState(live)
  const [focus, setFocus] = useState<TravelFocus | null>(null)
  const drag = useRef(false)
  const timer = useRef<number | null>(null)
  const liveKey = JSON.stringify(live)

  function pick(next: TravelFocus) {
    setFocus((cur) => {
      const value = cur === next ? null : next
      props.onFocus?.(value)
      return value
    })
  }

  const show = (id: TravelFocus) => focus === null || focus === id

  useEffect(() => {
    if (drag.current) return
    setDraft(live)
  }, [liveKey])

  useEffect(() => {
    return () => {
      if (timer.current != null) window.clearTimeout(timer.current)
    }
  }, [])

  function commit(next: TravelBoxValue, immediate = false) {
    setDraft(next)
    if (timer.current != null) window.clearTimeout(timer.current)
    drag.current = true
    const send = () => {
      props.onChange(next)
      drag.current = false
    }
    if (immediate) {
      send()
      return
    }
    timer.current = window.setTimeout(send, 80)
  }

  const off = props.disabled || !draft.enabled

  return (
    <div className={`travel-box${draft.enabled ? '' : ' is-off'}`}>
      <div className="travel-box-head">
        <p className="side-label">Limiters</p>
        <div className="travel-box-actions">
          <button
            type="button"
            className="ghost compact"
            disabled={props.disabled}
            onClick={() => commit({ ...ZERO_TRAVEL, enabled: draft.enabled }, true)}
            title="Reset head, body, turn, and eye limits"
          >
            Reset
          </button>
          <label className="travel-toggle">
            <input
              type="checkbox"
              checked={draft.enabled}
              disabled={props.disabled}
              onChange={(e) => commit({ ...draft, enabled: e.target.checked }, true)}
            />
            <span>On</span>
          </label>
        </div>
      </div>
      <ul className="travel-legend">
        <li>
          <button
            type="button"
            className={`is-head${focus === 'head' ? ' is-on' : ''}`}
            aria-pressed={focus === 'head'}
            title="Show only the head wall"
            onClick={() => pick('head')}
          >
            <i className="travel-swatch is-head" />
            Head
          </button>
        </li>
        <li>
          <button
            type="button"
            className={`is-body${focus === 'body' ? ' is-on' : ''}`}
            aria-pressed={focus === 'body'}
            title="Show only the body wall"
            onClick={() => pick('body')}
          >
            <i className="travel-swatch is-body" />
            Body
          </button>
        </li>
      </ul>

      {show('head') ? (
        <div className="travel-block">
          <p className="travel-sub">Head room</p>
          <ul className="feel travel-readout">
            {HEAD_ROOM.map(([key, label]) => (
              <RangeRow
                key={key}
                label={label}
                value={draft[key]}
                disabled={off}
                kind="room"
                onChange={(n) => commit({ ...draft, [key]: n })}
              />
            ))}
            {HEAD_TURN.map(([key, label, max]) => (
              <RangeRow
                key={key}
                label={label}
                value={draft[key]}
                max={max}
                disabled={off}
                kind="deg"
                onChange={(n) => commit({ ...draft, [key]: n })}
              />
            ))}
          </ul>
        </div>
      ) : null}

      {show('body') ? (
        <div className="travel-block">
          <p className="travel-sub">Body room</p>
          <ul className="feel travel-readout">
            {BODY_ROOM.map(([key, label]) => (
              <RangeRow
                key={key}
                label={label}
                value={draft[key]}
                disabled={off}
                kind="room"
                onChange={(n) => commit({ ...draft, [key]: n })}
              />
            ))}
          </ul>
        </div>
      ) : null}

      <div className="travel-block">
        <p className="travel-sub">Look</p>
        <ul className="feel travel-readout">
          <RangeRow
            label="Look up"
            value={draft.pitch_up}
            max={PITCH_UP_MAX}
            disabled={off}
            kind="deg"
            onChange={(pitch_up) => commit({ ...draft, pitch_up })}
          />
          <RangeRow
            label="Look down"
            value={draft.pitch_down}
            max={PITCH_DOWN_MAX}
            disabled={off}
            kind="deg"
            onChange={(pitch_down) => commit({ ...draft, pitch_down })}
          />
          <RangeRow label="Eyes" value={draft.eye} disabled={off} kind="eye" onChange={(eye) => commit({ ...draft, eye })} />
          <RangeRow label="Size" value={draft.size} disabled={off} kind="size" onChange={(size) => commit({ ...draft, size })} />
        </ul>
      </div>
    </div>
  )
}

function RangeRow(props: {
  label: string
  value: number
  disabled: boolean
  kind: 'room' | 'eye' | 'deg' | 'size'
  max?: number
  onChange: (value: number) => void
}) {
  const kind = props.kind
  const max = props.max ?? (kind === 'eye' ? 1 : kind === 'size' ? SIZE_MAX : ROOM_MAX)
  const step = kind === 'deg' ? 1 : 0.01
  const clamp = (n: number) =>
    kind === 'eye' ? clampEye(n) : kind === 'size' ? clampSize(n) : kind === 'deg' ? clampDeg(n, max) : clampRoom(n)
  return (
    <li>
      <span>{props.label}</span>
      <input
        type="range"
        min={0}
        max={max}
        step={step}
        value={props.value}
        disabled={props.disabled}
        onChange={(e) => props.onChange(clamp(Number(e.target.value)))}
      />
      <ValueBox
        label={props.label}
        value={props.value}
        disabled={props.disabled}
        kind={kind}
        clamp={clamp}
        onChange={props.onChange}
      />
    </li>
  )
}

// Typed values use the units the readout shows: degrees, percent, or face heights.
function ValueBox(props: {
  label: string
  value: number
  disabled: boolean
  kind: 'room' | 'eye' | 'deg' | 'size'
  clamp: (n: number) => number
  onChange: (value: number) => void
}) {
  const kind = props.kind
  const scale = kind === 'eye' || kind === 'size' ? 100 : 1
  const shown = (props.value * scale).toFixed(kind === 'room' ? 2 : 0)
  const [typed, setTyped] = useState<string | null>(null)
  const cancel = useRef(false)

  function apply() {
    const raw = typed
    setTyped(null)
    if (cancel.current || raw == null) {
      cancel.current = false
      return
    }
    const n = Number(raw.replace(/[^0-9.-]/g, ''))
    if (raw.trim() === '' || !Number.isFinite(n)) return
    props.onChange(props.clamp(n / scale))
  }

  return (
    <em className="travel-num">
      {kind === 'size' ? <i>±</i> : null}
      <input
        type="text"
        inputMode="decimal"
        aria-label={`${props.label} value`}
        value={typed ?? shown}
        disabled={props.disabled}
        onFocus={(e) => e.currentTarget.select()}
        onChange={(e) => setTyped(e.target.value)}
        onBlur={apply}
        onKeyDown={(e) => {
          if (e.key === 'Enter') e.currentTarget.blur()
          if (e.key === 'Escape') {
            cancel.current = true
            e.currentTarget.blur()
          }
        }}
      />
      {kind === 'eye' || kind === 'size' ? <i>%</i> : kind === 'deg' ? <i>°</i> : null}
    </em>
  )
}
