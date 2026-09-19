import { useEffect, useRef, useState } from 'react'
import type { TravelBox as TravelBoxValue } from '../api'
import { Toggle } from './widgets'

const ROOM_MAX = 1.2
const PAD_MAX = 200
const YAW_MAX = 80
const ROLL_MAX = 80
const PITCH_UP_MAX = 50
const PITCH_DOWN_MAX = 32
const BODY_YAW_MAX = 80
const BODY_ROLL_MAX = 80

const ZERO: TravelBoxValue = {
  enabled: true,
  side: true,
  rotate: true,
  look_up: true,
  look_down: true,
  body: true,
  body_rotate: true,
  eyes: true,
  left: 0,
  right: 0,
  up: 0,
  down: 0,
  body_left: 0,
  body_right: 0,
  body_up: 0,
  body_down: 0,
  yaw: YAW_MAX,
  roll: ROLL_MAX,
  pitch_up: PITCH_UP_MAX,
  pitch_down: PITCH_DOWN_MAX,
  body_yaw: BODY_YAW_MAX,
  body_roll: BODY_ROLL_MAX,
  eye_x: 0.78,
  eye_y: 0.78,
  pad_px: 50,
}

type Props = {
  value?: TravelBoxValue | null
  disabled?: boolean
  onChange: (patch: TravelBoxValue) => void
}

function fill(raw: TravelBoxValue | null | undefined): TravelBoxValue {
  return {
    enabled: raw?.enabled !== false,
    side: raw?.side !== false,
    rotate: raw?.rotate !== false,
    look_up: raw?.look_up !== false,
    look_down: raw?.look_down !== false,
    body: raw?.body !== false,
    body_rotate: raw?.body_rotate !== false,
    eyes: raw?.eyes !== false,
    left: clampRoom(raw?.left ?? ZERO.left),
    right: clampRoom(raw?.right ?? ZERO.right),
    up: 0,
    down: 0,
    body_left: clampRoom(raw?.body_left ?? ZERO.body_left),
    body_right: clampRoom(raw?.body_right ?? ZERO.body_right),
    body_up: clampRoom(raw?.body_up ?? ZERO.body_up),
    body_down: clampRoom(raw?.body_down ?? ZERO.body_down),
    yaw: clampDeg(raw?.yaw ?? ZERO.yaw, YAW_MAX),
    roll: clampDeg(raw?.roll ?? ZERO.roll, ROLL_MAX),
    pitch_up: clampDeg(raw?.pitch_up ?? ZERO.pitch_up, PITCH_UP_MAX),
    pitch_down: clampDeg(raw?.pitch_down ?? ZERO.pitch_down, PITCH_DOWN_MAX),
    body_yaw: clampDeg(raw?.body_yaw ?? ZERO.body_yaw, BODY_YAW_MAX),
    body_roll: clampDeg(raw?.body_roll ?? ZERO.body_roll, BODY_ROLL_MAX),
    eye_x: clampEye(raw?.eye_x ?? ZERO.eye_x),
    eye_y: clampEye(raw?.eye_y ?? ZERO.eye_y),
    pad_px: clampPad(raw?.pad_px ?? ZERO.pad_px),
  }
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
  if (!Number.isFinite(n)) return ZERO.eye_x
  return Math.max(0, Math.min(1, n))
}

function clampPad(n: number) {
  if (!Number.isFinite(n)) return ZERO.pad_px
  return Math.max(0, Math.min(PAD_MAX, Math.round(n)))
}

export function TravelBox(props: Props) {
  const live = fill(props.value)
  const [draft, setDraft] = useState(live)
  const drag = useRef(false)
  const timer = useRef<number | null>(null)

  useEffect(() => {
    if (drag.current) return
    setDraft(live)
  }, [
    live.enabled,
    live.side,
    live.rotate,
    live.look_up,
    live.look_down,
    live.body,
    live.body_rotate,
    live.eyes,
    live.left,
    live.right,
    live.body_left,
    live.body_right,
    live.body_up,
    live.body_down,
    live.yaw,
    live.roll,
    live.pitch_up,
    live.pitch_down,
    live.body_yaw,
    live.body_roll,
    live.eye_x,
    live.eye_y,
    live.pad_px,
  ])

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

  const masterOff = props.disabled || !draft.enabled

  return (
    <div className={`travel-box${draft.enabled ? '' : ' is-off'}`}>
      <div className="group-head travel-box-head">
        <h2 className="group-title">Limiters</h2>
        <div className="travel-box-actions">
          <button
            type="button"
            className="btn ghost compact"
            disabled={props.disabled}
            onClick={() => commit({ ...ZERO, enabled: draft.enabled }, true)}
            title="Reset size, head, look, eyes, and skeleton limits"
          >
            Reset
          </button>
          <Toggle
            label="On"
            checked={draft.enabled}
            disabled={props.disabled}
            onChange={(enabled) => commit({ ...draft, enabled }, true)}
          />
        </div>
      </div>
      <ul className="travel-legend">
        <li title="How far the whole character may scale before a walk is pushed back.">
          <i className="travel-swatch is-size" />
          Size
          <span className="travel-kind">Scale</span>
        </li>
        <li title="Pink face overlay. Drag a slider and the head walks to that max.">
          <i className="travel-swatch is-head" />
          Head
          <span className="travel-kind">Form</span>
        </li>
        <li title="Blue shoulders, arms, and chest. Drag a slider and the skeleton walks to that max.">
          <i className="travel-swatch is-body" />
          Skeleton
          <span className="travel-kind">Form</span>
        </li>
        <li title="Look, turn, and tilt. The overlay nods or twists to the angle you set.">
          <span className="travel-kind">Rotation</span>
        </li>
      </ul>

      <div className="travel-limits">
        <div className="travel-block">
          <div className="travel-block-head">
            <h3
              className="group-subtitle"
              title="How far the whole character drawing, hair included, may scale"
            >
              Size
            </h3>
            <span className="travel-kind">Scale</span>
          </div>
          <ul className="lab-sliders travel-readout">
            <RangeRow
              label="Size pad"
              title="Scale room around the character. Bigger pad = more room before a walk is pushed back."
              value={draft.pad_px}
              min={0}
              max={PAD_MAX}
              step={1}
              disabled={masterOff}
              text={`${draft.pad_px}px`}
              onChange={(n) => commit({ ...draft, pad_px: clampPad(n) })}
            />
          </ul>
        </div>

        <div className="travel-block">
          <div className="travel-block-head">
            <h3
              className="group-subtitle"
              title="Face mesh only (jaw, brows, eyes, nose). Hair stays in Size."
            >
              Head
            </h3>
            <span className="travel-kind">Form</span>
            <Toggle
              label="On"
              checked={draft.side}
              disabled={masterOff}
              title="Pink overlay around the face mesh (jaw, brows, eyes, nose)."
              onChange={(side) => commit({ ...draft, side }, true)}
            />
          </div>
          <ul className="lab-sliders travel-readout">
            <RangeRow
              label="Left"
              value={draft.left}
              disabled={masterOff || !draft.side}
              title="Slide the head overlay left to this max."
              kind="room"
              onChange={(left) => commit({ ...draft, left })}
            />
            <RangeRow
              label="Right"
              value={draft.right}
              disabled={masterOff || !draft.side}
              title="Slide the head overlay right to this max."
              kind="room"
              onChange={(right) => commit({ ...draft, right })}
            />
          </ul>
        </div>

        <div className="travel-block">
          <div className="travel-block-head">
            <h3
              className="group-subtitle"
              title="Nod-up and nod-down caps. The head overlay nods to the angle as you drag."
            >
              Look
            </h3>
            <span className="travel-kind">Rotation</span>
          </div>
          <div className="travel-inline-toggles">
            <Toggle
              label="Look up"
              checked={draft.look_up}
              disabled={masterOff}
              title="Nod-up cap sent to Track Lab."
              onChange={(look_up) => commit({ ...draft, look_up }, true)}
            />
            <Toggle
              label="Look down"
              checked={draft.look_down}
              disabled={masterOff}
              title="Nod-down cap sent to Track Lab."
              onChange={(look_down) => commit({ ...draft, look_down }, true)}
            />
          </div>
          <ul className="lab-sliders travel-readout">
            <RangeRow
              label="Look up"
              value={draft.pitch_up}
              max={PITCH_UP_MAX}
              disabled={masterOff || !draft.look_up}
              title="Head overlay nods up to this angle."
              kind="deg"
              onChange={(pitch_up) => commit({ ...draft, pitch_up })}
            />
            <RangeRow
              label="Look down"
              value={draft.pitch_down}
              max={PITCH_DOWN_MAX}
              disabled={masterOff || !draft.look_down}
              title="Head overlay nods down to this angle."
              kind="deg"
              onChange={(pitch_down) => commit({ ...draft, pitch_down })}
            />
          </ul>
        </div>

        <div className="travel-block">
          <div className="travel-block-head">
            <h3 className="group-subtitle">Eyes</h3>
            <span className="travel-kind">Form</span>
            <Toggle
              label="On"
              checked={draft.eyes}
              disabled={masterOff}
              title="Stop how far the pupils can travel inside each eye."
              onChange={(eyes) => commit({ ...draft, eyes }, true)}
            />
          </div>
          <ul className="lab-sliders travel-readout">
            <RangeRow
              label="Side"
              value={draft.eye_x}
              disabled={masterOff || !draft.eyes}
              title="Pupils slide to this side limit."
              kind="eye"
              onChange={(eye_x) => commit({ ...draft, eye_x })}
            />
            <RangeRow
              label="Up / down"
              value={draft.eye_y}
              disabled={masterOff || !draft.eyes}
              title="Pupils slide to this up / down limit."
              kind="eye"
              onChange={(eye_y) => commit({ ...draft, eye_y })}
            />
          </ul>
        </div>

        <div className="travel-block">
          <div className="travel-block-head">
            <h3 className="group-subtitle">Skeleton</h3>
            <span className="travel-kind">Form</span>
            <Toggle
              label="On"
              checked={draft.body}
              disabled={masterOff}
              title="Stop how far the shoulders, arms, and chest can slide."
              onChange={(body) => commit({ ...draft, body }, true)}
            />
          </div>
          <ul className="lab-sliders travel-readout">
            <RangeRow
              label="Left"
              value={draft.body_left}
              disabled={masterOff || !draft.body}
              title="Slide the skeleton overlay left to this max."
              kind="room"
              onChange={(body_left) => commit({ ...draft, body_left })}
            />
            <RangeRow
              label="Right"
              value={draft.body_right}
              disabled={masterOff || !draft.body}
              title="Slide the skeleton overlay right to this max."
              kind="room"
              onChange={(body_right) => commit({ ...draft, body_right })}
            />
            <RangeRow
              label="Up"
              value={draft.body_up}
              disabled={masterOff || !draft.body}
              title="Slide the skeleton overlay up to this max."
              kind="room"
              onChange={(body_up) => commit({ ...draft, body_up })}
            />
            <RangeRow
              label="Down"
              value={draft.body_down}
              disabled={masterOff || !draft.body}
              title="Slide the skeleton overlay down to this max."
              kind="room"
              onChange={(body_down) => commit({ ...draft, body_down })}
            />
          </ul>
          <div className="travel-block-head">
            <h3 className="group-subtitle">Skel rotate</h3>
            <span className="travel-kind">Rotation</span>
            <Toggle
              label="On"
              checked={draft.body_rotate}
              disabled={masterOff}
              title="Stop how far the torso can turn and the shoulders can tilt."
              onChange={(body_rotate) => commit({ ...draft, body_rotate }, true)}
            />
          </div>
          <ul className="lab-sliders travel-readout">
            <RangeRow
              label="Turn"
              value={draft.body_yaw}
              max={BODY_YAW_MAX}
              disabled={masterOff || !draft.body_rotate}
              title="Skeleton overlay turns to this angle."
              kind="deg"
              onChange={(body_yaw) => commit({ ...draft, body_yaw })}
            />
            <RangeRow
              label="Tilt"
              value={draft.body_roll}
              max={BODY_ROLL_MAX}
              disabled={masterOff || !draft.body_rotate}
              title="Skeleton overlay tilts to this angle."
              kind="deg"
              onChange={(body_roll) => commit({ ...draft, body_roll })}
            />
          </ul>
        </div>
      </div>
    </div>
  )
}

function RangeRow(props: {
  label: string
  value: number
  disabled: boolean
  title?: string
  kind?: 'room' | 'eye' | 'deg'
  min?: number
  max?: number
  step?: number
  text?: string
  onChange: (value: number) => void
}) {
  const kind = props.kind
  const max = props.max ?? (kind === 'eye' ? 1 : kind === 'deg' ? 0 : ROOM_MAX)
  const min = props.min ?? 0
  const step = props.step ?? (kind === 'deg' || props.text?.endsWith('px') ? 1 : 0.01)
  const clamp = (n: number) => {
    if (kind === 'eye') return clampEye(n)
    if (kind === 'deg') return clampDeg(n, max)
    if (props.text?.endsWith('px')) return clampPad(n)
    if (kind === 'room') return clampRoom(n)
    return n
  }
  const text =
    props.text ??
    (kind === 'eye'
      ? `${Math.round(props.value * 100)}%`
      : kind === 'deg'
        ? `${Math.round(props.value)}°`
        : props.value.toFixed(2))
  return (
    <li title={props.title}>
      <span>{props.label}</span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={props.value}
        disabled={props.disabled}
        onChange={(e) => props.onChange(clamp(Number(e.target.value)))}
      />
      <em className="mono">{text}</em>
    </li>
  )
}
