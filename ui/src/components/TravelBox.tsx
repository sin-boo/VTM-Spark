import { useEffect, useRef, useState } from 'react'
import type { TravelBox as TravelBoxValue } from '../api'
import { useI18n, type MessageKey } from '../i18n'
import { Toggle } from './widgets'

const ROOM_MAX = 1.2
const YAW_MAX = 80
const ROLL_MAX = 80
const PITCH_UP_MAX = 50
const PITCH_DOWN_MAX = 32
const SIZE_MAX = 0.7
const VERSION = 2

const ZERO: TravelBoxValue = {
  version: VERSION,
  enabled: true,
  left: 0.68,
  right: 0.65,
  up: 0.29,
  down: 0.57,
  body_left: 0.54,
  body_right: 0.48,
  body_up: 0.1,
  body_down: 0.12,
  turn_left: 21,
  turn_right: 22,
  tilt_left: 22,
  tilt_right: 12,
  pitch_up: 14,
  pitch_down: 3,
  eye: 0.56,
  size: 0,
}

type RoomKey = 'left' | 'right' | 'up' | 'down' | 'body_left' | 'body_right' | 'body_up' | 'body_down'

const HEAD_ROOM: [RoomKey, MessageKey][] = [
  ['left', 'common.left'],
  ['right', 'common.right'],
  ['up', 'common.up'],
  ['down', 'common.down'],
]

type TurnKey = 'turn_left' | 'turn_right' | 'tilt_left' | 'tilt_right'

// Screen sides of the character: which way the face turns, or the crown tips.
const HEAD_TURN: [TurnKey, MessageKey, MessageKey, number][] = [
  ['turn_left', 'travel.turnLeft', 'travel.turnLeftTitle', YAW_MAX],
  ['turn_right', 'travel.turnRight', 'travel.turnRightTitle', YAW_MAX],
  ['tilt_left', 'travel.tiltLeft', 'travel.tiltLeftTitle', ROLL_MAX],
  ['tilt_right', 'travel.tiltRight', 'travel.tiltRightTitle', ROLL_MAX],
]

const BODY_ROOM: [RoomKey, MessageKey][] = [
  ['body_left', 'common.left'],
  ['body_right', 'common.right'],
  ['body_up', 'common.up'],
  ['body_down', 'common.down'],
]

type Props = {
  value?: TravelBoxValue | null
  disabled?: boolean
  show?: boolean
  onShow?: (on: boolean) => void
  onChange: (patch: TravelBoxValue) => void
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
  if (!Number.isFinite(n)) return ZERO.eye
  return Math.max(0, Math.min(1, n))
}

function clampSize(n: number) {
  if (!Number.isFinite(n)) return ZERO.size
  return Math.max(0, Math.min(SIZE_MAX, n))
}

function fill(raw: TravelBoxValue | null | undefined): TravelBoxValue {
  const fresh = raw?.version === VERSION
  const room = (key: RoomKey) => clampRoom(fresh ? (raw?.[key] ?? ZERO[key]) : ZERO[key])
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
    turn_left: clampDeg(raw?.turn_left ?? raw?.yaw ?? ZERO.turn_left, YAW_MAX),
    turn_right: clampDeg(raw?.turn_right ?? raw?.yaw ?? ZERO.turn_right, YAW_MAX),
    tilt_left: clampDeg(raw?.tilt_left ?? raw?.roll ?? ZERO.tilt_left, ROLL_MAX),
    tilt_right: clampDeg(raw?.tilt_right ?? raw?.roll ?? ZERO.tilt_right, ROLL_MAX),
    pitch_up: clampDeg(raw?.pitch_up ?? ZERO.pitch_up, PITCH_UP_MAX),
    pitch_down: clampDeg(raw?.pitch_down ?? ZERO.pitch_down, PITCH_DOWN_MAX),
    eye: clampEye(raw?.eye ?? ZERO.eye),
    size: clampSize(raw?.size ?? ZERO.size),
  }
}

export function TravelBox(props: Props) {
  const { t } = useI18n()
  const live = fill(props.value)
  const [draft, setDraft] = useState(live)
  const drag = useRef(false)
  const timer = useRef<number | null>(null)
  const liveKey = JSON.stringify(live)

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
      <div className="group-head travel-box-head">
        <h2 className="group-title">{t('travel.title')}</h2>
        <div className="travel-box-actions">
          <button
            type="button"
            className="btn ghost compact"
            disabled={props.disabled}
            onClick={() => commit({ ...ZERO, enabled: draft.enabled }, true)}
            title={t('travel.resetTitle')}
          >
            {t('common.reset')}
          </button>
          <Toggle
            label={t('common.show')}
            checked={Boolean(props.show)}
            disabled={props.disabled}
            title={t('travel.showTitle')}
            onChange={(on) => props.onShow?.(on)}
          />
          <Toggle
            label={t('common.on')}
            checked={draft.enabled}
            disabled={props.disabled}
            onChange={(enabled) => commit({ ...draft, enabled }, true)}
          />
        </div>
      </div>

      <div className="travel-limits">
        <div className="travel-block">
          <div className="travel-block-head">
            <i className="travel-swatch is-head" />
            <h3 className="group-subtitle" title={t('travel.headTitle')}>
              {t('travel.head')}
            </h3>
          </div>
          <ul className="lab-sliders travel-readout">
            {HEAD_ROOM.map(([key, label]) => (
              <RangeRow
                key={key}
                label={t(label)}
                value={draft[key]}
                disabled={off}
                title={t('travel.headMove', { dir: t(label).toLowerCase() })}
                kind="room"
                onChange={(n) => commit({ ...draft, [key]: n })}
              />
            ))}
            {HEAD_TURN.map(([key, label, title, max]) => (
              <RangeRow
                key={key}
                label={t(label)}
                value={draft[key]}
                max={max}
                disabled={off}
                title={t(title)}
                kind="deg"
                onChange={(n) => commit({ ...draft, [key]: n })}
              />
            ))}
          </ul>
        </div>

        <div className="travel-block">
          <div className="travel-block-head">
            <i className="travel-swatch is-body" />
            <h3 className="group-subtitle" title={t('travel.bodyTitle')}>
              {t('travel.body')}
            </h3>
          </div>
          <ul className="lab-sliders travel-readout">
            {BODY_ROOM.map(([key, label]) => (
              <RangeRow
                key={key}
                label={t(label)}
                value={draft[key]}
                disabled={off}
                title={t('travel.bodyMove', { dir: t(label).toLowerCase() })}
                kind="room"
                onChange={(n) => commit({ ...draft, [key]: n })}
              />
            ))}
          </ul>
        </div>

        <div className="travel-block">
          <div className="travel-block-head">
            <h3 className="group-subtitle" title={t('travel.lookTitle')}>
              {t('travel.look')}
            </h3>
          </div>
          <ul className="lab-sliders travel-readout">
            <RangeRow
              label={t('travel.lookUp')}
              value={draft.pitch_up}
              max={PITCH_UP_MAX}
              disabled={off}
              kind="deg"
              title={t('travel.lookUpTitle')}
              onChange={(pitch_up) => commit({ ...draft, pitch_up })}
            />
            <RangeRow
              label={t('travel.lookDown')}
              value={draft.pitch_down}
              max={PITCH_DOWN_MAX}
              disabled={off}
              kind="deg"
              onChange={(pitch_down) => commit({ ...draft, pitch_down })}
            />
            <RangeRow
              label={t('travel.eyes')}
              value={draft.eye}
              disabled={off}
              title={t('travel.eyesTitle')}
              kind="eye"
              onChange={(eye) => commit({ ...draft, eye })}
            />
            <RangeRow
              label={t('travel.size')}
              value={draft.size}
              disabled={off}
              title={t('travel.sizeTitle')}
              kind="size"
              onChange={(size) => commit({ ...draft, size })}
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
    <li title={props.title}>
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
  const { t } = useI18n()
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
        aria-label={t('travel.value', { label: props.label })}
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
