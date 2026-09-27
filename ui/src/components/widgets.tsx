import { useEffect, useRef, useState, type ReactNode } from 'react'

// Shown under a bar once it has run a few seconds, so a long compile or
// model load reads as busy backstage, not frozen. Flavour only: the real
// step stays in the label above.
const BACKSTAGE_LINES = [
  'Waking up the avatar…',
  'Teaching the eyes to blink…',
  'Placing character on layer 1…',
  'Asking chat to hold on…',
  'Ironing the green screen…',
  'Tuning the hair physics…',
  'Rehearsing the intro wave…',
  'Warming up the vocal cords…',
  'Loading kawaii.dll…',
  'Polishing the cat ears…',
  'Rigging the smile…',
  'Setting the lighting to cozy…',
  'Convincing the GPU it is showtime…',
  'Counting eyelashes…',
  'Finding outfit…',
  'Feeding the tracker a snack…',
  'Stretching before stream…',
  'Straightening the hoodie strings…',
  'Practicing the “otsukare”…',
  'Adding sparkles to the highlights…',
  'Checking the mic is not muted…',
  'Hiding the spaghetti code…',
  'Loading today’s catchphrase…',
  'Finding the good camera angle…',
  'Taping down the cables…',
  'Practicing the head tilt…',
  'Warming up the smile…',
  'Checking the stream title for typos…',
  'Picking the thumbnail face…',
  'Rehearsing the “welcome back”…',
  'Dusting off the webcam…',
  'Adjusting the chair height…',
  'Hiding the messy desk…',
  'Loading the idle bounce…',
  'Adding shine to the eyes…',
  'Tuning the blush…',
  'Warming up the mouth shapes…',
  'Double-checking the overlay…',
  'Clearing the throat…',
  'Straightening the headset…',
  'Choosing the opening line…',
  'Filling up the energy bar…',
  'Syncing lips to the voice…',
  'Practicing the surprised face…',
  'Waving at the early viewers…',
  'Adjusting the key light…',
  'Loading the signature pose…',
  'Hanging the stream banner…',
  'Practicing the laugh…',
  'Spinning up the GPU fans…',
  'Getting into character…',
  'Rehearsing the goodbye wave…',
  'Setting the scene…',
]

// Each line stays up for a random 4–6 s, so the changes do not tick like a clock.
const LINE_MIN_MS = 4000
const LINE_MAX_MS = 6000
// Fake creep: while the real value sits still, the bar drifts ahead a little
// (at most CREEP_MAX, easing in over CREEP_TAU_MS) and never past 99 %.
const CREEP_MAX = 0.06
const CREEP_TAU_MS = 9000
// One wait = every bar shown back to back. A stage change can swap or remount
// the bar; only this long with no bar on screen starts a new wait.
const SESSION_GAP_MS = 2000

/** Every line once, in random order, before any repeats. */
function shuffledLines(): string[] {
  const out = [...BACKSTAGE_LINES]
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[out[i], out[j]] = [out[j], out[i]]
  }
  return out
}

type WaitSession = { started: number; seen: number; lines: string[]; holds: number[] }
let waitSession: WaitSession | null = null

/** The running wait: shared start time (the seconds counter) and line order. */
function currentWait(now: number): WaitSession {
  if (waitSession === null || now - waitSession.seen > SESSION_GAP_MS) {
    const lines = shuffledLines()
    const holds = lines.map(() => LINE_MIN_MS + Math.random() * (LINE_MAX_MS - LINE_MIN_MS))
    waitSession = { started: now, seen: now, lines, holds }
  }
  waitSession.seen = now
  return waitSession
}

/** The line showing ``elapsed`` ms into a wait (loops after the last one). */
function lineAt(wait: WaitSession, elapsed: number): string {
  const total = wait.holds.reduce((sum, ms) => sum + ms, 0)
  let t = total > 0 ? elapsed % total : 0
  for (let i = 0; i < wait.lines.length; i++) {
    if (t < wait.holds[i]) return wait.lines[i]
    t -= wait.holds[i]
  }
  return wait.lines[0]
}

export function ProgressMeter({
  label,
  value,
  children,
}: {
  /** The real step. Shown on hover; the bar itself shows a backstage line. */
  label: string
  value: number
  children?: ReactNode
}) {
  // Never step back, count the whole wait, and keep something moving so a
  // long compile never looks frozen.
  const [now, setNow] = useState(() => Date.now())
  const real = useRef({ value: 0, since: now })
  const shown = useRef(0)
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 250)
    return () => window.clearInterval(id)
  }, [])
  const wait = currentWait(now)
  if (value > real.current.value + 1e-4) {
    real.current = { value, since: Date.now() }
  }
  const idle = Math.max(0, now - real.current.since)
  const creep = CREEP_MAX * (1 - Math.exp(-idle / CREEP_TAU_MS))
  const target = value >= 1 ? 1 : Math.min(0.99, real.current.value + creep)
  shown.current = Math.max(shown.current, target)
  const pct = Math.max(0, Math.min(100, Math.round(shown.current * 100)))
  const elapsed = Math.max(0, now - wait.started)
  const secs = Math.floor(elapsed / 1000)
  const line = lineAt(wait, elapsed)
  return (
    <div
      className="progress"
      role="progressbar"
      aria-valuenow={pct}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-label={label}
      title={label}
    >
      <div className="progress-meta">
        <span key={line} className="progress-label progress-quip">
          {line}
        </span>
        <span className="progress-pct mono">
          {secs >= 3 ? <span className="progress-secs">{secs}s</span> : null}
          {pct}%
        </span>
      </div>
      <div className="progress-track">
        <div className="progress-fill" style={{ width: `${pct}%` }} />
      </div>
      {children}
    </div>
  )
}

export function Lamp({
  on,
  pending,
  idle,
  label,
}: {
  on: boolean
  pending?: boolean
  idle?: boolean
  label: string
}) {
  const kind = pending ? 'is-pending' : on ? 'is-on' : idle ? 'is-idle' : 'is-off'
  const copy = pending ? 'hold' : on ? 'live' : idle ? 'ok' : 'off'
  return (
    <span className={`lamp ${kind}`} title={label} aria-label={label}>
      {copy}
    </span>
  )
}

export function Toggle({
  label,
  checked,
  onChange,
  disabled,
  light,
  lightTitle,
  title,
  className,
}: {
  label: string
  checked: boolean
  onChange: (v: boolean) => void
  disabled?: boolean
  light?: 'on' | 'off' | 'pending' | 'fail'
  lightTitle?: string
  title?: string
  className?: string
}) {
  return (
    <label
      className={['toggle', checked ? 'is-on' : '', disabled ? 'is-disabled' : '', className ?? '']
        .filter(Boolean)
        .join(' ')}
      title={title}
    >
      <input
        type="checkbox"
        checked={checked}
        disabled={disabled}
        onChange={(e) => onChange(e.target.checked)}
      />
      <span className="toggle-copy">{label}</span>
      {light ? (
        <span
          className={`signal-light is-${light}`}
          title={lightTitle || undefined}
          aria-label={lightTitle || `status ${light}`}
        />
      ) : null}
    </label>
  )
}
