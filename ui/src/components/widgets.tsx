import type { ReactNode } from 'react'

export function ProgressMeter({
  label,
  value,
  children,
}: {
  label: string
  value: number
  children?: ReactNode
}) {
  const pct = Math.max(0, Math.min(100, Math.round(value * 100)))
  return (
    <div className="progress" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
      <div className="progress-meta">
        <span className="progress-label">{label}</span>
        <span className="progress-pct mono">{pct}%</span>
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
