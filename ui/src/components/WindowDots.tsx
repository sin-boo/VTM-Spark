import { useI18n } from '../i18n'
import { nativeWindow } from '../nativeWindow'

function Icon({ kind }: { kind: 'min' | 'max' | 'close' }) {
  if (kind === 'min') {
    return (
      <svg viewBox="0 0 10 10" aria-hidden="true">
        <path d="M1.5 5h7" />
      </svg>
    )
  }
  if (kind === 'max') {
    return (
      <svg viewBox="0 0 10 10" aria-hidden="true">
        <rect x="1.75" y="1.75" width="6.5" height="6.5" fill="none" />
      </svg>
    )
  }
  return (
    <svg viewBox="0 0 10 10" aria-hidden="true">
      <path d="M2 2l6 6M8 2l-6 6" />
    </svg>
  )
}

export function WindowDots() {
  const { t } = useI18n()
  return (
    <nav
      className="win-caption"
      aria-label={t('win.label')}
      onMouseDown={(e) => e.stopPropagation()}
    >
      <button type="button" className="is-min" aria-label={t('win.minimize')} onClick={() => nativeWindow('minimize')}>
        <Icon kind="min" />
      </button>
      <button type="button" className="is-max" aria-label={t('win.maximize')} onClick={() => nativeWindow('toggle_max')}>
        <Icon kind="max" />
      </button>
      <button type="button" className="is-close" aria-label={t('win.close')} onClick={() => nativeWindow('close')}>
        <Icon kind="close" />
      </button>
    </nav>
  )
}
