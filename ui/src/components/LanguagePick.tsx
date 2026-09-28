import { useEffect, useState } from 'react'
import { LANGUAGES, messageIn, useI18n, type Lang } from '../i18n'
import { dragIfPrimary } from '../nativeWindow'
import { WindowDots } from './WindowDots'

/**
 * First run only: once the splash has loaded everything, ask for a language
 * before the desk opens. Every line shows in both languages. The OS language
 * is the highlighted choice.
 */
export function LanguagePick() {
  const { lang, setLang } = useI18n()
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    const root = document.documentElement
    root.setAttribute('data-splash', '')
    return () => root.removeAttribute('data-splash')
  }, [])

  function choose(next: Lang) {
    if (busy) return
    setBusy(true)
    // The desk opens either way; a failed save only means we ask again next launch.
    void setLang(next).catch(() => undefined)
  }

  return (
    <div className="krita-splash lang-pick" onMouseDown={(e) => dragIfPrimary(e.button)}>
      <div className="lang-pick-caption" onMouseDown={(e) => e.stopPropagation()}>
        <WindowDots />
      </div>
      <div className="krita-splash-art" aria-hidden="true" />
      <aside className="krita-splash-panel">
        <div className="krita-splash-plate">
          <header className="krita-splash-brand">
            <h1>
              <img className="krita-splash-logo" src="/vtm-spark-logo.svg" width={176} height={60} alt="VTM Spark" />
            </h1>
          </header>
          <div className="lang-pick-body" role="radiogroup" aria-labelledby="lang-pick-title">
            <h2 id="lang-pick-title" className="lang-pick-title">
              <span lang="en">{messageIn('en', 'pick.title')}</span>
              <span lang="ja">{messageIn('ja', 'pick.title')}</span>
            </h2>
            <div className="lang-pick-options" onMouseDown={(e) => e.stopPropagation()}>
              {LANGUAGES.map((row) => (
                <button
                  key={row.id}
                  type="button"
                  role="radio"
                  lang={row.id}
                  aria-checked={false}
                  className={`btn lang-pick-option${row.id === lang ? ' primary' : ''}`}
                  disabled={busy}
                  autoFocus={row.id === lang}
                  onClick={() => choose(row.id)}
                >
                  {row.label}
                </button>
              ))}
            </div>
            <p className="lang-pick-hint">
              <span lang="en">{messageIn('en', 'pick.hint')}</span>
              <span lang="ja">{messageIn('ja', 'pick.hint')}</span>
            </p>
          </div>
        </div>
      </aside>
    </div>
  )
}
