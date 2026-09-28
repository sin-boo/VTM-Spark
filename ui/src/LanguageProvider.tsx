import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react'
import { api } from './api'
import { I18nContext, STORAGE_KEY, firstLang, isLang, makeI18n, paintLang, type Lang } from './i18n'

export function LanguageProvider({ children }: { children: ReactNode }) {
  const [lang, setLangState] = useState<Lang>(firstLang)
  const [picked, setPicked] = useState<boolean | null>(null)

  useEffect(() => {
    paintLang(lang)
  }, [lang])

  useEffect(() => {
    let alive = true
    const ask = (triesLeft: number) => {
      api
        .uiPrefs()
        .then((prefs) => {
          if (!alive) return
          if (isLang(prefs.language)) setLangState(prefs.language)
          // Blank = nobody has chosen yet: the desk opens on the language picker.
          setPicked(isLang(prefs.language))
        })
        .catch((e: unknown) => {
          if (!alive) return
          // Older backend without /api/ui-prefs: never block the desk on a picker.
          if (/404|not found/i.test(String(e)) || triesLeft <= 0) setPicked(true)
          else window.setTimeout(() => ask(triesLeft - 1), 500)
        })
    }
    ask(20)
    return () => {
      alive = false
    }
  }, [])

  const setLang = useCallback(async (next: Lang) => {
    setLangState(next)
    setPicked(true)
    try {
      window.localStorage.setItem(STORAGE_KEY, next)
    } catch {
      /* storage can be off in private mode */
    }
    await api.setUiPrefs({ language: next })
  }, [])

  const value = useMemo(() => makeI18n(lang, setLang, picked), [lang, setLang, picked])
  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>
}
