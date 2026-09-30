import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './styles/tokens.css'
import App from './App.tsx'
import { LanguageProvider } from './LanguageProvider'

// A file dropped outside a drop zone would make WebView2 open it in place of the app.
for (const type of ['dragover', 'drop'] as const) {
  window.addEventListener(type, (ev) => {
    if (ev.defaultPrevented || !ev.dataTransfer?.types.includes('Files')) return
    ev.preventDefault()
    ev.dataTransfer.dropEffect = 'none'
  })
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <LanguageProvider>
      <App />
    </LanguageProvider>
  </StrictMode>,
)
