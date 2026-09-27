import { useEffect, useRef, useState } from 'react'
import {
  characterThumb,
  exportNote,
  type CharacterCard,
  type CharacterExportResult,
  type CharacterInfo,
  type CharacterMeta,
} from '../api'

type Props = {
  card: CharacterCard
  onInfo: (id: string) => Promise<CharacterInfo>
  onMeta: (meta: CharacterMeta) => Promise<CharacterCard>
  onExport: (id: string, name: string) => Promise<CharacterExportResult>
  /** Desk is busy or streaming; pack writes wait like other character actions. */
  busy?: boolean
  onClose: () => void
}

type Draft = { name: string; author: string; license: string; description: string }

function cleanError(e: unknown): string {
  return String(e).replace(/^Error:\s*/, '')
}

function draftOf(info: CharacterInfo | null, card: CharacterCard): Draft {
  return {
    name: info?.name || card.name,
    author: info?.author ?? card.author ?? '',
    license: info?.license ?? '',
    description: info?.description ?? '',
  }
}

function stamp(value: string | number | undefined): string {
  if (value === undefined || value === null || value === '') return '—'
  const when = typeof value === 'number' ? new Date(value < 1e12 ? value * 1000 : value) : new Date(value)
  return Number.isNaN(when.getTime()) ? String(value) : when.toLocaleString()
}

function bytes(size: number | undefined): string {
  if (!size || size < 0) return '—'
  if (size < 1024) return `${size} B`
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`
  return `${(size / (1024 * 1024)).toFixed(1)} MB`
}

function yes(on: boolean | undefined): string {
  return on ? 'Yes' : 'No'
}

export function CharacterDetails(props: Props) {
  const { card } = props
  const nameRef = useRef<HTMLInputElement>(null)
  const [info, setInfo] = useState<CharacterInfo | null>(null)
  const [loading, setLoading] = useState(true)
  const [draft, setDraft] = useState<Draft>(() => draftOf(null, card))
  const [saving, setSaving] = useState(false)
  const [exporting, setExporting] = useState(false)
  const [error, setError] = useState('')
  const [note, setNote] = useState('')
  const [reload, setReload] = useState(0)
  const busy = Boolean(props.busy)
  const frozen = !info || saving
  const saved = draftOf(info, card)
  const dirty =
    draft.name.trim() !== saved.name ||
    draft.author.trim() !== saved.author ||
    draft.license.trim() !== saved.license ||
    draft.description.trim() !== saved.description
  const model = info?.model
  const includes = info?.includes

  useEffect(() => {
    let alive = true
    setLoading(true)
    setError('')
    setNote('')
    props
      .onInfo(card.id)
      .then((next) => {
        if (!alive) return
        setInfo(next)
        setDraft(draftOf(next, card))
      })
      .catch((e) => {
        if (alive) setError(cleanError(e))
      })
      .finally(() => {
        if (alive) setLoading(false)
      })
    return () => {
      alive = false
    }
    // Reload only when a different character is opened (or on Retry).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [card.id, reload])

  useEffect(() => {
    nameRef.current?.focus()
  }, [])

  function edit(key: keyof Draft, value: string) {
    setNote('')
    setDraft((cur) => ({ ...cur, [key]: value }))
  }

  async function save() {
    const name = draft.name.trim()
    if (!name || saving || busy || !info || !dirty) return
    const sent = {
      name,
      author: draft.author.trim(),
      license: draft.license.trim(),
      description: draft.description.trim(),
    }
    setSaving(true)
    setError('')
    setNote('')
    try {
      await props.onMeta({ id: card.id, ...sent })
    } catch (e) {
      setError(`Save failed: ${cleanError(e)}`)
      setSaving(false)
      return
    }
    // The pack is written; a failed re-read should not look like a failed save.
    try {
      const next = await props.onInfo(card.id)
      setInfo(next)
      setDraft(draftOf(next, card))
    } catch {
      setInfo((cur) => (cur ? { ...cur, ...sent } : cur))
      setDraft(sent)
    }
    setNote('Saved.')
    setSaving(false)
  }

  async function exportPack() {
    if (exporting || busy) return
    const name = info?.name || card.name
    setExporting(true)
    setError('')
    setNote('')
    try {
      const res = await props.onExport(card.id, name)
      setNote(exportNote(name, res))
    } catch (e) {
      setError(`Export failed: ${cleanError(e)}`)
    } finally {
      setExporting(false)
    }
  }

  return (
    <div className="char-details" role="dialog" aria-labelledby="char-details-title">
      <header className="char-create-head">
        <h2 id="char-details-title" className="char-sheet-title">
          Details · {info?.name || card.name}
        </h2>
        <button type="button" className="btn ghost" onClick={props.onClose}>
          Close
        </button>
      </header>
      <div className="char-details-main">
        <div className="char-create-still" aria-hidden="true">
          <img src={characterThumb(card)} alt="" />
        </div>
        <div className="char-details-fields">
          <label className="field">
            <span>Name</span>
            <input
              ref={nameRef}
              value={draft.name}
              maxLength={80}
              disabled={frozen}
              onChange={(e) => edit('name', e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  e.preventDefault()
                  void save()
                }
              }}
            />
          </label>
          <label className="field">
            <span>Author</span>
            <input
              value={draft.author}
              maxLength={120}
              placeholder="Who made this character"
              disabled={frozen}
              onChange={(e) => edit('author', e.target.value)}
            />
          </label>
          <label className="field">
            <span>License</span>
            <input
              value={draft.license}
              maxLength={120}
              placeholder="e.g. Personal use only"
              disabled={frozen}
              onChange={(e) => edit('license', e.target.value)}
            />
          </label>
          <label className="field">
            <span>Description</span>
            <textarea
              value={draft.description}
              maxLength={2000}
              rows={3}
              disabled={frozen}
              onChange={(e) => edit('description', e.target.value)}
            />
          </label>
        </div>
      </div>

      {info && !info.model_match ? (
        <p className="char-details-warn" role="note">
          Made with a different model. It will be re-encoded for this model on first load.
          {model?.checkpoint ? <span className="mono"> ({model.checkpoint})</span> : null}
        </p>
      ) : null}

      {loading ? (
        <p className="hint">Reading character…</p>
      ) : !info ? (
        <div className="char-details-empty">
          <p className="hint">Details are not available for this character.</p>
          <button type="button" className="btn ghost" onClick={() => setReload((n) => n + 1)}>
            Retry
          </button>
        </div>
      ) : (
        <div className="char-details-facts">
          <section>
            <h3 className="group-subtitle">Made with</h3>
            <dl>
              <dt>Model</dt>
              <dd title={model?.checkpoint || ''}>{model?.checkpoint || '—'}</dd>
              <dt>Image</dt>
              <dd>{model?.image_size ? `${model.image_size} px` : '—'}</dd>
              <dt>Latent</dt>
              <dd className="mono">
                {model?.latent_shape?.length ? model.latent_shape.join(' × ') : '—'}
              </dd>
              <dt>This desk</dt>
              <dd>{info.model_match ? 'Same model' : 'Re-encode on load'}</dd>
            </dl>
          </section>
          <section>
            <h3 className="group-subtitle">Includes</h3>
            <dl>
              <dt>Source image</dt>
              <dd>{yes(includes?.source_image)}</dd>
              <dt>Pose keys</dt>
              <dd>{includes?.pose_keys ? includes.pose_keys : 'None'}</dd>
              <dt>Blend shapes</dt>
              <dd>{yes(includes?.blendshapes)}</dd>
              <dt>Hair</dt>
              <dd>{yes(includes?.hair)}</dd>
              <dt>Skeleton</dt>
              <dd>{yes(includes?.skeleton)}</dd>
              <dt>Limiters</dt>
              <dd>{yes(includes?.travel_box)}</dd>
            </dl>
          </section>
          <section>
            <h3 className="group-subtitle">File</h3>
            <dl>
              <dt>Format</dt>
              <dd>{info.version ? `v${info.version}` : '—'}</dd>
              <dt>Size</dt>
              <dd>{bytes(info.size_bytes)}</dd>
              <dt>Created</dt>
              <dd>{stamp(info.created_at)}</dd>
              <dt>Updated</dt>
              <dd>{stamp(info.updated_at)}</dd>
            </dl>
          </section>
        </div>
      )}

      {error ? <p className="status-error">{error}</p> : null}
      {note && !error ? (
        <p className="hint" role="status">
          {note}
        </p>
      ) : null}
      {busy && info ? <p className="hint">The desk is busy. Stop streaming to save or export.</p> : null}

      <div className="char-details-actions">
        <button
          type="button"
          className="btn primary"
          disabled={!info || busy || saving || !dirty || !draft.name.trim()}
          onClick={() => void save()}
        >
          {saving ? 'Saving…' : 'Save'}
        </button>
        <button
          type="button"
          className="btn"
          disabled={busy || exporting || loading}
          onClick={() => void exportPack()}
        >
          {exporting ? 'Exporting…' : 'Export .vtm'}
        </button>
      </div>
    </div>
  )
}
