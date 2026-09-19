import { useEffect, useRef, useState, type MouseEvent as ReactMouseEvent } from 'react'
import { createPortal } from 'react-dom'
import type { CharacterCard, CharacterLoadResult } from '../api'
import { ProgressMeter } from './widgets'

type Props = {
  currentId: string
  characters: CharacterCard[]
  creating: boolean
  createProgress?: number
  createLabel?: string
  createStillUrl?: string
  busy: boolean
  error: string
  onLoad: (
    id: string,
    opts?: { repair?: boolean },
  ) => Promise<CharacterLoadResult | void | undefined>
  onRemove: (id: string) => void
  onRename: (id: string, name: string) => void
  onCreate: (file: File) => Promise<CharacterCard | void | undefined>
}

type CtxMenu = { id: string; x: number; y: number }

export function CharacterLibrary(props: Props) {
  const createRef = useRef<HTMLInputElement>(null)
  const renameRef = useRef<HTMLInputElement>(null)
  const nameRef = useRef<HTMLInputElement>(null)
  const menuRef = useRef<HTMLDivElement>(null)
  const [libraryOpen, setLibraryOpen] = useState(false)
  const [pickedId, setPickedId] = useState<string | null>(null)
  const [confirmId, setConfirmId] = useState<string | null>(null)
  const [repairId, setRepairId] = useState<string | null>(null)
  const [renameId, setRenameId] = useState<string | null>(null)
  const [renameDraft, setRenameDraft] = useState('')
  const [nameId, setNameId] = useState<string | null>(null)
  const [nameDraft, setNameDraft] = useState('')
  const [localStill, setLocalStill] = useState('')
  const [menu, setMenu] = useState<CtxMenu | null>(null)
  const [pickError, setPickError] = useState('')
  const locked = props.busy || props.creating
  const creating = props.creating
  const current = props.characters.find((c) => c.id === props.currentId)
  const confirmCard = props.characters.find((c) => c.id === confirmId)
  const repairCard = props.characters.find((c) => c.id === repairId)
  const renameCard = props.characters.find((c) => c.id === renameId)
  const menuCard = props.characters.find((c) => c.id === menu?.id)
  const nameCard = props.characters.find((c) => c.id === nameId)
  const stillSrc = localStill || props.createStillUrl || nameCard?.preview_url || ''
  const previewSrc = current?.preview_url || (creating ? stillSrc : '')
  const createOpen = creating || Boolean(localStill) || Boolean(nameId)
  const notice = pickError || (createOpen ? props.error : '')

  function clearOverlays() {
    setConfirmId(null)
    setRepairId(null)
    setRenameId(null)
    setMenu(null)
    setPickError('')
  }

  function closeLibrary() {
    setLibraryOpen(false)
    setPickedId(null)
    clearOverlays()
  }

  function closeCreate() {
    if (creating) return
    setNameId(null)
    setNameDraft('')
    setPickError('')
    if (localStill) URL.revokeObjectURL(localStill)
    setLocalStill('')
  }

  useEffect(() => {
    function onKey(ev: KeyboardEvent) {
      if (ev.key !== 'Escape') return
      if (menu || renameId || confirmId || repairId) {
        clearOverlays()
        return
      }
      if (createOpen) {
        closeCreate()
        return
      }
      if (libraryOpen) closeLibrary()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [menu, renameId, confirmId, repairId, creating, nameId, libraryOpen, createOpen])

  useEffect(() => {
    if (!menu) return
    function onDoc(ev: MouseEvent) {
      if (menuRef.current?.contains(ev.target as Node)) return
      setMenu(null)
    }
    document.addEventListener('mousedown', onDoc)
    return () => document.removeEventListener('mousedown', onDoc)
  }, [menu])

  useEffect(() => {
    if (renameId) renameRef.current?.focus()
  }, [renameId])

  useEffect(() => {
    if (nameId && !creating) nameRef.current?.focus()
  }, [nameId, creating])

  function placeMenu(card: CharacterCard, x: number, y: number) {
    const width = 168
    const height = 88
    setConfirmId(null)
    setRenameId(null)
    setMenu({
      id: card.id,
      x: Math.min(Math.max(8, x), window.innerWidth - width - 8),
      y: Math.min(Math.max(8, y), window.innerHeight - height - 8),
    })
  }

  function openMenu(card: CharacterCard, ev: ReactMouseEvent) {
    ev.preventDefault()
    ev.stopPropagation()
    setPickedId(card.id)
    placeMenu(card, ev.clientX, ev.clientY)
  }

  function openMenuFromButton(card: CharacterCard, ev: ReactMouseEvent<HTMLButtonElement>) {
    ev.preventDefault()
    ev.stopPropagation()
    const box = ev.currentTarget.getBoundingClientRect()
    setPickedId(card.id)
    placeMenu(card, box.right - 168, box.bottom + 4)
  }

  function pickFile() {
    if (locked) return
    createRef.current?.click()
  }

  function openDock(ev?: ReactMouseEvent) {
    ev?.preventDefault()
    ev?.stopPropagation()
    if (locked) return
    setPickedId(props.currentId || null)
    setLibraryOpen(true)
  }

  async function beginCreate(file: File) {
    setLibraryOpen(false)
    setPickError('')
    setNameId(null)
    const url = URL.createObjectURL(file)
    setLocalStill((cur) => {
      if (cur) URL.revokeObjectURL(cur)
      return url
    })
    try {
      const card = await props.onCreate(file)
      if (card?.id) {
        setNameId(card.id)
        setNameDraft(card.name)
      } else {
        URL.revokeObjectURL(url)
        setLocalStill('')
      }
    } catch (e) {
      setPickError(String(e).replace(/^Error:\s*/, ''))
    }
  }

  function submitDockRename() {
    const name = renameDraft.trim()
    if (!renameId || !name) return
    props.onRename(renameId, name)
    setRenameId(null)
  }

  function submitCreateName() {
    const name = nameDraft.trim()
    if (!nameId || !name) return
    if (name !== (nameCard?.name || '')) props.onRename(nameId, name)
    closeCreate()
  }

  async function loadCard(id: string) {
    if (locked || id === props.currentId) return
    const res = await props.onLoad(id)
    if (res?.incompatible) setRepairId(id)
  }

  async function addCard(id: string) {
    if (!id || locked) return
    if (id !== props.currentId) await loadCard(id)
    closeLibrary()
  }

  async function addPicked() {
    if (pickedId) await addCard(pickedId)
  }

  const sheet = renameCard ? (
    <div className="char-sheet" role="dialog" aria-labelledby="char-rename-title">
      <p id="char-rename-title" className="char-sheet-title">
        Rename {renameCard.name}
      </p>
      <input
        ref={renameRef}
        className="char-rename-input"
        value={renameDraft}
        maxLength={80}
        onChange={(e) => setRenameDraft(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter') {
            e.preventDefault()
            submitDockRename()
          }
        }}
      />
      <div className="row">
        <button type="button" className="btn primary" disabled={!renameDraft.trim()} onClick={submitDockRename}>
          Save
        </button>
        <button type="button" className="btn ghost" onClick={() => setRenameId(null)}>
          Cancel
        </button>
      </div>
    </div>
  ) : confirmCard ? (
    <div className="char-sheet" role="alertdialog" aria-labelledby="char-confirm-title">
      <p id="char-confirm-title" className="char-sheet-title">
        Remove {confirmCard.name}?
      </p>
      <div className="char-sheet-preview">
        <img src={confirmCard.preview_url} alt="" />
        <p className="hint">
          {confirmCard.id === props.currentId ? 'Off the desk and out of the library.' : 'Out of the library.'}
        </p>
      </div>
      <div className="row">
        <button
          type="button"
          className="btn danger"
          onClick={() => {
            const id = confirmCard.id
            setConfirmId(null)
            if (pickedId === id) setPickedId(null)
            props.onRemove(id)
          }}
        >
          Remove
        </button>
        <button type="button" className="btn ghost" onClick={() => setConfirmId(null)}>
          Keep
        </button>
      </div>
    </div>
  ) : repairCard ? (
    <div className="char-sheet" role="alertdialog" aria-labelledby="char-repair-title">
      <p id="char-repair-title" className="char-sheet-title">
        Repair {repairCard.name}?
      </p>
      <p className="hint">Blend shapes do not match the current plan.</p>
      <div className="row">
        <button
          type="button"
          className="btn primary"
          disabled={locked}
          onClick={() => {
            void (async () => {
              const res = await props.onLoad(repairCard.id, { repair: true })
              if (res?.incompatible || !res?.ok) return
              setRepairId(null)
            })()
          }}
        >
          Repair
        </button>
        <button type="button" className="btn ghost" onClick={() => setRepairId(null)}>
          Not now
        </button>
      </div>
    </div>
  ) : null

  return (
    <>
      <button
        type="button"
        className={`char-preview${current ? ' is-on' : ' is-empty'}`}
        disabled={locked}
        aria-label={current ? `${current.name}. Open characters` : 'No character. Click to add'}
        onClick={openDock}
      >
        {previewSrc ? (
          <img src={previewSrc} alt="" />
        ) : (
          <span className="char-preview-empty">
            <span>No character</span>
            <span>Click to add</span>
          </span>
        )}
      </button>
      <input
        ref={createRef}
        type="file"
        accept="image/*"
        className="file-input-hidden"
        disabled={locked}
        onChange={(e) => {
          const f = e.target.files?.[0]
          e.target.value = ''
          if (!f) return
          void beginCreate(f)
        }}
      />

      {libraryOpen
        ? createPortal(
            <div
              className="char-modal-back"
              role="presentation"
              onClick={(e) => {
                if (e.target === e.currentTarget) closeLibrary()
              }}
            >
              <div className="char-modal" role="dialog" aria-labelledby="char-library-title">
                <header className="char-modal-head">
                  <h2 id="char-library-title" className="char-modal-title">
                    Characters
                  </h2>
                  <button type="button" className="btn ghost" onClick={closeLibrary}>
                    Close
                  </button>
                </header>
                <div className={props.characters.length ? 'char-grid' : 'char-grid is-empty'}>
                  {props.characters.length ? (
                    props.characters.map((card) => (
                    <article
                      key={card.id}
                      className={`char-card${card.id === props.currentId ? ' is-on' : ''}${
                        card.id === pickedId ? ' is-picked' : ''
                      }${menu?.id === card.id ? ' is-menu' : ''}${card.id === confirmId ? ' is-remove' : ''}`}
                      onContextMenu={(e) => openMenu(card, e)}
                    >
                      <button
                        type="button"
                        className="char-card-hit"
                        disabled={locked}
                        aria-pressed={card.id === pickedId}
                        title={card.name}
                        onClick={() => setPickedId(card.id)}
                        onDoubleClick={() => void addCard(card.id)}
                        onContextMenu={(e) => openMenu(card, e)}
                      >
                        <img src={card.preview_url} alt="" />
                        <span className="char-card-name">{card.name}</span>
                        {card.shapes_compatible === false ? (
                          <span className="char-card-warn">Incompatible</span>
                        ) : null}
                      </button>
                      <button
                        type="button"
                        className="char-card-more"
                        aria-label={`Actions for ${card.name}`}
                        aria-haspopup="menu"
                        aria-expanded={menu?.id === card.id}
                        disabled={locked}
                        onClick={(e) => openMenuFromButton(card, e)}
                        onContextMenu={(e) => openMenu(card, e)}
                      >
                        ⋯
                      </button>
                    </article>
                    ))
                  ) : (
                    <div className="char-empty-well" aria-hidden="true">
                      <span className="char-preview-empty">
                        <span>No character</span>
                        <span>Click to add</span>
                      </span>
                    </div>
                  )}
                </div>
                <div className="char-modal-actions">
                  <button type="button" className="btn" disabled={locked} onClick={pickFile}>
                    Create
                  </button>
                  <button
                    type="button"
                    className="btn primary"
                    disabled={locked || !pickedId}
                    onClick={() => void addPicked()}
                  >
                    Add
                  </button>
                </div>
              </div>
            </div>,
            document.body,
          )
        : null}

      {createOpen
        ? createPortal(
            <div className="char-modal-back" role="presentation">
              <div className="char-create" role="dialog" aria-labelledby="char-create-title">
                <div className="char-create-still" aria-hidden="true">
                  {stillSrc ? <img src={stillSrc} alt="" /> : null}
                </div>
                <div className="char-create-body">
                  <header className="char-create-head">
                    <h2 id="char-create-title" className="char-sheet-title">
                      {creating ? 'Creating character' : 'Name character'}
                    </h2>
                    <button type="button" className="btn ghost" disabled={creating} onClick={closeCreate}>
                      Close
                    </button>
                  </header>
                  {creating ? (
                    <ProgressMeter label={props.createLabel || 'Creating character…'} value={props.createProgress ?? 0} />
                  ) : (
                    <>
                      <input
                        ref={nameRef}
                        className="char-rename-input"
                        value={nameDraft}
                        maxLength={80}
                        placeholder="Name"
                        onChange={(e) => setNameDraft(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') {
                            e.preventDefault()
                            submitCreateName()
                          }
                        }}
                      />
                      <div className="row">
                        <button
                          type="button"
                          className="btn primary"
                          disabled={!nameDraft.trim()}
                          onClick={submitCreateName}
                        >
                          Save
                        </button>
                      </div>
                    </>
                  )}
                  {notice ? <p className="status-error">{notice}</p> : null}
                </div>
              </div>
            </div>,
            document.body,
          )
        : null}

      {sheet
        ? createPortal(
            <div
              className="char-modal-back"
              role="presentation"
              onClick={(e) => {
                if (e.target === e.currentTarget) clearOverlays()
              }}
            >
              {sheet}
            </div>,
            document.body,
          )
        : null}

      {menu && menuCard
        ? createPortal(
            <div
              ref={menuRef}
              className="char-ctx"
              role="menu"
              aria-label={`${menuCard.name} actions`}
              style={{ left: menu.x, top: menu.y }}
              onClick={(e) => e.stopPropagation()}
              onContextMenu={(e) => e.preventDefault()}
            >
              <button
                type="button"
                className="menu-item"
                role="menuitem"
                onClick={() => {
                  setRenameDraft(menuCard.name)
                  setRenameId(menuCard.id)
                  setMenu(null)
                }}
              >
                Rename
              </button>
              <button
                type="button"
                className="menu-item danger"
                role="menuitem"
                onClick={() => {
                  setConfirmId(menuCard.id)
                  setMenu(null)
                }}
              >
                Remove
              </button>
            </div>,
            document.body,
          )
        : null}
    </>
  )
}
