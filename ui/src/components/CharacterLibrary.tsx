import {
  useEffect,
  useRef,
  useState,
  type DragEvent as ReactDragEvent,
  type MouseEvent as ReactMouseEvent,
} from 'react'
import { createPortal } from 'react-dom'
import {
  characterThumb,
  type CharacterCard,
  type CharacterExportResult,
  type CharacterInfo,
  type CharacterLoadResult,
  type CharacterMeta,
  type TravelBox,
} from '../api'
import { useI18n } from '../i18n'
import { CharacterDetails } from './CharacterDetails'
import { CharacterFit, type CharacterFitHandle } from './CharacterFit'
import { ProgressMeter } from './widgets'

/** Details / sharing panel is asleep: kept in the code, hidden from the menu. Set true to bring it back. */
const SHOW_CHARACTER_DETAILS = false

type Props = {
  currentId: string
  characters: CharacterCard[]
  creating: boolean
  createProgress?: number
  createLabel?: string
  /** Model load and character build are separate bars; a new phase starts from zero. */
  createPhase?: 'model' | 'character'
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
  onImport: (file: File) => Promise<CharacterCard>
  onExport: (id: string, name: string) => Promise<CharacterExportResult>
  onReveal: (id: string) => Promise<unknown>
  onInfo: (id: string) => Promise<CharacterInfo>
  onMeta: (meta: CharacterMeta) => Promise<CharacterCard>
  onRefresh?: () => void
  travel?: TravelBox | null
  onTravel: (box: TravelBox) => Promise<boolean | undefined>
}

type CtxMenu = { id: string; x: number; y: number }

type LibNotice = { error: boolean; text: string }

/** Where files from Explorer can land: the rail preview or the open library. */
type DropZone = 'rail' | 'library'

function cleanError(e: unknown): string {
  return String(e).replace(/^Error:\s*/, '')
}

function hasFiles(ev: ReactDragEvent): boolean {
  return Array.from(ev.dataTransfer.types).includes('Files')
}

function isPack(file: File): boolean {
  return /\.vtm$/i.test(file.name)
}

/** Stills a character can be made from. Keep in step with stage_create_still (backend/character_pack.py). */
const STILL_EXTS = ['.png', '.jpg', '.jpeg', '.webp', '.bmp']
const STILL_TYPES = ['image/png', 'image/jpeg', 'image/webp', 'image/bmp']

function isStill(file: File): boolean {
  const name = file.name.toLowerCase()
  return STILL_TYPES.includes(file.type) || STILL_EXTS.some((ext) => name.endsWith(ext))
}

/** Grid thumbnail that drops back to the full preview if /thumb fails. */
function CardThumb({ card }: { card: CharacterCard }) {
  const [failed, setFailed] = useState(false)
  const src = failed ? card.preview_url : characterThumb(card)
  return (
    <img
      src={src}
      alt=""
      loading="lazy"
      onError={() => {
        if (!failed && src !== card.preview_url) setFailed(true)
      }}
    />
  )
}

export function CharacterLibrary(props: Props) {
  const { t, tr } = useI18n()
  const createRef = useRef<HTMLInputElement>(null)
  const importRef = useRef<HTMLInputElement>(null)
  const gridRef = useRef<HTMLDivElement>(null)
  const renameRef = useRef<HTMLInputElement>(null)
  const nameRef = useRef<HTMLInputElement>(null)
  const fitRef = useRef<CharacterFitHandle>(null)
  const menuRef = useRef<HTMLDivElement>(null)
  const chooseGen = useRef(0)
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
  const [createNote, setCreateNote] = useState('')
  const [detailsId, setDetailsId] = useState<string | null>(null)
  const [revealError, setRevealError] = useState('')
  const [importing, setImporting] = useState(false)
  const [libNotice, setLibNotice] = useState<LibNotice | null>(null)
  const [revealId, setRevealId] = useState<string | null>(null)
  const [dropOver, setDropOver] = useState<DropZone | null>(null)
  const locked = props.busy || props.creating
  const creating = props.creating
  const current = props.characters.find((c) => c.id === props.currentId)
  const confirmCard = props.characters.find((c) => c.id === confirmId)
  const repairCard = props.characters.find((c) => c.id === repairId)
  const renameCard = props.characters.find((c) => c.id === renameId)
  const menuCard = props.characters.find((c) => c.id === menu?.id)
  const nameCard = props.characters.find((c) => c.id === nameId)
  const detailsCard = props.characters.find((c) => c.id === detailsId)
  const stillSrc = localStill || props.createStillUrl || nameCard?.preview_url || ''
  const previewSrc = current?.preview_url || (creating ? stillSrc : '')
  const createOpen = creating || Boolean(localStill) || Boolean(nameId)
  const fitting = Boolean(nameId) && !creating
  const notice = pickError || (createOpen ? props.error : '')

  function clearOverlays() {
    setConfirmId(null)
    setRepairId(null)
    setRenameId(null)
    setRevealError('')
    setMenu(null)
    setPickError('')
  }

  function closeLibrary() {
    setLibraryOpen(false)
    setPickedId(null)
    setLibNotice(null)
    clearOverlays()
  }

  function closeCreate() {
    if (creating) return
    setNameId(null)
    setNameDraft('')
    setPickError('')
    setCreateNote('')
    if (localStill) URL.revokeObjectURL(localStill)
    setLocalStill('')
  }

  useEffect(() => {
    function onKey(ev: KeyboardEvent) {
      if (ev.key !== 'Escape') return
      if (menu || renameId || confirmId || repairId || revealError) {
        clearOverlays()
        return
      }
      if (detailsId) {
        setDetailsId(null)
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
  }, [
    menu,
    renameId,
    confirmId,
    repairId,
    revealError,
    detailsId,
    creating,
    nameId,
    libraryOpen,
    createOpen,
    localStill,
  ])

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
    if (!revealId) return
    const cards = gridRef.current?.querySelectorAll<HTMLElement>('[data-id]') ?? []
    const found = Array.from(cards).find((el) => el.dataset.id === revealId)
    if (!found) return
    found.scrollIntoView({ block: 'nearest' })
    setRevealId(null)
  }, [revealId, props.characters])

  useEffect(() => {
    if (fitting) nameRef.current?.focus()
  }, [fitting])

  function placeMenu(card: CharacterCard, x: number, y: number) {
    const width = 168
    const height = (card.shapes_compatible === false ? 148 : 116) + 56
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
    props.onRefresh?.()
  }

  async function beginCreate(file: File, note = '') {
    closeLibrary()
    setPickError('')
    setCreateNote(note)
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

  function pickImport() {
    if (locked || importing) return
    importRef.current?.click()
  }

  /** Picked or dropped files. Each .vtm is copied into the characters folder by the backend. */
  async function beginImport(files: File[]) {
    setLibNotice(null)
    const packs = files.filter(isPack)
    if (!packs.length) {
      setLibNotice({ error: true, text: t('lib.pickVtm') })
      return
    }
    const skipped = files.length - packs.length
    const skipNote = skipped ? ` ${t('lib.skippedNotVtm', { count: skipped })}` : ''

    setImporting(true)
    const added: CharacterCard[] = []
    const failed: string[] = []
    try {
      for (const file of packs) {
        try {
          const card = await props.onImport(file)
          if (card?.id) added.push(card)
        } catch (e) {
          failed.push(packs.length > 1 ? `${file.name}: ${cleanError(e)}` : cleanError(e))
        }
      }
    } finally {
      setImporting(false)
    }
    const last = added[added.length - 1]
    if (last) {
      setPickedId(last.id)
      setRevealId(last.id)
    }
    if (failed.length) {
      setLibNotice({
        error: true,
        text: t('lib.importFailed', { error: failed.join(' · ') }) + skipNote,
      })
    } else if (added.length > 1) {
      setLibNotice({ error: false, text: t('lib.importedMany', { count: added.length }) + skipNote })
    } else if (last) {
      setLibNotice({ error: false, text: t('lib.imported', { name: last.name }) + skipNote })
    }
  }

  /** .vtm files are imported; otherwise the first image becomes a new character. */
  function dropFiles(zone: DropZone, files: File[]) {
    const hasPack = files.some(isPack)
    const still = files.find(isStill)
    if (!hasPack && still) {
      const skipped = files.length - 1
      void beginCreate(still, skipped ? t('lib.skippedOneImage', { count: skipped }) : '')
      return
    }
    // Dropped on the rail: open the library so the new card and notice show.
    if (zone === 'rail') openDock()
    if (!hasPack) {
      setLibNotice({ error: true, text: t('lib.dropUnsupported') })
      return
    }
    void beginImport(files)
  }

  function dropZone(zone: DropZone) {
    const closed = locked || importing
    return {
      onDragOver(ev: ReactDragEvent<HTMLElement>) {
        if (!hasFiles(ev)) return
        ev.preventDefault()
        ev.dataTransfer.dropEffect = closed ? 'none' : 'copy'
        if (!closed) setDropOver(zone)
      },
      onDragLeave(ev: ReactDragEvent<HTMLElement>) {
        if (ev.currentTarget.contains(ev.relatedTarget as Node | null)) return
        setDropOver(null)
      },
      onDrop(ev: ReactDragEvent<HTMLElement>) {
        if (!hasFiles(ev)) return
        ev.preventDefault()
        setDropOver(null)
        if (closed) return
        dropFiles(zone, Array.from(ev.dataTransfer.files))
      },
    }
  }

  async function revealCard(card: CharacterCard) {
    setMenu(null)
    setRevealError('')
    try {
      await props.onReveal(card.id)
    } catch (e) {
      setRevealError(`${card.name}: ${cleanError(e)}`)
    }
  }

  function openDetails(card: CharacterCard) {
    setMenu(null)
    setConfirmId(null)
    setRenameId(null)
    setDetailsId(card.id)
  }

  function submitDockRename() {
    const name = renameDraft.trim()
    if (!renameId || !name) return
    props.onRename(renameId, name)
    setRenameId(null)
  }

  async function submitCreateName() {
    const name = nameDraft.trim()
    if (!nameId || !name) return
    try {
      await fitRef.current?.commit()
    } catch {
      return
    }
    if (name !== (nameCard?.name || '')) props.onRename(nameId, name)
    closeCreate()
  }

  async function beginEdit(card: CharacterCard) {
    setMenu(null)
    setLibraryOpen(false)
    setConfirmId(null)
    setRepairId(null)
    setRenameId(null)
    setPickError('')
    if (card.id !== props.currentId) {
      try {
        const res = await props.onLoad(card.id)
        if (res?.incompatible) {
          setRepairId(card.id)
          return
        }
      } catch (e) {
        setPickError(String(e).replace(/^Error:\s*/, ''))
        return
      }
    }
    setNameDraft(card.name)
    setNameId(card.id)
  }

  async function chooseCard(id: string) {
    if (!id || locked) return
    const gen = ++chooseGen.current
    setPickedId(id)
    setRepairId(null)
    setConfirmId(null)
    if (id === props.currentId) return
    const res = await props.onLoad(id)
    if (gen !== chooseGen.current) return
    if (res?.incompatible) setRepairId(id)
  }

  const sheet = revealError ? (
    <div className="char-sheet" role="alertdialog" aria-labelledby="char-reveal-title">
      <p id="char-reveal-title" className="char-sheet-title">
        {t('lib.revealFailed')}
      </p>
      <p className="status-error">{revealError}</p>
      <div className="row">
        <button type="button" className="btn ghost" onClick={() => setRevealError('')}>
          {t('common.ok')}
        </button>
      </div>
    </div>
  ) : renameCard ? (
    <div className="char-sheet" role="dialog" aria-labelledby="char-rename-title">
      <p id="char-rename-title" className="char-sheet-title">
        {t('lib.renameTitle', { name: renameCard.name })}
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
          {t('common.save')}
        </button>
        <button type="button" className="btn ghost" onClick={() => setRenameId(null)}>
          {t('common.cancel')}
        </button>
      </div>
    </div>
  ) : confirmCard ? (
    <div className="char-sheet" role="alertdialog" aria-labelledby="char-confirm-title">
      <p id="char-confirm-title" className="char-sheet-title">
        {t('lib.removeTitle', { name: confirmCard.name })}
      </p>
      <div className="char-sheet-preview">
        <img src={characterThumb(confirmCard)} alt="" />
        <p className="hint">
          {confirmCard.id === props.currentId ? t('lib.removeCurrent') : t('lib.removeOther')}
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
          {t('common.remove')}
        </button>
        <button type="button" className="btn ghost" onClick={() => setConfirmId(null)}>
          {t('lib.keep')}
        </button>
      </div>
    </div>
  ) : repairCard ? (
    <div className="char-sheet" role="alertdialog" aria-labelledby="char-repair-title">
      <p id="char-repair-title" className="char-sheet-title">
        {t('lib.repairTitle', { name: repairCard.name })}
      </p>
      <p className="hint">{t('lib.repairHint')}</p>
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
          {t('common.repair')}
        </button>
        <button type="button" className="btn ghost" onClick={() => setRepairId(null)}>
          {t('lib.notNow')}
        </button>
      </div>
    </div>
  ) : null

  return (
    <>
      <button
        type="button"
        className={`char-preview${current ? ' is-on' : ' is-empty'}${dropOver === 'rail' ? ' is-drop' : ''}`}
        disabled={locked}
        aria-label={current ? t('lib.openCharacters', { name: current.name }) : t('lib.emptyAria')}
        onClick={openDock}
        onContextMenu={(e) => {
          if (!current) return
          openMenu(current, e)
        }}
        {...dropZone('rail')}
      >
        {previewSrc ? (
          <img src={previewSrc} alt="" />
        ) : (
          <span className="char-preview-empty">
            <span>{t('lib.noCharacter')}</span>
            <span>{t('lib.clickToAdd')}</span>
          </span>
        )}
        {dropOver === 'rail' ? <span className="char-drop-hint">{t('lib.dropHint')}</span> : null}
      </button>
      <input
        ref={createRef}
        type="file"
        accept={[...STILL_EXTS, ...STILL_TYPES].join(',')}
        className="file-input-hidden"
        disabled={locked}
        onChange={(e) => {
          const f = e.target.files?.[0]
          e.target.value = ''
          if (!f) return
          void beginCreate(f)
        }}
      />
      <input
        ref={importRef}
        type="file"
        accept=".vtm"
        multiple
        className="file-input-hidden"
        disabled={locked || importing}
        onChange={(e) => {
          const files = Array.from(e.target.files ?? [])
          e.target.value = ''
          if (!files.length) return
          void beginImport(files)
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
              {...dropZone('library')}
            >
              <div
                className={`char-modal${dropOver === 'library' ? ' is-drop' : ''}`}
                role="dialog"
                aria-labelledby="char-library-title"
              >
                {dropOver === 'library' ? (
                  <div className="char-drop-hint">{t('lib.dropHint')}</div>
                ) : null}
                <header className="char-modal-head">
                  <h2 id="char-library-title" className="char-modal-title">
                    {t('lib.characters')}
                  </h2>
                  <button type="button" className="btn ghost" onClick={closeLibrary}>
                    {t('common.close')}
                  </button>
                </header>
                <div
                  ref={gridRef}
                  className={props.characters.length ? 'char-grid' : 'char-grid is-empty'}
                >
                  {props.characters.length ? (
                    props.characters.map((card) => (
                    <article
                      key={card.id}
                      className={`char-card${
                        card.id === (pickedId || props.currentId) ? ' is-on' : ''
                      }${menu?.id === card.id ? ' is-menu' : ''}${card.id === confirmId ? ' is-remove' : ''}`}
                      data-id={card.id}
                      onContextMenu={(e) => openMenu(card, e)}
                    >
                      <button
                        type="button"
                        className="char-card-hit"
                        disabled={locked}
                        aria-pressed={card.id === (pickedId || props.currentId)}
                        title={card.name}
                        onClick={() => void chooseCard(card.id)}
                        onContextMenu={(e) => openMenu(card, e)}
                      >
                        <CardThumb card={card} />
                        <span className="char-card-name">{card.name}</span>
                        {card.shapes_compatible === false ? (
                          <span className="char-card-warn">{t('lib.incompatible')}</span>
                        ) : null}
                      </button>
                      <button
                        type="button"
                        className="char-card-more"
                        aria-label={t('lib.actionsFor', { name: card.name })}
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
                        <span>{t('lib.noCharacter')}</span>
                        <span>{t('lib.dropHint')}</span>
                      </span>
                    </div>
                  )}
                </div>
                {libNotice ? (
                  <p className={libNotice.error ? 'status-error' : 'hint'} role="status">
                    {libNotice.text}
                  </p>
                ) : null}
                <div className="char-modal-actions">
                  <button type="button" className="btn" disabled={locked} onClick={pickFile}>
                    {t('lib.create')}
                  </button>
                  <button
                    type="button"
                    className="btn"
                    disabled={locked || importing}
                    title={t('lib.importTitle')}
                    onClick={pickImport}
                  >
                    {importing ? t('lib.importing') : t('lib.import')}
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
              <div
                className={`char-create${fitting ? ' is-fit' : ''}`}
                role="dialog"
                aria-labelledby="char-create-title"
              >
                {fitting ? (
                  <>
                    <header className="char-create-head">
                      <h2 id="char-create-title" className="char-sheet-title">
                        {t('lib.fitTitle')}
                      </h2>
                      <button type="button" className="btn ghost" onClick={closeCreate}>
                        {t('common.close')}
                      </button>
                    </header>
                    <CharacterFit
                      ref={fitRef}
                      stillUrl={nameCard?.preview_url || stillSrc}
                      travel={props.travel}
                      onTravel={props.onTravel}
                      onNotice={setPickError}
                    />
                    <div className="fit-save">
                      <input
                        ref={nameRef}
                        className="char-rename-input"
                        value={nameDraft}
                        maxLength={80}
                        placeholder={t('common.name')}
                        onChange={(e) => setNameDraft(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') {
                            e.preventDefault()
                            submitCreateName()
                          }
                        }}
                      />
                      <button
                        type="button"
                        className="btn primary"
                        disabled={!nameDraft.trim()}
                        onClick={submitCreateName}
                      >
                        {t('common.save')}
                      </button>
                    </div>
                    {createNote ? <p className="hint">{createNote}</p> : null}
                    {notice ? <p className="status-error">{notice}</p> : null}
                  </>
                ) : (
                  <>
                    <div className="char-create-still" aria-hidden="true">
                      {stillSrc ? <img src={stillSrc} alt="" /> : null}
                    </div>
                    <div className="char-create-body">
                      <header className="char-create-head">
                        <h2 id="char-create-title" className="char-sheet-title">
                          {t('lib.creatingTitle')}
                        </h2>
                        <button type="button" className="btn ghost" disabled={creating} onClick={closeCreate}>
                          {t('common.close')}
                        </button>
                      </header>
                      <ProgressMeter
                        key={props.createPhase ?? 'character'}
                        label={tr(props.createLabel || 'Creating character…')}
                        value={props.createProgress ?? 0}
                      />
                      {createNote ? <p className="hint">{createNote}</p> : null}
                      {notice ? <p className="status-error">{notice}</p> : null}
                    </div>
                  </>
                )}
              </div>
            </div>,
            document.body,
          )
        : null}

      {detailsCard
        ? createPortal(
            <div
              className="char-modal-back"
              role="presentation"
              onClick={(e) => {
                if (e.target === e.currentTarget) setDetailsId(null)
              }}
            >
              <CharacterDetails
                card={detailsCard}
                onInfo={props.onInfo}
                onMeta={props.onMeta}
                onExport={props.onExport}
                busy={locked}
                onClose={() => setDetailsId(null)}
              />
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
              aria-label={t('lib.menuAria', { name: menuCard.name })}
              style={{ left: menu.x, top: menu.y }}
              onClick={(e) => e.stopPropagation()}
              onContextMenu={(e) => e.preventDefault()}
            >
              <button
                type="button"
                className="menu-item"
                role="menuitem"
                onClick={() => void beginEdit(menuCard)}
              >
                {t('common.edit')}
              </button>
              {menuCard.shapes_compatible === false ? (
                <button
                  type="button"
                  className="menu-item"
                  role="menuitem"
                  onClick={() => {
                    setRepairId(menuCard.id)
                    setMenu(null)
                  }}
                >
                  {t('common.repair')}
                </button>
              ) : null}
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
                {t('common.rename')}
              </button>
              {SHOW_CHARACTER_DETAILS ? (
                <button
                  type="button"
                  className="menu-item"
                  role="menuitem"
                  onClick={() => openDetails(menuCard)}
                >
                  {t('lib.details')}
                </button>
              ) : null}
              <button
                type="button"
                className="menu-item"
                role="menuitem"
                title={t('lib.showInFolderTitle')}
                onClick={() => void revealCard(menuCard)}
              >
                {t('lib.showInFolder')}
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
                {t('common.remove')}
              </button>
            </div>,
            document.body,
          )
        : null}
    </>
  )
}
