import { useCallback, useEffect, useRef, useState } from 'react'
import {
  api,
  holdLabInput,
  labSourceOf,
  mergeLabStatus,
  type LabInputHold,
  ZERO_LAB_FEEL,
  openAppSocket,
  type AppStatus,
  type BootStatus,
  type CameraInfo,
  type CatalogOffer,
  type CharacterCard,
  type Checkpoint,
  type LabCommandReply,
  type LabStatus,
  type WsEvent,
} from './api'
import { CelStage } from './components/CelStage'
import { ControlRail } from './components/ControlRail'
import { MetricStrip } from './components/MetricStrip'
import { Splash } from './components/Splash'
import { WindowDots } from './components/WindowDots'
import { WindowResize } from './components/WindowResize'
import { dragIfPrimary } from './nativeWindow'
import './App.css'

const DESK_HANDOFF = new URLSearchParams(window.location.search).has('desk')

export default function App() {
  const [status, setStatus] = useState<AppStatus | null>(null)
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([])
  const [catalogOffers, setCatalogOffers] = useState<CatalogOffer[]>([])
  const [cameras, setCameras] = useState<CameraInfo[]>([])
  const [refPath, setRefPath] = useState('')
  const [frame, setFrame] = useState<string | null>(null)
  const [error, setError] = useState('')
  const [characters, setCharacters] = useState<CharacterCard[]>([])
  const [charactersCreating, setCharactersCreating] = useState(false)
  const [createStillUrl, setCreateStillUrl] = useState('')
  const [lab, setLab] = useState<LabStatus | null>(null)
  const [boot, setBoot] = useState<BootStatus | null>(null)
  const cameraPicked = useRef(false)
  const labRetryAt = useRef(0)
  const frameQueue = useRef<string[]>([])
  const labGen = useRef(0)
  const inputHold = useRef<LabInputHold | null>(null)
  const deskReady = Boolean(boot?.ready)

  const applyStatus = useCallback((s: AppStatus) => {
    setStatus((cur) => (cur ? { ...cur, ...s } : s))
    if (s.character_id === '') setRefPath('')
    else if (s.reference_path) setRefPath(s.reference_path)
    if (s.boot) setBoot(s.boot)
  }, [])

  const refreshCameras = useCallback(async (opts?: { silent?: boolean; savedIndex?: number }) => {
    try {
      const data = await api.cameras()
      setCameras(data.cameras)
      if (!cameraPicked.current) {
        cameraPicked.current = true
        const saved = opts?.savedIndex
        const listed = data.cameras.some((cam) => cam.index === saved)
        if (saved == null || !listed) {
          await api.settings({ camera_index: data.preferred })
        }
      }
    } catch (e) {
      if (!opts?.silent) setError(String(e))
    }
  }, [])

  const noteLab = useCallback(
    (next: LabStatus | ((cur: LabStatus | null) => LabStatus), seen: number) => {
      setLab((cur) => {
        const packet = typeof next === 'function' ? next(cur) : next
        const hold = inputHold.current
        const painted = holdLabInput(cur, packet, hold, seen)
        if (painted.release && inputHold.current === hold) inputHold.current = null
        return painted.lab
      })
    },
    [],
  )

  const refreshLab = useCallback(async () => {
    const seen = labGen.current
    try {
      const next = await api.labStatus()
      noteLab(next, seen)
      return next
    } catch {
      noteLab(
        (cur) => ({
          online: false,
          error: cur?.error || 'Track Lab is not running',
        }),
        seen,
      )
      return null
    }
  }, [noteLab])

  const refreshCharacters = useCallback(async () => {
    try {
      const listed = await api.characters()
      setCharacters(listed.characters ?? [])
    } catch (e) {
      setError(`Character: ${String(e)}`)
    }
  }, [])

  const refreshCheckpoints = useCallback(async () => {
    try {
      setCheckpoints(await api.checkpoints())
    } catch (e) {
      setError(`Model: ${String(e)}`)
    }
    try {
      const catalog = await api.modelCatalog()
      setCatalogOffers(catalog.offers ?? [])
    } catch {
      /* hub ping is optional */
    }
  }, [])

  useEffect(() => {
    if (deskReady) return
    let alive = true
    let started = DESK_HANDOFF
    let inflight = false
    const tick = async () => {
      if (inflight) return
      inflight = true
      try {
        const next = started ? await api.boot() : await api.startBoot()
        started = true
        if (alive) setBoot(next)
      } catch (e) {
        const msg = String(e)
        if (!alive) return
        if (msg.includes('404') || msg.toLowerCase().includes('not found')) {
          setBoot({
            ready: true,
            running: false,
            error: msg,
            awaiting: '',
            suggested: '',
            stages: {
              model: { state: 'error', progress: 1, label: 'Desk API is old — open the UI anyway' },
              character: { state: 'skip', progress: 1, label: 'Skipped' },
              lab: { state: 'skip', progress: 1, label: 'Skipped' },
            },
          })
        }
      } finally {
        inflight = false
      }
    }
    void tick()
    const id = window.setInterval(() => {
      void tick()
    }, 250)
    return () => {
      alive = false
      window.clearInterval(id)
    }
  }, [deskReady])

  useEffect(() => {
    if (!deskReady) return
    let alive = true
    ;(async () => {
      try {
        const [s, ck, catalog] = await Promise.all([
          api.status(),
          api.checkpoints(),
          api.modelCatalog().catch(() => ({ offers: [] as CatalogOffer[] })),
        ])
        if (!alive) return
        applyStatus(s)
        setCheckpoints(ck)
        setCatalogOffers(catalog.offers ?? [])
        const labStatus = await refreshLab()
        if (!labStatus?.cameras?.length) {
          await refreshCameras({ savedIndex: s.camera_index })
        }
      } catch (e) {
        if (alive) setError(String(e))
      }
      if (alive) await refreshCharacters()
    })()

    let raf = 0
    const pump = () => {
      raf = 0
      const next = frameQueue.current.shift()
      if (next) setFrame(next)
      if (frameQueue.current.length) raf = window.requestAnimationFrame(pump)
    }
    const enqueueFrame = (img: string) => {
      frameQueue.current.push(img)
      if (frameQueue.current.length > 12) {
        frameQueue.current.splice(0, frameQueue.current.length - 12)
      }
      if (!raf) raf = window.requestAnimationFrame(pump)
    }

    const ws = openAppSocket((ev: WsEvent) => {
      if (ev.type === 'status') applyStatus(ev.status)
      if (ev.type === 'frame') {
        if (ev.image) enqueueFrame(ev.image)
        else {
          frameQueue.current.length = 0
          setFrame(null)
        }
      }
    })
    return () => {
      alive = false
      if (raf) window.cancelAnimationFrame(raf)
      ws.close()
    }
  }, [applyStatus, deskReady, refreshCameras, refreshCharacters, refreshLab])

  useEffect(() => {
    if (!deskReady) return
    const ms = lab?.live ? 250 : 800
    let inflight = false
    const id = window.setInterval(() => {
      if (inflight) return
      inflight = true
      const retry = !lab?.online
      void refreshLab()
        .then((next) => {
          if (!retry || next?.online) return
          const now = Date.now()
          if (now - labRetryAt.current < 5000) return
          labRetryAt.current = now
          const seen = labGen.current
          return api
            .labConnect()
            .then((packet) => noteLab(packet, seen))
            .catch(() => {
              /* keep last lab snapshot */
            })
        })
        .finally(() => {
          inflight = false
        })
    }, ms)
    return () => window.clearInterval(id)
  }, [deskReady, lab?.live, lab?.online, noteLab, refreshLab])

  async function run<T>(label: string, fn: () => Promise<T>): Promise<T | undefined> {
    setError('')
    try {
      return await fn()
    } catch (e) {
      setError(`${label}: ${String(e)}`)
      return undefined
    }
  }

  function applyLabReply(reply: LabCommandReply) {
    if (reply.status) {
      noteLab({ ...reply.status, online: true }, labGen.current)
    }
    if (reply.ok === false && reply.error) {
      throw new Error(reply.error)
    }
  }

  async function sendLab(op: string, body: Record<string, unknown> = {}) {
    const reply = await api.labCommand(op, body)
    applyLabReply(reply)
    return reply
  }

  if (!deskReady) {
    return <Splash boot={boot} error={error} />
  }

  return (
    <div className="desk-shell">
      <WindowResize />
      <div
        className="desk-caption"
        onMouseDown={(e) => {
          if ((e.target as HTMLElement).closest('.win-caption')) return
          dragIfPrimary(e.button)
        }}
      >
        <img className="desk-caption-mark" src="/splash-mark.png" width={16} height={16} alt="" />
        <span className="desk-caption-title">VTM Noble</span>
        <WindowDots />
      </div>
      <div className="desk">
      <ControlRail
        status={status}
        checkpoints={checkpoints}
        catalogOffers={catalogOffers}
        cameras={cameras}
        refPath={refPath}
        error={error}
        onRefPath={setRefPath}
        onCheckpoint={(path) => run('Model', () => api.setCheckpoint(path))}
        onBrowseCheckpoint={() =>
          run('Model', async () => {
            const res = await api.browseCheckpoint()
            if (res.cancelled) return
            applyStatus(res.status)
            const ck = await api.checkpoints()
            setCheckpoints(ck)
          })
        }
        characters={characters}
        charactersCreating={charactersCreating}
        createStillUrl={createStillUrl}
        onRefreshCharacters={refreshCharacters}
        onCreateCharacter={(file) =>
          run('Character', async () => {
            frameQueue.current.length = 0
            setFrame(null)
            const url = URL.createObjectURL(file)
            setCreateStillUrl(url)
            setCharactersCreating(true)
            try {
              const res = await api.createCharacter(file)
              applyStatus(res.status)
              if (res.frame?.image) setFrame(res.frame.image)
              setCharacters((await api.characters()).characters ?? [])
              return res.character
            } finally {
              setCharactersCreating(false)
              URL.revokeObjectURL(url)
              setCreateStillUrl('')
            }
          })
        }
        onLoadCharacter={(id, opts) =>
          run('Character', async () => {
            const res = await api.loadCharacter(id, opts)
            if (res.status) applyStatus(res.status)
            if (res.frame?.image) setFrame(res.frame.image)
            const listed = await api.characters()
            setCharacters(listed.characters ?? [])
            return res
          })
        }
        onRemoveCharacter={(id) =>
          run('Character', async () => {
            const res = await api.removeCharacter(id)
            applyStatus(res.status)
            setCharacters(res.characters ?? [])
            if (!res.status?.character_id) {
              frameQueue.current.length = 0
              setFrame(null)
              setRefPath('')
            }
          })
        }
        onRenameCharacter={(id, name) =>
          run('Character', async () => {
            const res = await api.renameCharacter(id, name)
            applyStatus(res.status)
            setCharacters(res.characters ?? [])
          })
        }
        onImportCharacter={async (file) => {
          const res = await api.importCharacter(file)
          if (res.status) applyStatus(res.status)
          await refreshCharacters()
          return res.character
        }}
        onExportCharacter={(id, name) => api.exportCharacter(id, name)}
        onRevealCharacter={(id) => api.revealCharacter(id)}
        onCharacterInfo={(id) => api.characterInfo(id)}
        onCharacterMeta={async (meta) => {
          const res = await api.characterMeta(meta)
          if (res.characters) setCharacters(res.characters)
          else await refreshCharacters()
          void api.status().then(applyStatus).catch(() => undefined)
          return res.character
        }}
        onUploadRef={(file) =>
          run('Reference', async () => {
            const res = await api.uploadRef(file)
            setRefPath(res.path)
            applyStatus(res.status)
            if (res.frame?.image) setFrame(res.frame.image)
          }).finally(() => {
            void api.status().then(applyStatus).catch(() => undefined)
          })
        }
        onApplyRef={() =>
          run('Reference', async () => {
            const res = await api.applyRef(refPath.trim())
            applyStatus(res.status)
            if (res.frame?.image) setFrame(res.frame.image)
          }).finally(() => {
            void api.status().then(applyStatus).catch(() => undefined)
          })
        }
        onSettings={(patch) => {
          setStatus((cur) => (cur ? { ...cur, ...patch } : cur))
          void run('Settings', async () => {
            const next = await api.settings(patch)
            if (next && typeof next === 'object' && next.state) {
              applyStatus({ ...next, ...patch })
            }
          })
        }}
        onToggleTracking={() =>
          run('Tracking', async () => {
            const on = Boolean(status?.tracking) || Boolean(lab?.live)
            applyStatus(on ? await api.stopTracking() : await api.startTracking())
            await refreshLab()
          })
        }
        onCalibrate={() => run('Calibrate', () => api.recenter())}
        onGenerate={() => run('Generate', () => api.generate())}
        onToggleStream={() =>
          run('Stream', async () => {
            applyStatus(
              status?.streaming ? await api.stopStream() : await api.startStream(),
            )
          })
        }
        onTogglePause={() =>
          run('Stream', async () => {
            applyStatus(
              status?.paused ? await api.resumeStream() : await api.pauseStream(),
            )
          })
        }
        onToggleVirtualCam={() =>
          run('Virtual camera', async () => {
            const next = status?.virtual_cam
              ? await api.stopVirtualCam()
              : await api.startVirtualCam()
            applyStatus(next)
          })
        }
        lab={lab}
        onLabFeel={(patch) => {
          setLab((cur) =>
            cur
              ? { ...cur, feel: { ...ZERO_LAB_FEEL, ...cur.feel, ...patch } }
              : cur,
          )
          void run('Lab feel', async () => {
            await sendLab('set_feel', patch)
          })
        }}
        onLabCalibrate={(id) =>
          run('Calibrate', async () => {
            await sendLab('calibrate', { id })
          })
        }
        onLabSource={(source) => {
          const gen = ++labGen.current
          inputHold.current = { source, gen, settled: false }
          setLab((cur) =>
            mergeLabStatus(cur, {
              ...(cur ?? { online: false }),
              source,
              error: '',
            }),
          )
          void run('Input', async () => {
            try {
              await sendLab('set_input', { source })
              const live = Boolean(status?.tracking) || Boolean(lab?.live)
              if (source === 'ifm' && live) {
                await sendLab('start', { source: 'ifm' })
              }
              if (inputHold.current?.gen === gen) {
                inputHold.current = { source, gen: ++labGen.current, settled: true }
              }
            } catch (e) {
              const message = String(e).replace(/^Error:\s*/, '')
              setLab((cur) => ({
                ...(cur ?? { online: false }),
                source,
                error: message,
              }))
              throw e
            }
          })
        }}
        onCamera={(index) =>
          run('Camera', async () => {
            applyStatus(await api.settings({ camera_index: index }))
            setLab((cur) => (cur ? { ...cur, camera_index: index } : cur))
            if (!lab?.online || labSourceOf(lab) === 'ifm') return
            await sendLab('set_camera', { index })
          })
        }
        onRefreshCameras={() => {
          void refreshCameras({ silent: true, savedIndex: status?.camera_index })
        }}
        onRefreshCheckpoints={() => {
          void refreshCheckpoints()
        }}
        onDownloadModel={(name) =>
          run('Download', async () => {
            await api.startModelDownload(name)
            await refreshCheckpoints()
          })
        }
        onLabIfmPort={(port) =>
          run('iFacialMocap', async () => {
            await sendLab('set_ifm', { port })
          })
        }
        onReloadBackend={() => {
          void run('Reload', () => api.reloadBackend())
        }}
      />

      <main className="main">
        <CelStage
          image={frame}
          live={Boolean(status?.streaming) && !status?.paused}
          frozen={Boolean(status?.pose_frozen)}
          stillId={String(status?.character_id || status?.reference_path || '')}
        />
        <MetricStrip
          fps={status?.show_fps || status?.gen_fps || 0}
          genFps={status?.gen_fps ?? 0}
          checkpoint={status?.checkpoint ?? ''}
        />
      </main>
    </div>
    </div>
  )
}
