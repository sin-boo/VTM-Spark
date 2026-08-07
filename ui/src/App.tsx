import { useCallback, useEffect, useRef, useState } from 'react'
import {
  api,
  openAppSocket,
  type AppStatus,
  type CameraInfo,
  type Checkpoint,
  type WsEvent,
} from './api'
import { CelStage } from './components/CelStage'
import { ControlRail } from './components/ControlRail'
import { MetricStrip } from './components/MetricStrip'
import './App.css'

export default function App() {
  const [status, setStatus] = useState<AppStatus | null>(null)
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([])
  const [cameras, setCameras] = useState<CameraInfo[]>([])
  const [refPath, setRefPath] = useState('')
  const [frame, setFrame] = useState<string | null>(null)
  const [error, setError] = useState('')
  const cameraPicked = useRef(false)

  const applyStatus = useCallback((s: AppStatus) => {
    setStatus(s)
    if (s.reference_path) setRefPath(s.reference_path)
  }, [])

  const refreshCameras = useCallback(async () => {
    try {
      const data = await api.cameras()
      setCameras(data.cameras)
      if (!cameraPicked.current) {
        cameraPicked.current = true
        await api.settings({ camera_index: data.preferred })
      }
    } catch (e) {
      setError(String(e))
    }
  }, [])

  useEffect(() => {
    let alive = true
    ;(async () => {
      try {
        const [s, ck, def] = await Promise.all([
          api.status(),
          api.checkpoints(),
          api.defaultRef(),
        ])
        if (!alive) return
        applyStatus(s)
        setCheckpoints(ck)
        if (!s.reference_path && def.exists === 'true') setRefPath(def.path)
        await refreshCameras()
      } catch (e) {
        if (alive) setError(String(e))
      }
    })()

    const ws = openAppSocket((ev: WsEvent) => {
      if (ev.type === 'status') applyStatus(ev.status)
      if (ev.type === 'frame' && ev.image) setFrame(ev.image)
    })
    return () => {
      alive = false
      ws.close()
    }
  }, [applyStatus, refreshCameras])

  async function run(label: string, fn: () => Promise<unknown>) {
    setError('')
    try {
      await fn()
    } catch (e) {
      setError(`${label}: ${String(e)}`)
    }
  }

  return (
    <div className="desk">
      <ControlRail
        status={status}
        checkpoints={checkpoints}
        cameras={cameras}
        refPath={refPath}
        error={error}
        onRefPath={setRefPath}
        onCheckpoint={(path) => run('Model', () => api.setCheckpoint(path))}
        onBrowseCheckpoint={() =>
          run('Model', async () => {
            const res = await api.browseCheckpoint()
            if (!res.cancelled) {
              applyStatus(res.status)
              const ck = await api.checkpoints()
              setCheckpoints(ck)
            }
          })
        }
        onUploadRef={(file) =>
          run('Reference', async () => {
            const res = await api.uploadRef(file)
            setRefPath(res.path)
            applyStatus(res.status)
            if (res.frame?.image) setFrame(res.frame.image)
          }).finally(async () => {
            // If apply failed mid-flight, pull cleared busy/error from the backend.
            try {
              applyStatus(await api.status())
            } catch {
              /* ignore */
            }
          })
        }
        onApplyRef={() =>
          run('Reference', async () => {
            const res = await api.applyRef(refPath.trim())
            applyStatus(res.status)
            if (res.frame?.image) setFrame(res.frame.image)
          }).finally(async () => {
            try {
              applyStatus(await api.status())
            } catch {
              /* ignore */
            }
          })
        }
        onSettings={(patch) =>
          run('Settings', async () => {
            applyStatus(await api.settings(patch))
          })
        }
        onToggleTracking={() =>
          run('Tracking', () =>
            status?.tracking ? api.stopTracking() : api.startTracking(),
          )
        }
        onCalibrate={() => run('Calibrate', () => api.recenter())}
        onGenerate={() => run('Generate', () => api.generate())}
        onToggleStream={() =>
          run('Stream', () => (status?.streaming ? api.stopStream() : api.startStream()))
        }
        onRefreshCameras={() => void refreshCameras()}
      />

      <main className="main">
        <CelStage image={frame} live={Boolean(status?.streaming)} />
        <MetricStrip
          fps={status?.gen_fps ?? 0}
          timing={status?.timing ?? ''}
          checkpoint={status?.checkpoint ?? ''}
        />
      </main>
    </div>
  )
}
