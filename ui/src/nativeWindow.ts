type Cmd = 'minimize' | 'toggle_max' | 'close' | 'start_drag'

export type ResizeEdge =
  | 'left'
  | 'right'
  | 'top'
  | 'bottom'
  | 'topleft'
  | 'topright'
  | 'bottomleft'
  | 'bottomright'

type ResizeApi = {
  start_resize?: (side: string) => unknown
  begin_resize?: (side: string) => unknown
  move_resize?: () => unknown
  end_resize?: () => unknown
}

function pyApi() {
  return (
    window as unknown as {
      pywebview?: { api?: Partial<Record<Cmd, () => unknown>> & ResizeApi }
    }
  ).pywebview?.api
}

function swallow(ret: unknown) {
  if (ret && typeof (ret as Promise<unknown>).then === 'function') {
    void (ret as Promise<unknown>).catch(() => undefined)
  }
  return ret
}

export function nativeWindow(cmd: Cmd) {
  try {
    swallow(pyApi()?.[cmd]?.())
  } catch {
    /* native bridge not ready */
  }
}

export function beginResize(edge: ResizeEdge) {
  const api = pyApi()
  try {
    return swallow(api?.begin_resize?.(edge) ?? api?.start_resize?.(edge))
  } catch {
    return undefined
  }
}

export function moveResize() {
  try {
    return swallow(pyApi()?.move_resize?.())
  } catch {
    return undefined
  }
}

export function endResize() {
  try {
    return swallow(pyApi()?.end_resize?.())
  } catch {
    return undefined
  }
}

export function dragIfPrimary(button: number) {
  if (button !== 0) return
  nativeWindow('start_drag')
}
