/** Overlay slots 0–36. L/R is the person's left/right (iFacialMocap / OSF). */

export type TrackPoint = {
  id: number
  ref: string
  name: string
  group: string
  role: string
  side: 'l' | 'r' | 'c'
  legacy: string
}

export const POINTS: TrackPoint[] = [
  { id: 0, ref: 'JAW.R', name: 'jaw_r', group: 'jaw', role: 'side', side: 'r', legacy: 'face_0' },
  { id: 1, ref: 'JAW.R.MID', name: 'jaw_r_mid', group: 'jaw', role: 'mid', side: 'r', legacy: 'face_1' },
  { id: 2, ref: 'CHIN', name: 'chin', group: 'jaw', role: 'chin', side: 'c', legacy: 'face_2' },
  { id: 3, ref: 'JAW.L.MID', name: 'jaw_l_mid', group: 'jaw', role: 'mid', side: 'l', legacy: 'face_3' },
  { id: 4, ref: 'JAW.L', name: 'jaw_l', group: 'jaw', role: 'side', side: 'l', legacy: 'face_4' },
  { id: 5, ref: 'BROW.L.OUT', name: 'brow_l_outer', group: 'brow', role: 'outer', side: 'l', legacy: 'face_5' },
  { id: 6, ref: 'BROW.L', name: 'brow_l', group: 'brow', role: 'mid', side: 'l', legacy: 'face_6' },
  { id: 7, ref: 'BROW.L.IN', name: 'brow_l_inner', group: 'brow', role: 'inner', side: 'l', legacy: 'face_7' },
  { id: 8, ref: 'BROW.R.IN', name: 'brow_r_inner', group: 'brow', role: 'inner', side: 'r', legacy: 'face_8' },
  { id: 9, ref: 'BROW.R', name: 'brow_r', group: 'brow', role: 'mid', side: 'r', legacy: 'face_9' },
  { id: 10, ref: 'BROW.R.OUT', name: 'brow_r_outer', group: 'brow', role: 'outer', side: 'r', legacy: 'face_10' },
  { id: 11, ref: 'EYE.L.IN', name: 'eye_l_inner', group: 'eye', role: 'inner', side: 'l', legacy: 'face_11' },
  { id: 12, ref: 'EYE.L.LID', name: 'eye_l_lid', group: 'eye', role: 'lid', side: 'l', legacy: 'face_12' },
  { id: 13, ref: 'EYE.L.OUT', name: 'eye_l_outer', group: 'eye', role: 'outer', side: 'l', legacy: 'face_13' },
  { id: 14, ref: 'NOSE.R', name: 'nose_r', group: 'nose', role: 'ala', side: 'r', legacy: 'face_14' },
  { id: 15, ref: 'NOSE', name: 'nose', group: 'nose', role: 'tip', side: 'c', legacy: 'face_15' },
  { id: 16, ref: 'NOSE.L', name: 'nose_l', group: 'nose', role: 'ala', side: 'l', legacy: 'face_16' },
  { id: 17, ref: 'EYE.R.IN', name: 'eye_r_inner', group: 'eye', role: 'inner', side: 'r', legacy: 'face_17' },
  { id: 18, ref: 'EYE.R.LID', name: 'eye_r_lid', group: 'eye', role: 'lid', side: 'r', legacy: 'face_18' },
  { id: 19, ref: 'EYE.R.OUT', name: 'eye_r_outer', group: 'eye', role: 'outer', side: 'r', legacy: 'face_19' },
  { id: 20, ref: 'MOUTH.U.R', name: 'mouth_upper_r', group: 'mouth', role: 'upper', side: 'r', legacy: 'face_20' },
  { id: 21, ref: 'MOUTH.U', name: 'mouth_upper', group: 'mouth', role: 'upper', side: 'c', legacy: 'face_21' },
  { id: 22, ref: 'MOUTH.U.L', name: 'mouth_upper_l', group: 'mouth', role: 'upper', side: 'l', legacy: 'face_22' },
  { id: 23, ref: 'MOUTH.R', name: 'mouth_corner_r', group: 'mouth', role: 'corner', side: 'r', legacy: 'face_23' },
  { id: 24, ref: 'MOUTH.D.R', name: 'mouth_lower_r', group: 'mouth', role: 'lower', side: 'r', legacy: 'face_24' },
  { id: 25, ref: 'MOUTH.D', name: 'mouth_lower', group: 'mouth', role: 'lower', side: 'c', legacy: 'face_25' },
  { id: 26, ref: 'MOUTH.L', name: 'mouth_corner_l', group: 'mouth', role: 'corner', side: 'l', legacy: 'face_26' },
  { id: 27, ref: 'MOUTH.D.L', name: 'mouth_lower_l', group: 'mouth', role: 'lower', side: 'l', legacy: 'face_27' },
  { id: 28, ref: 'IRIS.L', name: 'iris_l', group: 'iris', role: 'pupil', side: 'l', legacy: 'right_iris' },
  { id: 29, ref: 'IRIS.R', name: 'iris_r', group: 'iris', role: 'pupil', side: 'r', legacy: 'left_iris' },
  { id: 30, ref: 'BODY.NOSE', name: 'body_nose', group: 'body', role: 'nose', side: 'c', legacy: 'nose' },
  { id: 31, ref: 'NECK', name: 'neck', group: 'body', role: 'neck', side: 'c', legacy: 'neck' },
  { id: 32, ref: 'SHO.R', name: 'shoulder_r', group: 'body', role: 'shoulder', side: 'r', legacy: 'right_shoulder' },
  { id: 33, ref: 'ELB.R', name: 'elbow_r', group: 'body', role: 'elbow', side: 'r', legacy: 'right_elbow' },
  { id: 34, ref: 'SHO.L', name: 'shoulder_l', group: 'body', role: 'shoulder', side: 'l', legacy: 'left_shoulder' },
  { id: 35, ref: 'ELB.L', name: 'elbow_l', group: 'body', role: 'elbow', side: 'l', legacy: 'left_elbow' },
  { id: 36, ref: 'CHEST', name: 'chest', group: 'body', role: 'chest', side: 'c', legacy: 'chest' },
]

export const CAM_REFS: Record<number, string> = {
  36: 'CAM.EYE.R.OUT',
  37: 'CAM.EYE.R.UP.O',
  38: 'CAM.EYE.R.UP.I',
  39: 'CAM.EYE.R.IN',
  40: 'CAM.EYE.R.DN.I',
  41: 'CAM.EYE.R.DN.O',
  42: 'CAM.EYE.L.IN',
  43: 'CAM.EYE.L.UP.I',
  44: 'CAM.EYE.L.UP.O',
  45: 'CAM.EYE.L.OUT',
  46: 'CAM.EYE.L.DN.O',
  47: 'CAM.EYE.L.DN.I',
  48: 'CAM.LIP.OUT.R',
  49: 'CAM.LIP.OUT.UR',
  50: 'CAM.LIP.OUT.U',
  51: 'CAM.LIP.OUT.UL',
  52: 'CAM.LIP.OUT.L',
  53: 'CAM.LIP.OUT.LL',
  54: 'CAM.LIP.OUT.D',
  55: 'CAM.LIP.OUT.LR',
  56: 'CAM.LIP.OUT.DR',
  57: 'CAM.LIP.OUT.DL',
  58: 'CAM.LIP.R',
  59: 'CAM.LIP.U.R',
  60: 'CAM.LIP.U',
  61: 'CAM.LIP.U.L',
  62: 'CAM.LIP.L',
  63: 'CAM.LIP.D.L',
  64: 'CAM.LIP.D',
  65: 'CAM.LIP.D.R',
  66: 'CAM.LID.R',
  67: 'CAM.LID.L',
}

const BY_ID = Object.fromEntries(POINTS.map((p) => [p.id, p])) as Record<number, TrackPoint>

export function refOf(slot: number): string {
  return BY_ID[slot]?.ref ?? String(slot)
}

export function camRefOf(index: number): string {
  return CAM_REFS[index] ?? String(index)
}

export function camShortOf(index: number): string {
  const ref = CAM_REFS[index]
  return ref ? ref.replace(/^CAM\./, '') : String(index)
}

const DRAFT: Record<string, { open: number; spread: number; lift: number }> = {
  smile: { open: 0.04, spread: 0.1, lift: 0.14 },
  sad: { open: 0.05, spread: 0.02, lift: -0.14 },
  A: { open: 0.32, spread: 0.04, lift: 0 },
  I: { open: 0.07, spread: 0.2, lift: 0.04 },
  U: { open: 0.12, spread: -0.16, lift: 0 },
  E: { open: 0.12, spread: 0.16, lift: 0.02 },
  O: { open: 0.24, spread: -0.12, lift: 0 },
}

export function draftMouth(name: string, rest: number[][]): number[][] {
  const out = rest.map((row) => row.slice())
  const spec = DRAFT[name]
  if (!spec || out.length < 28) return out
  const right = out[23]
  const left = out[26]
  if (!right || !left) return out
  const ax = left[0] - right[0]
  const ay = left[1] - right[1]
  const width = Math.hypot(ax, ay)
  if (width < 1e-3) return out
  const alongX = ax / width
  const alongY = ay / width
  let downX = -alongY
  let downY = alongX
  if (downY < 0) {
    downX = -downX
    downY = -downY
  }
  const openPx = width * spec.open
  const spreadPx = width * spec.spread
  const liftPx = width * spec.lift
  const move = (i: number, dx: number, dy: number) => {
    const row = out[i]
    if (!row) return
    row[0] += dx
    row[1] += dy
  }
  move(20, -downX * openPx * 0.45, -downY * openPx * 0.45)
  move(21, -downX * openPx * 0.5, -downY * openPx * 0.5)
  move(22, -downX * openPx * 0.45, -downY * openPx * 0.45)
  move(24, downX * openPx * 0.55, downY * openPx * 0.55)
  move(25, downX * openPx * 0.62, downY * openPx * 0.62)
  move(27, downX * openPx * 0.55, downY * openPx * 0.55)
  move(23, -alongX * spreadPx - downX * liftPx, -alongY * spreadPx - downY * liftPx)
  move(26, alongX * spreadPx - downX * liftPx, alongY * spreadPx - downY * liftPx)
  return out
}

export const MOUTH_ENDS = ['rest', 'smile', 'sad', 'A', 'I', 'U', 'E'] as const

export const MOUTH_LABEL: Record<(typeof MOUTH_ENDS)[number], string> = {
  rest: 'Rest',
  smile: 'Smile',
  sad: 'Sad',
  A: 'A',
  I: 'I',
  U: 'U',
  E: 'E',
}

export function pairId(a: string, b: string): string {
  const i = MOUTH_ENDS.indexOf(a as (typeof MOUTH_ENDS)[number])
  const j = MOUTH_ENDS.indexOf(b as (typeof MOUTH_ENDS)[number])
  if (i < 0 || j < 0 || i === j) return ''
  return i < j ? `${a}+${b}` : `${b}+${a}`
}

/** Saved stop at t. 0.5 keeps the legacy `rest+smile` id. */
export function keyId(a: string, b: string, t: number): string {
  const pair = pairId(a, b)
  if (!pair) return ''
  const n = Math.round(Math.min(0.999, Math.max(0.001, t)) * 1000)
  if (n === 500) return pair
  return `${pair}@${n}`
}

export function keyT(id: string): number | null {
  const at = id.lastIndexOf('@')
  const base = at < 0 ? id : id.slice(0, at)
  const parts = base.split('+')
  if (parts.length !== 2 || pairId(parts[0], parts[1]) !== base) return null
  if (at < 0) return 0.5
  const n = Number(id.slice(at + 1))
  if (!Number.isInteger(n) || n <= 0 || n >= 1000) return null
  return n / 1000
}

export function keysOn(ids: string[], a: string, b: string): { id: string; t: number }[] {
  const pair = pairId(a, b)
  if (!pair) return []
  const seen = new Map<string, { id: string; t: number }>()
  for (const id of ids) {
    if (id !== pair && !id.startsWith(`${pair}@`)) continue
    const t = keyT(id)
    if (t == null) continue
    seen.set(id, { id, t })
  }
  return [...seen.values()].sort((x, y) => x.t - y.t)
}

export function sampleMouth(
  left: number[][],
  right: number[][],
  keys: { t: number; pts: number[][] }[],
  t: number,
): number[][] {
  const knots = [
    { t: 0, pts: left },
    ...keys.filter((key) => key.pts.length >= 28 && key.t > 0 && key.t < 1),
    { t: 1, pts: right },
  ].sort((a, b) => a.t - b.t)
  const u = Math.min(1, Math.max(0, t))
  let lo = knots[0]
  let hi = knots[knots.length - 1]
  for (let i = 0; i < knots.length - 1; i++) {
    if (u <= knots[i + 1].t || i === knots.length - 2) {
      lo = knots[i]
      hi = knots[i + 1]
      break
    }
  }
  const span = hi.t <= lo.t ? 0 : (u - lo.t) / (hi.t - lo.t)
  return blendMouth(lo.pts, hi.pts, span)
}

export function blendMouth(a: number[][], b: number[][], t = 0.5): number[][] {
  const u = 1 - t
  return a.map((row, i) => {
    const other = b[i] ?? row
    return [row[0] * u + other[0] * t, row[1] * u + other[1] * t, row[2] ?? 1]
  })
}
