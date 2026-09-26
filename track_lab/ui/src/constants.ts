import type { FeelSettings, MixWeights, MouthPreset } from './api'

export type Busy =
  | 'track'
  | 'reset'
  | 'load'
  | 'apply'
  | 'osf'
  | 'cam'
  | 'cal'
  | 'gen'
  | ''

export const METERS: (keyof MixWeights)[] = ['smile', 'sad', 'A', 'I', 'U', 'E', 'O']

export const ZERO_WEIGHTS: MixWeights = {
  A: 0,
  I: 0,
  U: 0,
  E: 0,
  O: 0,
  smile: 0,
  sad: 0,
}

export const FEEL: { key: keyof FeelSettings; label: string; max?: number }[] = [
  { key: 'response', label: 'Response' },
  { key: 'smoothing', label: 'Smooth' },
  { key: 'mouth', label: 'Mouth', max: 2 },
  { key: 'gaze_gain', label: 'Gaze', max: 2 },
  { key: 'gaze_smooth', label: 'Gaze smooth' },
  { key: 'hair_pin', label: 'Hair pin' },
  { key: 'hair_width', label: 'Hair width', max: 2 },
]

export const OVERLAY: { key: keyof FeelSettings; label: string; title?: string }[] = [
  { key: 'use_visemes', label: 'Visemes' },
  { key: 'show_face', label: 'Face' },
  { key: 'show_skeleton', label: 'Skeleton' },
  { key: 'show_hair', label: 'Hair' },
  { key: 'show_ids', label: 'Ids' },
]

export const ZERO_FEEL: FeelSettings = {
  response: 0.65,
  smoothing: 0.48,
  mouth: 0.5,
  use_visemes: 1,
  show_face: 1,
  show_skeleton: 1,
  show_hair: 1,
  show_ids: 0,
  hair_pin: 0.7,
  hair_width: 1,
  gaze_gain: 1,
  gaze_smooth: 0.28,
}

export const DEFAULT_PRESETS: MouthPreset[] = [
  { id: 'rest', label: 'Rest', ready: false },
  { id: 'smile', label: 'Smile', ready: false },
  { id: 'sad', label: 'Sad', ready: false },
  { id: 'A', label: 'A', ready: false },
  { id: 'I', label: 'I', ready: false },
  { id: 'U', label: 'U', ready: false },
  { id: 'E', label: 'E', ready: false },
]
