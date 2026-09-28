Lean runtime deps for VTM Spark (synced by packaging/sync-vendor.ps1).

Contains:
  torch_train/     DiT inference helpers
  tools/live-poser
  tools/openseeface
  tools/pose-traker  (minimal: adapters + anime-face-detector src + small weights)

DiT checkpoints are NOT here — they download into ../../models/dit (VTM-1.5.1.pt)
