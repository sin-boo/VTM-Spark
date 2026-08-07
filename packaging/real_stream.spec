# DEPRECATED: do not use. Freezing torch/transformers with PyInstaller can
# consume tens of GB of RAM and crash the machine.
# Use packaging/launcher.spec + packaging/build.ps1 (thin launcher + runtime/).
raise SystemExit(
    "Use packaging/launcher.spec via packaging/build.ps1 — "
    "do not freeze the ML stack with this old spec."
)
