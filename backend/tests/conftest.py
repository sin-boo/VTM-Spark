"""Shared pytest setup: vendor import paths (live-poser, torch_train, …)."""

from backend.paths import ensure_import_paths

ensure_import_paths()
