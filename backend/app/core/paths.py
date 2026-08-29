"""
Centralized, portable path resolution for Sentinel.

Every path in this project used to be hardcoded as an absolute path
specific to one sandboxed build environment (/home/claude/sentinel/...).
That silently worked throughout development because it happened to be
where the code was actually being run FROM during building -- and would
have completely failed the moment anyone else ran this anywhere else,
including on Windows. Found via a clean-room test that actually ran the
documented README setup steps from an isolated location instead of
assuming the sandbox's own path was portable -- it wasn't; every script
was silently writing to /home/claude/sentinel/data regardless of where it
was actually invoked from, and that directory doesn't exist anywhere but
this one build environment.

All paths here are computed relative to THIS FILE's own location, which
resolves correctly regardless of operating system or where the repo is
cloned to. Every other module imports from here instead of hardcoding
its own path.
"""

from pathlib import Path

# This file lives at backend/app/core/paths.py -- four levels up is the
# Sentinel project root (the parent of backend/).
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
BACKEND_ROOT = PROJECT_ROOT / "backend"
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = BACKEND_ROOT / "app" / "core" / "trained_models"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
