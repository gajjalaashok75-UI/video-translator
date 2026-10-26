#!/usr/bin/env python3
"""
Regenerate the OpenAPI snapshot the frontend contract test reads.

The snapshot is what keeps `frontend/src/api/types.ts` honest: the contract test
compares the hand-written types against this document, so a field renamed or
dropped in `src/api/schemas.py` fails the frontend tests instead of surfacing as
a blank cell in the UI.

Run from the repository root, with the API's dependencies available:

    .venv/bin/python frontend/scripts/export_openapi.py

Importing the app builds the FastAPI object only; no inference is run and no
model is loaded.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = REPO_ROOT / "frontend" / "src" / "test" / "contract" / "openapi.json"

# The app imports torch and ctranslate2 through src.utils.config, so the
# repository root must be importable and config must load first.
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    # Imported here so `sys.path` is set up before the app's own imports run.
    from src.api.main import app

    spec = app.openapi()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(spec, indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT.relative_to(REPO_ROOT)} ({OUTPUT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
