"""
Import-order bootstrap for the API process.

`app.py` establishes an order that is load-bearing on Windows:

    patches -> encoding patch -> config/languages -> CUDA/DLL setup
    -> late patches -> application imports

`src.utils.config` must be imported *before* `torch`/`transformers`, otherwise
CTranslate2 and Torch load conflicting DLLs and the process dies with
`WinError 127`. `config` itself does real work at import time (creates the
temp/output/jobs directories, sets up CUDA DLL paths, pre-loads CTranslate2),
so anything that imports it first also inherits that setup.

This module reproduces exactly that sequence, and nothing more. It deliberately
does *not* import the pipeline: importing `src.core.video_translator` would pull
in torch, transformers and every optional backend, which is far too heavy for an
API that must answer `/api/health` quickly and must start even when optional
models are missing.

Call `bootstrap()` before importing anything that touches the pipeline.
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)

_bootstrapped = False
_lock = threading.Lock()


def bootstrap() -> None:
    """
    Apply the pipeline's import-order requirements once per process.

    Idempotent and thread-safe: FastAPI may import the app from several threads
    during startup, and applying the encoding patch twice would wrap
    `builtins.open` twice.
    """
    global _bootstrapped
    with _lock:
        if _bootstrapped:
            return
        _bootstrapped = True

        # 1. Encoding patches (critical on Windows; no-op elsewhere).
        from src.utils.patches import apply_encoding_patch
        apply_encoding_patch()

        # 2. Config + language registry. MUST precede torch/transformers.
        from src.utils import config, languages  # noqa: F401  (import order matters)

        # 3. CUDA/DLL setup. `config` already ran this at import; calling again
        #    is what `app.py` does and is safe (the setup is idempotent).
        if hasattr(config, "setup_cuda_dlls"):
            config.setup_cuda_dlls()

        # 4. Late patches (need torch/libraries loaded).
        from src.utils.patches import apply_late_patches
        apply_late_patches()

        logger.info("API bootstrap complete (device=%s)", config.DEVICE)


def is_bootstrapped() -> bool:
    """Whether `bootstrap()` has run in this process."""
    return _bootstrapped
