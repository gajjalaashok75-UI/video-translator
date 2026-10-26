"""
Hosted-deployment helpers.

`src.deploy.runtime` is the shared implementation used by the Colab and Kaggle
notebooks and by any ephemeral VM. Nothing here imports the pipeline or torch at
module import time, so it is safe to import before `src.utils.config`.
"""

from src.deploy.runtime import (  # noqa: F401
    ServerHandle,
    configure_runtime,
    default_runtime_root,
    detect_platform,
    ensure_ffmpeg,
    hardware_report,
    is_notebook,
    notebook_public_url,
    public_url,
    runtime_dirs,
    start_server,
    wait_until_ready,
)

__all__ = [
    "ServerHandle",
    "configure_runtime",
    "default_runtime_root",
    "detect_platform",
    "ensure_ffmpeg",
    "hardware_report",
    "is_notebook",
    "notebook_public_url",
    "public_url",
    "runtime_dirs",
    "start_server",
    "wait_until_ready",
]
