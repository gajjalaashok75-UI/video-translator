#!/usr/bin/env python3
"""
Build the Colab and Kaggle notebooks from one source.

Both notebooks do the same thing — provision a GPU box, point the runtime tree
at a writable directory, install the package, start the server, and show the
user a URL — and differ only in the platform-specific opening cell (which drive
to mount, which proxy to use, what to tell the user about the URL).

Authoring them as `.ipynb` JSON by hand would let the two drift, and a
hand-edited notebook is not reviewable in a diff. So the content lives here as
plain Python data and this script writes both files. `--check` re-renders and
compares against what is on disk, which is what the test suite runs, so a
notebook that was edited in the browser without updating this script fails CI.

Usage:
    python scripts/build_notebooks.py            # write both notebooks
    python scripts/build_notebooks.py --check    # fail if they are out of date
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_DIR = PROJECT_ROOT / "deploy" / "notebooks"

REPO_URL = "https://github.com/gajjalaashok75-UI/video-translator"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "id": _cell_id(text), "metadata": {},
            "source": text.strip("\n")}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "id": _cell_id(text),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n"),
    }


def _cell_id(text: str) -> str:
    """
    A stable cell id.

    nbformat 4.5 requires one and browsers rewrite it on save, so deriving it
    from the content keeps a re-render byte-identical to what is on disk —
    which is what `--check` compares.
    """
    import hashlib

    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()[:12]


def notebook(cells: list[dict], *, kernel_name: str, display_name: str) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": display_name,
                "language": "python",
                "name": kernel_name,
            },
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# ---------------------------------------------------------------------------
# Shared cells
# ---------------------------------------------------------------------------
TITLE_MD = """
# Video Translator — hosted deployment

Run the full Video Translator (FastAPI API + React UI) on a free GPU host and
open it in your browser. The heavy lifting — vocal separation, transcription,
translation, voice cloning, lip-sync — runs on the machine this notebook is
attached to, not on your laptop.

**Before you start:** this needs a **GPU runtime**.
{platform_hint}

> ⏱️ First run takes 10–20 minutes: the package install pulls PyTorch and the ML
> stack. Later runs in the same session are fast because the install is cached.
>
> **Python version:** this stack is verified on **3.10–3.12**. Colab's default
> kernel is now 3.13, and several extras have no cp313 wheels yet, so the
> install cell fails there. Switch the runtime to **Python 3.11** before
> running: *Runtime → Change runtime type → Python 3.11 + GPU*.
"""

CHECKS_MD = """
## 1. Check the machine

Confirms we have a GPU, picks the runtime directory for this platform, and makes
sure FFmpeg is present. Nothing here downloads models — that happens lazily on
the first render, so this cell is quick.
"""

CHECKS_CODE = """
import os
import sys

# The repository is cloned into the working directory below.
REPO_DIR = "{repo_dir}"
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)

from src.deploy import runtime

platform = runtime.detect_platform()
report = runtime.hardware_report()

print(f"Platform:  {{platform}}")
print(f"Python:    {{sys.version.split()[0]}}")
if report["cuda_available"]:
    print(f"GPU:       {{report['gpu_name']}} ({{report['vram_gb']}} GB VRAM)")
else:
    print("GPU:       none — renders will run on CPU and will be slow")
print(f"Preset:    {{report['recommended_preset']}}")
for note in report["notes"]:
    print(f"  note: {{note}}")

if runtime.ensure_ffmpeg():
    print("FFmpeg:    ready")
else:
    print("FFmpeg:    NOT available — install it before rendering")
"""

CONFIGURE_MD = """
## 2. Point the runtime at a writable directory

The application writes temporary files, finished renders and job state. On a
hosted machine the checkout is not the right place for that (it is small, and on
Kaggle it is read-only). This cell redirects `TEMP_DIR`, `OUTPUT_DIR`,
`JOBS_DIR` and the uploads directory to a platform-appropriate location.

**It must run before the application is imported**, because `src.utils.config`
reads those variables at import time.
"""

CONFIGURE_CODE = """
from pathlib import Path

runtime_dirs = runtime.configure_runtime()
runtime_root = next(iter(runtime_dirs.values())).parent

print(f"Runtime root: {runtime_root}")
for name, path in runtime_dirs.items():
    print(f"  {name:8} -> {path}")

# Finished videos land in the output directory; remember the path for later.
OUTPUT_DIR = runtime_dirs["output"]
"""

INSTALL_MD = """
## 3. Install the application

Installs the package with the extras a GPU render needs: the CUDA PyTorch build,
the API layer, the default TTS and translation backends, audio processing, and
the optional stages the notebook advertises. This is the slow cell.

Every stage the UI can offer is installed here so the app reports it as
available rather than falling back: Whisper transcription (via
`translation-llm`), Google/NLLB translation, Edge TTS, Demucs vocal
separation, speaker diarization, lip-sync and on-screen text translation.
Diarization and IndicF5 voice cloning need a Hugging Face token (see step 5);
the rest runs with no keys.
"""

INSTALL_CODE = """
import subprocess
import sys

PIP_EXTRAS = "[gpu-cu124,api,tts,translation-llm,audio-processing,diarization,lipsync,ocr]"

# `-e` so the repo is importable as `src.*` and edits are picked up.
cmd = [
    sys.executable, "-m", "pip", "install",
    "-e", f"{REPO_DIR}{PIP_EXTRAS}",
]
print("Installing:", " ".join(cmd))
print("Python:", sys.version.split()[0])
# Stream pip to the notebook in real time and keep the full transcript, so a
# failure shows exactly what was attempted and which package died instead of a
# generic message with nothing to go on.
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, bufsize=1)
transcript = []
for line in proc.stdout:
    print(line, end="")
    transcript.append(line)
proc.wait()

if proc.returncode != 0:
    print()
    print("=" * 60)
    print("INSTALL FAILED. Full pip output above this line.")
    print("=" * 60)
    # `pyproject.toml` caps the interpreter at <3.13, and several extras in
    # this stack (paddlepaddle 2.6.x, opencv 4.10.0.84, the pinned torch
    # wheel) have no cp313 wheels yet, so pip falls back to a source build and
    # fails. Surface that plainly instead of burying it in the scrollback.
    if sys.version_info >= (3, 13):
        raise SystemExit(
            "This stack is only verified on Python 3.10-3.12: several extras "
            "(paddlepaddle 2.6.x, opencv 4.10.0.84, the pinned torch wheel) "
            "have no cp313 wheels, so pip tries to build from source and "
            "fails. Switch the runtime to Python 3.11 (Colab: Runtime -> "
            "Change runtime type -> Python 3.11 + GPU) and re-run this cell.")
    raise SystemExit(
        "The install failed. Read the full pip output above this line for the "
        "real cause (a missing wheel, an unresolvable pin, a gated model). "
        "Re-run this cell after fixing it.")
print("Install complete.")
"""

START_MD = """
## 4. Start the server

Starts the API and the built UI in this process and waits until it answers
`/api/health`. The cell prints the URL to open.

The server runs in a background thread of this kernel, so the notebook stays
responsive. Re-running the cell stops the previous server first, so you never
end up with two processes fighting over the port.
"""

START_CODE = """
# Stop any server a previous run of this cell left behind.
_previous = globals().get("server")
if _previous is not None:
    _previous.stop()
    print("Stopped the previous server.")

PORT = 8000
server = runtime.start_server(host="0.0.0.0", port=PORT, ready_timeout=300)

url = runtime.notebook_public_url(PORT)
print()
print("=" * 62)
print("  Video Translator is running")
print(f"  Open:  {{url}}")
print("=" * 62)
print()
print("Keep this notebook tab open while you use the app.")
{url_note}
"""

USE_MD = """
## 5. Use it

1. Open the URL printed above.
2. On the **Dub** page, drop in a video and pick the source and target
   languages. Hindi and the other Indic languages are supported.
3. Leave the quality preset at the recommended value shown in step 1
   (`FAST` on a CPU box, `BALANCED` or `HIGH` on a GPU).
4. Press **Start** and watch the progress.

Renders are written to the output directory shown in step 2. The **Jobs** page
lists everything this session has produced.

### Optional: a Hugging Face token

Speaker diarization, IndicF5 voice cloning (the 11 Indian languages) and the
Llama/ALMA translation models are gated on Hugging Face. Set it before
starting the server, then restart the server cell:

```python
import os
os.environ["HF_TOKEN"] = "hf_..."   # or use the Secrets panel on the left
```

Without it those stages report as unavailable and the pipeline falls back:
diarization runs as a single speaker, Indian languages speak via Edge TTS
instead of a cloned IndicF5 voice, and Llama/ALMA translation is not offered.
Lip-sync (Wav2Lip/LivePortrait) and on-screen text translation (PaddleOCR) are
installed by this notebook and work with no token.
"""

STOP_MD = """
## 6. Stop the server

Frees the port and the GPU memory. Run this when you are done, or just close the
notebook — the server is a daemon thread and dies with the kernel.
"""

STOP_CODE = """
server.stop()
print("Server stopped:", not server.is_running())
"""

TROUBLESHOOT_MD = """
## Troubleshooting

| Symptom | Fix |
|---|---|
| `The server did not become ready` | Re-run the install cell; a partial install is the usual cause. Check the log output above the error. |
| `The install failed. ... This stack is only verified on Python 3.10-3.12` | Colab's default kernel is now 3.13 and several extras have no cp313 wheels. Switch the runtime to **Python 3.11** (*Runtime → Change runtime type → Python 3.11 + GPU*) and re-run this cell. |
| `The install failed. ... Read the full pip output above` | A different resolver error. Scroll up inside this cell to see which package failed to build. |
| Out of GPU memory during a render | Use the `FAST` preset (no lip-sync), or process a shorter clip. |
| `ffmpeg not found` | Re-run step 1; on Colab and Kaggle the cell installs it. |
| The URL does not open | The link is only reachable from inside the VM. Re-run step 4 to get a fresh proxy URL, and keep the notebook open. |
| Diarization, Llama translation or IndicF5 cloning unavailable | Set `HF_TOKEN` (see step 5) and restart the server. |
"""


# ---------------------------------------------------------------------------
# Platform-specific cells
# ---------------------------------------------------------------------------
COLAB_HINT = """
In Colab: **Runtime → Change runtime type → Python 3.11 + T4 GPU** (the 3.13
default is not supported by this stack yet). Then **Runtime → Run all**.
"""

KAGGLE_HINT = """
In Kaggle: **Settings → Accelerator → GPU T4 x2** (or P100), turn **Internet**
on, and set the kernel to **Python 3.11** (Kaggle defaults to 3.13, which this
stack does not support yet — see the note at the top). Then **Run → Run all**.
"""

COLAB_SETUP_MD = """
## 0. Get the code (Colab)

Colab gives a GPU VM with a writable `/content`. We clone the repository there
and install FFmpeg, which the pipeline shells out to for every render.

**Runtime:** this stack needs **Python 3.11**, not Colab's 3.13 default —
several extras have no cp313 wheels. Set *Runtime → Change runtime type →
Python 3.11 + T4 GPU* before running this cell.
"""

COLAB_SETUP_CODE = """
import os
import subprocess

REPO_DIR = "/content/video-translator"

if not os.path.isdir(os.path.join(REPO_DIR, ".git")):
    print("Cloning the repository...")
    subprocess.run(
        ["git", "clone", "--depth", "1", "{repo_url}", REPO_DIR], check=True)
else:
    print("Repository already present; updating...")
    subprocess.run(["git", "-C", REPO_DIR, "pull", "--ff-only"], check=True)

print("Installing FFmpeg...")
subprocess.run(["apt-get", "-qq", "update"], check=False)
subprocess.run(["apt-get", "-qq", "install", "-y", "ffmpeg"], check=True)
print("Ready:", REPO_DIR)
"""

KAGGLE_SETUP_MD = """
## 0. Get the code (Kaggle)

Kaggle mounts `/kaggle/working` as the writable, persistent directory, so the
checkout and the runtime tree both go there. **Internet must be enabled** in
the notebook settings, or the install and the model downloads will fail.

**Runtime:** Kaggle defaults to Python 3.13, which this stack does not support
yet — set the kernel to **Python 3.11** in the notebook settings before
running.
"""

KAGGLE_SETUP_CODE = """
import os
import subprocess

REPO_DIR = "/kaggle/working/video-translator"

if not os.path.isdir(os.path.join(REPO_DIR, ".git")):
    print("Cloning the repository...")
    subprocess.run(
        ["git", "clone", "--depth", "1", "{repo_url}", REPO_DIR], check=True)
else:
    print("Repository already present; updating...")
    subprocess.run(["git", "-C", REPO_DIR, "pull", "--ff-only"], check=True)

print("Installing FFmpeg...")
subprocess.run(["apt-get", "-qq", "update"], check=False)
subprocess.run(["apt-get", "-qq", "install", "-y", "ffmpeg"], check=True)
print("Ready:", REPO_DIR)
"""

COLAB_URL_NOTE = """
# On Colab this URL is proxied automatically and needs no account or token.
# It is valid only while this notebook is running.
"""

KAGGLE_URL_NOTE = """
# Kaggle has no built-in port proxy, so a link to a public tunnel is shown when
# one is configured (set PUBLIC_HOST before starting the server, for example
# with a Cloudflare or ngrok tunnel). Otherwise the URL above is reachable only
# from inside this VM, and you should drive the app from a notebook cell or
# attach a tunnel of your own.
"""


def build_colab() -> dict:
    cells = [
        md(TITLE_MD.format(platform_hint=COLAB_HINT)),
        md(COLAB_SETUP_MD),
        code(COLAB_SETUP_CODE.format(repo_url=REPO_URL)),
        md(CHECKS_MD),
        code(CHECKS_CODE.format(repo_dir="/content/video-translator")),
        md(CONFIGURE_MD),
        code(CONFIGURE_CODE),
        md(INSTALL_MD),
        code(INSTALL_CODE),
        md(START_MD),
        code(START_CODE.format(url_note=COLAB_URL_NOTE)),
        md(USE_MD),
        md(STOP_MD),
        code(STOP_CODE),
        md(TROUBLESHOOT_MD),
    ]
    return notebook(cells, kernel_name="python3", display_name="Python 3")


def build_kaggle() -> dict:
    cells = [
        md(TITLE_MD.format(platform_hint=KAGGLE_HINT)),
        md(KAGGLE_SETUP_MD),
        code(KAGGLE_SETUP_CODE.format(repo_url=REPO_URL)),
        md(CHECKS_MD),
        code(CHECKS_CODE.format(repo_dir="/kaggle/working/video-translator")),
        md(CONFIGURE_MD),
        code(CONFIGURE_CODE),
        md(INSTALL_MD),
        code(INSTALL_CODE),
        md(START_MD),
        code(START_CODE.format(url_note=KAGGLE_URL_NOTE)),
        md(USE_MD),
        md(STOP_MD),
        code(STOP_CODE),
        md(TROUBLESHOOT_MD),
    ]
    return notebook(cells, kernel_name="python3", display_name="Python 3")


def render(name: str) -> str:
    """The notebook serialised exactly as it is written to disk."""
    import nbformat

    builder = {"colab": build_colab, "kaggle": build_kaggle}[name]
    nb = nbformat.from_dict(builder())
    nbformat.validate(nb)
    return nbformat.writes(nb, version=4) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--check", action="store_true",
                        help="Verify the notebooks on disk match this script.")
    args = parser.parse_args(argv)

    targets = {
        "colab": NOTEBOOK_DIR / "colab.ipynb",
        "kaggle": NOTEBOOK_DIR / "kaggle.ipynb",
    }

    problems = []
    for name, path in targets.items():
        content = render(name)
        if args.check:
            if not path.is_file():
                problems.append(f"{path} is missing")
            elif path.read_text(encoding="utf-8") != content:
                problems.append(
                    f"{path} is out of date; run scripts/build_notebooks.py")
            else:
                print(f"OK  {path.relative_to(PROJECT_ROOT)}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            print(f"wrote {path.relative_to(PROJECT_ROOT)}")

    if problems:
        for problem in problems:
            print(f"ERROR: {problem}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
