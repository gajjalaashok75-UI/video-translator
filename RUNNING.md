# Running video-translator end to end

This is the practical run guide: what to install, the exact commands per
operating system, what you should see at each step, and how to tell a real
problem from noise. It is written for the codebase in this archive.

If anything here disagrees with `README.md`, `README.md` describes intent and
this file describes the commands. The three things this project cannot run
without are **Python 3.10-3.12**, **FFmpeg on PATH**, and (for anything but the
smallest models) a **GPU**.

---

## 0. First: what `debug_crash.log` is (read this before worrying)

`debug_crash.log` is written by the application itself, not by VS Code. Opening
the folder in VS Code created it because the editor imports the project to build
its index, and importing the project runs this at the top of
`src/utils/config.py`:

```python
DEBUG_LOG_PATH = BASE_DIR / "debug_crash.log"
debug_log("--- SESSION START ---")     # runs at import time
debug_log(f"Python: {sys.version}")
debug_log(f"Platform: {sys.platform}")
```

So any action that imports `src.utils.config` — a test run, `python app.py`, or
your editor — appends a session block and creates the file if missing. The file
holds at most a few lines per session and is **not** evidence of a crash unless
you also see a real error later in it.

The two lines you pasted:

```
Checking src/lib: ...\src\lib
src/lib not found!
Checking torch/lib: ...\Python311\torch\lib
torch/lib not found at expected path: ...
```

are **normal first-run messages**, not errors. `src/lib` is a vendored DLL
folder that does not exist in a source checkout, and `torch/lib` is only found
when PyTorch is installed. Both lines appear before you install anything. They
will keep appearing on a system where the packages are not installed yet — that
is the expected state before step 2.

This file is ignored by git (`.gitignore` has `*.log`), so it will not show up
in `git status`.

---

## 1. Prerequisites

| Need | Version | Why |
|------|---------|-----|
| Python | 3.10-3.12 (3.11 recommended) | 3.13 is not supported by several pinned ML wheels |
| FFmpeg | any recent | audio extract, mux, container changes |
| Git | any | to clone/update |
| GPU + driver | NVIDIA, CUDA 12.4+ | 8 GB VRAM min, 12 GB+ for HD lip-sync |
| Rubberband | optional | higher-quality time-stretching |

Install FFmpeg:

```bash
# Windows (PowerShell)
winget install ffmpeg        # then open a NEW terminal

# Linux (Debian/Ubuntu)
sudo apt update && sudo apt install -y ffmpeg

# macOS
brew install ffmpeg
```

Verify all three before going on:

```bash
python --version       # expect 3.10.x - 3.12.x
ffmpeg -version        # expect a version line
nvidia-smi             # optional; shows the GPU and driver
```

If `python --version` prints 3.13, install 3.11 and use it explicitly (the
installer scripts look for `python3.10`, `python3.11`, `python3.12`).

---

## 2. Install

### Option A — the smart installer (recommended)

It detects the OS, GPU and CUDA version and installs the right extras.

```bash
# --- Linux / macOS ---
chmod +x scripts/install.sh
./scripts/install.sh                  # auto-detect
./scripts/install.sh --cpu            # force CPU-only
./scripts/install.sh --cuda 12.4      # force CUDA 12.4
./scripts/install.sh --cuda 12.8      # RTX 50 series / Blackwell
./scripts/install.py --verify         # only check an existing install
```

```bat
:: --- Windows ---
scripts\install.bat
:: or, from an activated venv:
python scripts\install.py
python scripts\install.py --cpu
python scripts\install.py --cuda 12.8
```

The installer creates a `venv/` in the project root and installs into it.

### Option B — manual, by OS

```bash
# ---------- Linux ----------
python3.11 -m venv venv
source venv/bin/activate
pip install -U pip setuptools wheel
pip install -e ".[all]"               # CUDA 12.4 default
# pip install -e ".[all-cu128]"       # CUDA 12.8
# pip install -e ".[all-cpu]"         # CPU only
```

```bash
# ---------- macOS (Apple Silicon: CPU only, no CUDA) ----------
python3.11 -m venv venv
source venv/bin/activate
pip install -U pip setuptools wheel
pip install -e ".[all-cpu]"
```

```bat
:: ---------- Windows ----------
py -3.11 -m venv venv
venv\Scripts\activate
python -m pip install -U pip setuptools wheel
pip install -e ".[all]"
```

Some packages are platform-limited on purpose (see `pyproject.toml`):
`bitsandbytes` and `paddlepaddle-gpu` are Linux-only, and
`onnxruntime-gpu` is chosen for Windows/Linux while macOS gets plain
`onnxruntime`.

### Two optional packages the installer does not add

```bash
pip install voicefixer            # only needed for Audio Enhancement / HIGH preset
pip install piper-tts==1.2.0      # only if you want the Piper voice (Linux)
```

`voicefixer` has broken package metadata on pip >= 24.1 and is intentionally not
in the dependency set. Without it, `enable_audio_enhancement=True` (the HIGH
preset) is unavailable; everything else works.

### Download the model weights

```bash
python scripts/download_models.py
```

This fetches Wav2Lip (`models/wav2lip/wav2lip_gan.pth`) and GFPGAN
(`models/gfpgan/GFPGANv1.4.pth`). Heavier models (Whisper, IndicF5, NLLB,
LivePortrait, speaker embeddings) download on first use and need `HF_TOKEN` in
`.env` for the gated ones:

```env
# .env in the project root
HF_TOKEN=hf_your_token_here
```

### Verify

```bash
python scripts/install.py --verify
# or run the suite:
pytest -q
```

Expected: the verification prints one line per core dependency with `OK`, and
the tests finish **1059 passed, 14 skipped, 0 failed**. Skips are tests that need
model weights or a GPU. A handful of passed tests is not enough — check the
count.

---

## 3. Run

### 3a. The web UI (normal way)

```bash
# Linux / macOS
source venv/bin/activate
python app.py
```

```bat
:: Windows
venv\Scripts\activate
python app.py
```

Expected console output (first run downloads models, so it is slow):

```
[Config] Initializing DLL paths... BASE_DIR=...
[Config] CTranslate2 loaded successfully. Version: ...
* Running on local_url=http://0.0.0.0:7860
```

Open <http://127.0.0.1:7860> (the app listens on `0.0.0.0:7860`). Then:

1. Upload a video (MP4, MKV, MOV).
2. Pick source language (or Auto-detect) and target language.
3. Pick models. For a first run use the defaults:
   `Torchaudio HDemucs` separation, `edge` TTS, `Google Translate (Online, Fast)`,
   `Large v3 Turbo (Fast)` transcription.
4. Leave the optional toggles off for the first run.
5. Click **Process Video** and watch the progress panel.

What a good run looks like, stage by stage: extraction -> separation (two stems)
-> VAD -> transcription (segments appear with text) -> translation -> TTS clips
-> synchronize (merged speech) -> mix -> final mux. The finished file lands in
`output/`. Durations should match the input within about a tenth of a second,
and the audio should not click or clip.

### 3a-bis. The read-only HTTP API (F1)

The FastAPI layer is **additive**. It reports what this machine can do — device,
model availability, languages, backends, options, voices — and does not run any
dubbing yet. It needs the `api` extra:

```bash
pip install -e ".[api]"
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Interactive schema at <http://127.0.0.1:8000/docs>. Endpoints:

| Endpoint | What it answers |
| --- | --- |
| `GET /api/health` | Liveness. Cheap; loads no models. |
| `GET /api/system` | Python, OS, FFmpeg, device, CUDA, GPU. |
| `GET /api/models` | Per-component state: `available`, `not_installed`, `model_not_downloaded`, `token_required`, `unsupported_platform`, `unavailable`, `unknown`. |
| `GET /api/languages` | Every language, derived from `src/utils/languages.py`. |
| `GET /api/capabilities` | Backends, cloning engines, presets, features, route matrix. |
| `GET /api/options` | The dropdown lists (reproduced from `app.py` during F1). |
| `GET /api/voices?tts_model=edge&language=hi` | Voices for a backend and language. |

Probing is read-only: it never downloads weights and never runs inference, so it
is safe on a CPU-only machine with no `HF_TOKEN`. The API process starts even
when optional models are missing, and `python app.py` continues to be the UI.

### 3a-ter. The integrated app: one command, one origin (F7)

For the React UI plus the API in a single process, build the frontend once and
launch with the bundled launcher:

```bash
cd frontend && npm install && npm run build   # once, writes frontend/dist
cd ..
python run.py                                 # http://127.0.0.1:8000
```

`python run.py` is the production entry point. It checks the interpreter and the
web dependencies, prints a short banner, then serves the API and the built UI
from one origin with a single in-process worker, so Ctrl+C stops it cleanly and
leaves no orphaned child process. Useful flags:

```bash
python run.py --open                     # open a browser once the server is up
python run.py --host 0.0.0.0 --port 8080 # bind elsewhere
python run.py --check                    # run the environment checks and exit
```

The same knobs work from the environment (`API_HOST`, `API_PORT`,
`API_OPEN_BROWSER`, `API_LOG_LEVEL`, `API_FRONTEND_DIST`,
`API_SERVE_FRONTEND`, `API_REQUIRE_FRONTEND`) so a service file can set them
without changing the command line. A missing frontend build is **not** fatal:
the server starts API-only and `/` says how to build the UI. Set
`API_REQUIRE_FRONTEND=1` to make a missing build an error instead.

Development is unchanged: run `python run.py` (or `uvicorn src.api.main:app
--reload`) and `cd frontend && npm run dev` in two terminals. See
`frontend/README.md` for the dev proxy and the F7 serving rules.

### 3a-quater. On a free GPU host: Colab and Kaggle (A9)

If you have no GPU locally, run the same application on Colab or Kaggle:

- **Colab** — open `deploy/notebooks/colab.ipynb`, choose
  **Runtime → Change runtime type → T4 GPU**, then **Run all**. The printed URL
  is proxied by Colab and opens with no account or token.
- **Kaggle** — open `deploy/notebooks/kaggle.ipynb`, turn on **Internet** and a
  GPU accelerator, then **Run all**. Kaggle has no port proxy, so to open the UI
  from your browser set `PUBLIC_HOST` to a tunnel you run (Cloudflare, ngrok).

Both notebooks do the same five things: check the machine, redirect the runtime
directories to a writable location, install the package, start the server, and
print a URL. They are generated from one source:

```bash
python scripts/build_notebooks.py           # regenerate both notebooks
python scripts/build_notebooks.py --check   # verify they are current
```

The shared logic lives in `src/deploy/runtime.py` and is tested without a GPU.
Every deployment knob — the four runtime directories, `API_*`, `HF_TOKEN`,
`RUNTIME_DIR`, `PUBLIC_HOST` — is tabulated in `docs/DEPLOYMENT.md`.

### 3b. From Python (scripting, no UI)

`VideoTranslator.process_video` is a generator, so iterate it:

```python
from src.core.video_translator import VideoTranslator

vt = VideoTranslator()
for kind, payload in vt.process_video(
        "samples/hgf_scene.webm",
        source_lang="en",
        target_lang="te",
        enable_diarization=False,
        enable_time_stretch=True,
        enable_lipsync=False,
):
    if kind == "log":
        print(payload)
    elif kind == "progress":
        print(f"{payload}%")
    elif kind == "result":
        print("output:", payload)
```

Quality presets are the short way to set the visual/audio quality knobs
without changing what is said or which voice says it:

```python
vt.process_video("in.mp4", source_lang="en", target_lang="te",
                 quality_preset="FAST")      # no lip-sync, no enhancement
# "BALANCED" = standard Wav2Lip; "HIGH" = LivePortrait + enhancement (needs voicefixer)
```

### 3c. Resuming a job

Jobs live in `jobs/<source>-<timestamp>-<id>/`. To continue instead of
re-rendering, pass the job id back; the manifest's own config is replayed, so
what is said and whose voice says it stay identical:

```python
vt.process_video("in.mp4", job_id="hgf_scene-20261021-142210-f20a32a1",
                 resume=True)
```

Changing an allow-listed quality key (lip-sync, enhancement, the mix) on a
finished job invalidates only the lip-sync and final-mux stages; transcription,
diarization, translation and TTS stay cached. Changing anything else raises
`JobError` on purpose.

### 3d. Try the bundled sample offline

The archive ships a 96.7 s public-domain clip and a finished job, so you can
watch a whole run without sourcing your own video:

- `samples/hgf_scene.webm` — the input.
- `jobs/hgf_scene-20261021-142210-f20a32a1/` — its cached state (stems,
  segments, TTS takes).
- `output/translated_hgf_scene.webm` — a finished render (this one predates the
  Phase 7 mixer, so compare it against a fresh render).

Afraid of burning GPU time? Set `enable_lipsync=False` for the first pass.
Lip-sync is by far the slowest stage.

---

## 4. Expected results and how to read them

| Signal | Healthy | Note |
|--------|---------|------|
| Exit status | 0 | a Python traceback means it failed |
| Final file | in `output/`, `*.webm`/`*.mp4` | container follows the input |
| Duration | within ~0.1 s of the source | larger drift is a real bug |
| Audio | no clicks, no clipping | peak under 0.95 |
| `debug_crash.log` | a few lines per session | only interesting if an error follows |
| `git status` | clean | the log is ignored |

Expected test suite: **1059 passed, 14 skipped, 0 failed**. If you see failures,
run just the Phase 7 suites to localise:

```bash
pytest tests/test_phase7_scoping_lipsync_mix.py tests/test_phase7_real_validation.py -q
```

The real-sample suite needs FFmpeg and the sample clip; its face-detector tests
download the S3FD/2DFAN4 weights on first run, which needs network once.

---

## 5. Troubleshooting

**`ModuleNotFoundError: No module named 'ctranslate2'` (or torch, etc.)**
The venv is not active, or the install did not finish. Activate the venv and
re-run `pip install -e ".[all]"`. In VS Code, select `venv/Scripts/python.exe`
(Windows) or `venv/bin/python` (Linux/macOS) as the interpreter.

**`src/lib not found!` / `torch/lib not found at expected path`**
Normal before installing, and harmless afterwards if PyTorch still loads. If
imports fail *and* these lines are the last thing in `debug_crash.log`, then
PyTorch genuinely is not installed. Re-run the install.

**`ffmpeg not found`**
FFmpeg is not on PATH. Install it (step 1) and open a new terminal so PATH is
re-read.

**CUDA out of memory**
Lower the load: `enable_lipsync=False`, or `quality_preset="FAST"`, or a smaller
transcription model. The code already falls back to CPU face detection on a CUDA
error, but that is slow.

**`voicefixer` ImportError**
Expected — it is optional. Do not use the HIGH preset's enhancement until you
`pip install voicefixer`.

**Slow first run**
Models download on first use (Whisper large-v3 alone is ~1.5 GB). Subsequent
runs use the cache. This is the single most common "it is stuck" confusion; watch
the progress panel, not the clock.

**VS Code recreates `debug_crash.log` after deleting it**
Expected. The editor imports the package to index it. Leave it; it is ignored by
git. To stop it entirely, point the editor at the venv interpreter so the import
succeeds silently, but the file will still be created.

---

## 6. What is and is not in this archive

Included: `src/`, `tests/`, `scripts/`, `deploy/`, docs, `samples/`,
`output/`, `temp/`, `jobs/`.

Excluded: `references/` (design input only), `.venv/` (recreate with the
installer), model weights (download with `scripts/download_models.py`), and
caches (`__pycache__/`, `node_modules/`, `.pytest_cache/`). `debug_crash.log` is
in `.gitignore`; it may exist locally and is not tracked.

After unzipping, the first two commands to run are the install (step 2) and
`pytest -q` (step 2 "Verify"). If those two are clean, the code is sound and
anything left is configuration.
