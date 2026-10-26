# Deployment

How to run Video Translator somewhere other than the machine you develop on.
There are three supported targets, in increasing order of effort:

| Target | Best for | GPU | Setup time |
|--------|----------|-----|------------|
| [Local](#local) | Everyday use, privacy | your own | minutes |
| [Docker](#docker) | A server you control, reproducible deps | NVIDIA | ~30 min (first build) |
| [Colab / Kaggle](#colab-and-kaggle) | Trying it with a free GPU | provided | ~15 min |

All three run the *same* application: the FastAPI backend and the built React UI
from one origin, started by `run.py` (locally and in Docker) or by
`src.deploy.runtime.start_server` (in a notebook). Nothing here changes the
pipeline; deployment only decides where the process runs and where it writes.

---

## What a deployment needs

Regardless of target, a working install needs:

1. **Python 3.10–3.12.** Several pinned ML wheels have no 3.13 build; the
   launcher refuses 3.13 with an explanation rather than failing later.
2. **FFmpeg on `PATH`.** Every render shells out to it. The notebooks install
   it; Docker bakes it in; locally you install it yourself.
3. **A writable runtime tree.** Temporary files, finished renders, job state and
   staged uploads. Redirectable — see [Runtime directories](#runtime-directories).
4. **Disk for model weights.** Downloaded lazily on first use, several GB total.
5. **A GPU, optionally but strongly recommended.** Everything runs on CPU, just
   slowly, and lip-sync needs a GPU.

### Runtime directories

The application writes to four directories. Each is overridable by an
environment variable, so a deployment that must not write inside the checkout
can redirect the whole tree without patching code:

| Variable | Default | Holds |
|----------|---------|-------|
| `TEMP_DIR` | `<repo>/temp` | Intermediate audio/video, scratch files |
| `OUTPUT_DIR` | `<repo>/output` | Finished renders |
| `JOBS_DIR` | `<repo>/jobs` | Durable job state and manifests |
| `API_UPLOADS_DIR` | `$TEMP_DIR/uploads` | Staged uploads awaiting a job |

> **Ordering matters.** `src/utils/config.py` reads these at **import time** and
> creates the directories. Set them before anything imports the application.
> `src.deploy.runtime.configure_runtime()` does exactly that and is what the
> notebooks call.

---

## Local

The canonical path, and the one in the README:

```bash
git clone https://github.com/gajjalaashok75-UI/video-translator.git
cd video-translator
python -m venv venv && source venv/bin/activate   # or venv\Scripts\activate
pip install -e ".[gpu-cu124,api,tts,translation-llm,audio-processing]"
python run.py --open
```

`run.py` checks the interpreter and the web dependencies, prints a banner, and
serves the API and the built UI on `http://127.0.0.1:8000`. It binds to loopback
by default: a local tool should not be reachable from the network unless asked.

Useful flags:

```bash
python run.py --check                       # run the checks, start nothing
python run.py --host 0.0.0.0 --port 8080    # expose on the network
python run.py --log-level debug
```

Environment equivalents: `API_HOST`, `API_PORT`, `API_LOG_LEVEL`,
`API_OPEN_BROWSER`.

If the frontend has not been built, the API still starts and `/` explains how to
build it:

```bash
cd frontend && npm install && npm run build
```

---

## Docker

A CUDA image with the full dependency set, for a server you control.

**Prerequisites:** an NVIDIA driver, Docker Engine (or Docker Desktop), and the
NVIDIA Container Toolkit so the container can see the GPU.

```bash
docker build -f deploy/docker/Dockerfile -t video-translator .
docker run --gpus all -p 7860:7860 \
  -v "$PWD/output:/app/output" \
  -v "$PWD/jobs:/app/jobs" \
  --name video-translator video-translator
```

Open `http://localhost:7860`. The two volume mounts matter: without them a
finished render and its job state live only inside the container and are lost
when it is removed.

To point the container at a different runtime tree, pass the variables:

```bash
docker run --gpus all -p 7860:7860 \
  -e JOBS_DIR=/data/jobs -e OUTPUT_DIR=/data/output -e TEMP_DIR=/data/temp \
  -v "$PWD/data:/data" \
  --name video-translator video-translator
```

**Port conflicts.** On Windows, Hyper-V and WinNAT reserve large port ranges and
7860 is sometimes among them. Map a different host port
(`-p 7950:7860`) and open that instead. To see what is reserved:

```powershell
netsh interface ipv4 show excludedportrange protocol=tcp
```

> The Docker image runs the Gradio UI (`app.py`), which is the historical entry
> point. To serve the FastAPI + React stack in a container instead, override the
> command with `python run.py --host 0.0.0.0 --port 7860`.

---

## Colab and Kaggle

Both notebooks live in [`deploy/notebooks/`](../deploy/notebooks/) and do the
same thing: provision a GPU box, point the runtime tree at a writable directory,
install the package, start the server, and print a URL.

- **Colab** — `deploy/notebooks/colab.ipynb`. Choose
  **Runtime → Change runtime type → T4 GPU**, then **Run all**. Colab proxies
  the port for you, so the printed link works with no account or token.
- **Kaggle** — `deploy/notebooks/kaggle.ipynb`. Turn on **Internet** in the
  notebook settings, choose a GPU accelerator, then **Run all**. Kaggle has no
  built-in port proxy, so to reach the UI from your browser set `PUBLIC_HOST` to
  a tunnel you run (Cloudflare Tunnel, ngrok, …) before starting the server.

Both notebooks are generated from one source, `scripts/build_notebooks.py`, so
they cannot drift. Edit that script, not the `.ipynb`:

```bash
python scripts/build_notebooks.py           # regenerate both notebooks
python scripts/build_notebooks.py --check   # verify they are up to date
```

The logic they share is in `src/deploy/runtime.py`, which is ordinary importable
code and is tested without a GPU (`tests/test_deploy_runtime.py`).

### Reaching the app

A notebook runs on a machine you are not sitting at, so `127.0.0.1:8000` there is
not reachable from your browser. The notebooks call
`runtime.notebook_public_url(port)`, which:

1. uses Colab's built-in port proxy when on Colab;
2. otherwise uses `PUBLIC_HOST` (and `PUBLIC_SCHEME`, default `https`) when set;
3. otherwise falls back to loopback and tells you it is VM-local.

---

## Configuration reference

Every knob is an environment variable. Defaults are the local-development values,
so nothing here needs to be set for a normal local run.

### Application

| Variable | Default | Meaning |
|----------|---------|---------|
| `HF_TOKEN` | unset | Hugging Face token; needed for diarization and some local translation models |
| `USE_CUDA` | `true` | Set `false` to force CPU even when a GPU is present |
| `NLLB_MODEL_ID` | `facebook/nllb-200-distilled-600M` | Local translation model |
| `NLLB_LOCAL_FILES_ONLY` | `false` | Never reach the network for the NLLB weights |
| `CHUNK_DURATION` | `300` | Long-video chunk length, seconds |
| `CHUNK_OVERLAP` | `2` | Overlap between chunks, seconds |

### Server

| Variable | Default | Meaning |
|----------|---------|---------|
| `API_HOST` | `127.0.0.1` | Bind interface (`0.0.0.0` to expose) |
| `API_PORT` | `8000` | Listen port |
| `API_LOG_LEVEL` | `info` | Uvicorn log level |
| `API_OPEN_BROWSER` | `false` | Open a browser once ready |
| `API_SERVE_FRONTEND` | `true` | Serve the built React app from this process |
| `API_FRONTEND_DIST` | `<repo>/frontend/dist` | Where the build is |
| `API_CORS_DEV_ORIGINS` | `true` | Allow the Vite dev-server origins |
| `API_MAX_UPLOAD_BYTES` | 4 GiB | Reject larger uploads |

### Storage

| Variable | Default |
|----------|---------|
| `TEMP_DIR` | `<repo>/temp` |
| `OUTPUT_DIR` | `<repo>/output` |
| `JOBS_DIR` | `<repo>/jobs` |
| `API_UPLOADS_DIR` | `$TEMP_DIR/uploads` |

### Notebook deployment

| Variable | Default | Meaning |
|----------|---------|---------|
| `RUNTIME_DIR` | platform-specific | Root for the four runtime directories |
| `PUBLIC_HOST` | unset | Hostname to show instead of loopback |
| `PUBLIC_SCHEME` | `https` | Scheme for `PUBLIC_HOST` |

---

## Choosing a quality preset

The preset is a hint about how much GPU the machine has. `runtime.hardware_report()`
recommends one from the available VRAM:

| VRAM | Preset | What it does |
|------|--------|--------------|
| < 8 GB, or CPU | `FAST` | No lip-sync |
| ≥ 8 GB | `BALANCED` | Wav2Lip lip-sync |
| ≥ 16 GB | `HIGH` | LivePortrait lip-sync + audio enhancement |

A preset only ever changes quality settings (lip-sync model, enhancement); it
never changes the language, the backends, or the text, so a preset can never
change *what the dub says*.

---

## Troubleshooting

| Symptom | Cause and fix |
|---------|---------------|
| `Python 3.13 is newer than the supported 3.12` | Use 3.10–3.12. Several pinned wheels have no 3.13 build. |
| `Missing Python dependencies: fastapi, uvicorn` | `pip install -e ".[api]"`. |
| `FFmpeg was not found on PATH` | Install FFmpeg and reopen the terminal. The pipeline shells out to it for every render. |
| `The server did not become ready` (notebook) | Usually a partial install; re-run the install cell and read the log above the error. |
| Out of GPU memory mid-render | Switch to `FAST` (no lip-sync) or process a shorter clip. |
| Frontend shows instructions instead of the UI | The React build is absent: `cd frontend && npm install && npm run build`. |
| Port already in use | Another run is holding it. Stop it, or start on another port. |
| The notebook URL does not open | It is VM-local. Re-run the start cell for a fresh proxy URL, or set `PUBLIC_HOST` to a tunnel. |
