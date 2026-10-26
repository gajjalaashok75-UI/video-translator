# 🌍 AI Video Translator (Local)

> **Break language barriers with cinema-quality video translation — privately, on your own hardware.**

Transform any video into a professional multilingual production with natural voice cloning, lip-sync, and on-screen text translation. No cloud APIs, no subscriptions, no data leaving your machine.

## 📝 Enjoying the project? Please star it ⭐️! It helps me gauge interest and keep working on new features.

---

## ✨ Why This Project?

**Traditional dubbing** is expensive, time-consuming, and requires professional studios. **AI Video Translator** democratizes video localization by bringing Hollywood-grade technology to your desktop:

- 🎬 **Content Creators**: Expand your audience globally without hiring voice actors
- 🎓 **Educators**: Make training content accessible in any language
- 📰 **Journalists & Documentarians**: Localize footage for international audiences
- 🎮 **Game Developers**: Dub cutscenes and trailers cost-effectively
- 🏢 **Businesses**: Translate corporate videos, presentations, and webinars
- 🔒 **Privacy-Focused Users**: Keep sensitive content 100% local

---

## 🎯 What It Does

Upload a video, select your target language, and let the AI handle everything:

```
📹 Input Video (English) → 🤖 AI Pipeline → 📹 Output Video (French, with cloned voice & synced lips)
```

**The full pipeline includes:**
1. **Vocal Separation** — Isolates speech from music/sound effects
2. **Transcription** — Converts speech to text with word-level precision
3. **Translation** — Translates text using local LLMs or Google Translate
4. **Voice Cloning** — Regenerates speech in the target language with the original speaker's voice
5. **Lip-Sync** — Adjusts mouth movements to match the new audio
6. **Visual Text Translation** — Detects and replaces on-screen text (subtitles, signs, etc.)
7. **Audio Enhancement** — Cleans and restores generated speech for broadcast quality

---

## 🚀 Key Features

### Audio Intelligence
| Feature | Technology | Description |
|---------|------------|-------------|
| **Vocal Separation** | HDemucs (Meta) | Cleanly separates speech from background music/sfx with GPU chunking for long videos |
| **Transcription** | Faster-Whisper (Large v3 Turbo) | 30-50% faster with Silero VAD preprocessing and word-level confidence filtering |
| **Speaker Diarization** | PyAnnote / SpeechBrain | Identifies individual speakers for multi-voice dubbing |
| **EQ Spectral Matching** | Custom | Applies original voice tonal characteristics to TTS output |
| **Voice Enhancement** | VoiceFixer | Restores degraded speech and removes noise (optional) |

### Translation Engine
| Model | Type | Best For |
|-------|------|----------|
| **NLLB-200** | Local | Default. Direct Indic↔Indic and 200 languages, no pivot |
| **Tencent HY-MT1.5** | Local (1.8B) | Better context preservation |
| **Llama 3.1 8B Instruct** | Local | Nuanced, human-like translations |
| **ALMA-R 7B** | Local | State-of-the-art translation quality |
| **Google Translate** | Online | Fast, reliable everyday translation |

All local models support **context-aware mode** using full-transcript context for superior coherence.

### Voice Synthesis (TTS)
| Model | Type | Highlights |
|-------|------|------------|
| **IndicF5** | Local | Reference-cloning for 11 Indian languages (As Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu, Assamese); exact-duration `fix_duration` for A/V sync |
| **Edge-TTS** | Online | Natural Microsoft voices, zero GPU needed |
| **Piper TTS** | Local | Robust offline neural TTS (auto-downloaded) |
| **XTTS-v2** | Local | High-fidelity voice cloning with emotion control (Happy, Sad, Angry) |
| **F5-TTS** | Local | Ultra-fast zero-shot voice cloning with Sway Sampling |
| **VibeVoice** | Local | Microsoft's frontier long-form multi-speaker TTS (1.5B/7B) |

### 🇮🇳 Indian Languages
IndicF5 (`ai4bharat/IndicF5`) is the default voice engine for all 11 supported Indian
languages. It clones the **original speaker's voice** from a clean reference clip and
keeps generated speech inside the original dialogue window, so the dub stays
lip-sync-ready without stretching the audio.

| Language | Code | Cloned voice | Script |
|----------|------|--------------|--------|
| Assamese | `as` | IndicF5 | Assamese |
| Bengali | `bn` | IndicF5 | Bengali |
| Gujarati | `gu` | IndicF5 | Gujarati |
| Hindi | `hi` | IndicF5 | Devanagari |
| Kannada | `kn` | IndicF5 | Kannada |
| Malayalam | `ml` | IndicF5 | Malayalam |
| Marathi | `mr` | IndicF5 | Devanagari |
| Odia | `or` | IndicF5 | Odia |
| Punjabi | `pa` | IndicF5 | Gurmukhi |
| Tamil | `ta` | IndicF5 | Tamil |
| Telugu | `te` | IndicF5 | Telugu |

Translation for these languages goes **directly** Indic→Indic through NLLB-200 —
no pivot through English — so the meaning is preserved in one pass.

> [!NOTE]
> IndicF5 is gated on Hugging Face. Set `HF_TOKEN` before rendering or these
> languages fall back to Edge TTS (works, but without a cloned voice).

### Visual Enhancements
| Feature | Technology | Description |
|---------|------------|-------------|
| **Lip-Sync (Fast)** | Wav2Lip-GAN | Smooth, blended lip synchronization |
| **Lip-Sync (HD)** | Wav2Lip + GFPGAN | Face restoration eliminates blurriness |
| **Lip-Sync (Cinema)** | LivePortrait | State-of-the-art cinematic lip sync with natural facial animation |
| **Visual Text Translation** | PaddleOCR / EasyOCR | Detects and replaces on-screen text with OpenCV inpainting |

## 🌐 Supported Languages

Video Translator supports a wide range of languages for both source and target
translation. Indian languages get a **cloned voice** via IndicF5; everything else
uses the selected TTS backend.

| Language | Code | Cloned |
|:---|:---:|:---:|
| **Auto Detect** | `auto` | — |
| **English** | `en` | XTTS/F5 |
| **Spanish** | `es` | XTTS/F5 |
| **French** | `fr` | XTTS/F5 |
| **German** | `de` | XTTS/F5 |
| **Italian** | `it` | XTTS/F5 |
| **Portuguese** | `pt` | XTTS/F5 |
| **Polish** | `pl` | XTTS/F5 |
| **Turkish** | `tr` | XTTS/F5 |
| **Russian** | `ru` | XTTS/F5 |
| **Dutch** | `nl` | XTTS/F5 |
| **Czech** | `cs` | XTTS/F5 |
| **Arabic** | `ar` | XTTS/F5 |
| **Chinese (Simplified)** | `zh` | XTTS/F5 |
| **Japanese** | `ja` | XTTS/F5 |
| **Korean** | `ko` | XTTS/F5 |
| **Hindi** | `hi` | **IndicF5** |
| **Assamese** | `as` | **IndicF5** |
| **Bengali** | `bn` | **IndicF5** |
| **Gujarati** | `gu` | **IndicF5** |
| **Kannada** | `kn` | **IndicF5** |
| **Malayalam** | `ml` | **IndicF5** |
| **Marathi** | `mr` | **IndicF5** |
| **Odia** | `or` | **IndicF5** |
| **Punjabi** | `pa` | **IndicF5** |
| **Tamil** | `ta` | **IndicF5** |
| **Telugu** | `te` | **IndicF5** |

### Production-Ready
- 🖥️ **Friendly Gradio UI** — Easy drag-and-drop interface
- 🎛️ **Fine-Grained Control** — Beam size, VAD settings, voice selection, and more
- 🖼️ **Visual Text Translation** — Detects, translates, and seamlessly replaces text in video frames (cached for speed)
- 📝 **Auto-Generated Subtitles** — Exports `.srt` files alongside translated videos
- 🔄 **Smart Segment Merging** — Combines choppy phrases into natural sentences
- ⏳ **Real-time Progress & ETA** — Track detailed progress with estimated time remaining
- 🎤 **Speaker Voice Cloning** — One clean reference clip per speaker, reused across all their lines
- ⚡ **GPU Optimized** — One-model-at-a-time policy for maximum VRAM efficiency
- 🛡️ **Global CPU Fallback** — Automatically switches to CPU if GPU fails
- 📦 **Durable Jobs** — Atomic manifest + content-addressed cache, so a render survives a crash and resumes without redoing finished stages

---

## 🎬 Use Cases

### 🎥 YouTube & Social Media Creators
> "I have 50 English tutorials and want to reach Spanish speakers."

Upload each video, select English → Spanish, and export professional dubs with your cloned voice. No re-recording needed!

### 🎓 Corporate Training & E-Learning
> "Our compliance training is in English but we have offices in 12 countries."

Batch-translate training videos while maintaining the presenter's voice for authenticity. Export with or without subtitles.

### 🎞️ Film & Documentary Localization
> "I want my indie film to premiere at international festivals."

Use LivePortrait (HD) lip-sync for cinema-quality dubbing that doesn't look like a bad overdub.

### 📢 Marketing & Advertising
> "We need our 30-second ad in French, German, and Japanese by tomorrow."

Process multiple language versions simultaneously with local LLM translation for brand-appropriate messaging.

### 🔐 Sensitive Content Translation
> "Our video contains confidential product demos."

Everything runs locally — no data leaves your machine. Perfect for legal teams, medical content, or proprietary information.

---

## 🛠️ Prerequisites

| Requirement | Details |
|-------------|---------|
| **Python** | 3.10–3.12 (3.11 recommended) |
| **PyTorch** | 2.5.1+ with CUDA 12.4+ |
| **GPU** | NVIDIA GPU recommended (RTX 30/40/50 series supported) |
| **VRAM** | 8GB minimum, 12GB+ recommended for HD lip-sync |
| **FFmpeg** | Must be in system PATH |
| **Rubberband** | Recommended for high-quality audio time-stretching |

> **Deploying somewhere else?** Colab, Kaggle, Docker and remote-server
> instructions live in **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)**.

<details>
<summary><strong>📥 FFmpeg Installation</strong></summary>

**Windows (Option 1):**
```powershell
winget install ffmpeg
# Restart terminal after installation
```

**Windows (Option 2 - Manual):**
1. Download from [ffmpeg.org/download](https://ffmpeg.org/download.html) (Windows builds → gyan.dev)
2. Extract to `C:\ffmpeg`
3. Add `C:\ffmpeg\bin` to your system PATH
4. Restart terminal and verify: `ffmpeg -version`

**Linux:**
```bash
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```
</details>

<details>
<summary><strong>📥 Rubberband Installation</strong></summary>

Download from [Rubberband Releases](https://breakfastquay.com/rubberband/). Extract and add to PATH, or place `rubberband-program.exe` in the project folder.
</details>

---

## 📦 Installation

### 🚀 Quick Install (Recommended)

The smart installer auto-detects your platform, GPU, and CUDA version:

```bash
# Clone the repository
git clone https://github.com/gajjalaashok75-UI/video-translator.git
cd video-translator

# Windows:
scripts\install.bat

# Linux / macOS:
chmod +x scripts/install.sh
./scripts/install.sh
```

### 🔧 Manual Install

<details>
<summary><strong>Windows (CUDA 12.8 — RTX 30/40/50 series)</strong></summary>

```powershell
py -3.10 -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```
</details>

<details>
<summary><strong>Linux (CUDA 12.4)</strong></summary>

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r deploy/docker/requirements.docker.txt
```
</details>

<details>
<summary><strong>macOS (Apple Silicon — CPU only)</strong></summary>

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

> [!NOTE]
> Some GPU-accelerated features have reduced performance on Apple Silicon. Platform-specific
> package substitutions (`paddlepaddle` for `paddlepaddle-gpu`, `onnxruntime` for `onnxruntime-gpu`)
> are handled automatically by platform markers in the requirements files.
</details>

<details>
<summary><strong>Advanced: Using pyproject.toml extras</strong></summary>

```bash
# Full install with CUDA 12.4 (default):
pip install -e ".[all]"

# Full install with CUDA 12.8 (RTX 50 series):
pip install -e ".[all-cu128]" --extra-index-url https://download.pytorch.org/whl/cu128

# CPU only:
pip install -e ".[all-cpu]"

# Minimal (core + dev tools only):
pip install -e ".[dev]"

# Selective install (pick what you need):
pip install -e ".[tts,ocr,lipsync]"
```
</details>

### ✅ Verify Installation

```bash
# Run smoke tests to validate your install
pytest tests/test_smoke_install.py -m smoke -v

# Or use the installer's verify mode
python scripts/install.py --verify
```

### Optional Components

| Feature | Requirement |
|---------|-------------|
| **NeMo Diarization** | `nemo_toolkit[asr]` |
| **Wav2Lip** | Model file at `models/wav2lip/wav2lip_gan.pth` |
| **F5-TTS** | `f5-tts` package (GPU recommended) |
| **Enhanced Lip-Sync** | `gfpgan` and `basicsr` (included in requirements) |
| **LivePortrait** | ~2GB VRAM, auto-downloads to `models/live_portrait` |
| **Llama 3.1 / NeMo** | HuggingFace token (`HF_TOKEN` env variable) |
| **VoiceFixer** | `pip install voicefixer>=0.1.2` (may need pip < 24.1) |

### 🐳 Docker Deployment (GPU)

You can run the application in a Docker container with NVIDIA GPU support.

**Prerequisites:**
- **NVIDIA Driver** (compatible with CUDA 12.x)
- **Docker Desktop** (Windows) or Docker Engine (Linux)
- **NVIDIA Container Toolkit** (for GPU access inside Docker)

#### Build and Run (Recommended)

1.  **Build the image** (run from the project root):
    ```bash
    docker build -f deploy/docker/Dockerfile -t video-translator .
    ```

3.  **Run the container**:
    ```bash
    # For PowerShell:
    docker run --gpus all -p 7860:7860 -v ${PWD}/output:/app/output --name video-translator video-translator

    # For Command Prompt (CMD):
    docker run --gpus all -p 7860:7860 -v %cd%/output:/app/output --name video-translator video-translator
    ```

    > **Note:** If you encounter a "Ports are not available" error (common on Windows with Hyper-V), try mapping to a different port like **7950**:
    > ```bash
    > docker run --gpus all -p 7950:7860 -v ${PWD}/output:/app/output --name video-translator video-translator
    > ```

4.  **Access the App**:
    Open your browser to `http://localhost:7860` (or `http://localhost:7950` if you used the alternate port).

#### 🔧 Troubleshooting

**"Ports are not available" / "Access is denied"**
On Windows, Hyper-V or WinNAT often reserves large ranges of ports (including 7860).
- **Solution:** Use a different host port (like 7950 or 8080) as shown in the note above.
- **Check reserved ranges:** Run `netsh interface ipv4 show excludedportrange protocol=tcp` to see which ports are blocked.

---

### ☁️ Hosted Deployment (Colab / Kaggle)

No GPU at home? Run the same application on a free GPU host and open it in your
browser. Two notebooks are provided:

| Platform | Notebook | Notes |
|----------|----------|-------|
| **Google Colab** | [`deploy/notebooks/colab.ipynb`](deploy/notebooks/colab.ipynb) | Select a GPU runtime, then Run all. The URL is proxied automatically. |
| **Kaggle** | [`deploy/notebooks/kaggle.ipynb`](deploy/notebooks/kaggle.ipynb) | Enable **Internet** and a GPU accelerator, then Run all. |

Each notebook checks the machine, redirects the runtime directories to a
writable location, installs the package, starts the server, and prints a URL.

The notebooks are generated from one source, so they cannot drift apart:

```bash
python scripts/build_notebooks.py           # regenerate both notebooks
python scripts/build_notebooks.py --check   # verify they are current
```

Full details — runtime directories, every environment variable, and
troubleshooting — are in **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)**.

---

## 🖥️ Usage

### Quick Start

```bash
# Activate environment
.\venv\Scripts\activate

# Launch the application
python app.py
```

Open your browser to `http://127.0.0.1:7860`

### Step-by-Step Translation

1. **Upload Video** — Drag & drop MP4, MKV, or MOV files
2. **Select Languages** — Source (or Auto-detect) → Target
3. **Choose Models:**
   - **Translation**: NLLB-200 (default, local) / HY-MT / Llama 3.1 / ALMA-R / Google
   - **TTS**: IndicF5 (Indian languages, cloned) / Edge / XTTS / F5 / Piper
4. **Enable Features** (optional):
   - ✅ Speaker Diarization — Multi-speaker videos (needs `HF_TOKEN`)
   - ✅ Lip-Sync — Select quality level (Fast/HD/Cinema)
   - ✅ Visual Text Translation — Replace on-screen text
   - ✅ Audio Enhancement — VoiceFixer post-processing
5. **Click "Process Video"** and monitor progress

---

## ⚙️ Configuration

### Directory Structure
```
video-translator/
├── temp/           # Intermediate files (auto-cleaned)
├── output/         # Final translated videos
├── models/         # Downloaded model weights
└── .env            # Environment variables (HF_TOKEN, etc.)
```

### Environment Variables
```env
HF_TOKEN=your_huggingface_token  # Required for Llama 3.1, NeMo
```

---

## 🧩 Pipeline Architecture

```mermaid
flowchart TD
    Video[Input Video] --> Extract[Extract Audio via FFmpeg]
    Extract --> Separator[Audio Separator - HDemucs]

    Separator --> Vocals[Vocal Track]
    Separator --> Background[Background Track]

    Vocals --> VAD[VAD Preprocessing - Silero VAD]
    VAD --> Transcribe[Transcribe - Faster-Whisper]
    Transcribe --> Segments[Text Segments]
    Segments --> Merge[Smart Segment Merging]

    Vocals -.-> Diarize[Diarize - PyAnnote / SpeechBrain]
    Diarize -.-> SpeakerProfiling[Speaker Profiling]

    Merge --> Translate[Translate - NLLB / HY-MT / Llama / ALMA]
    Translate --> SRT[Export SRT Subtitles]

    Translate --> TTS[Neural TTS - IndicF5 / XTTS / Edge / F5]
    SpeakerProfiling -.-> TTS
    TTS -> TTSAudio[Generated Speech Clips]

    TTSAudio --> EQ[EQ Spectral Matching]
    EQ --> Sync[Synchronize - PyRubberband]
    Sync --> MergedSpeech[Merged Speech Track]

    MergedSpeech -.-> VoiceEnhance[Voice Enhancement - VoiceFixer]
    VoiceEnhance -.-> Mix
    MergedSpeech --> Mix[Mix Audio]
    Background --> Mix

    Mix --> FinalAudio[Final Audio Track]

    Video --> VisualTrans[Visual Translation - PaddleOCR / EasyOCR]
    VisualTrans --> LipSync[Lip-Sync - LivePortrait / Wav2Lip / GFPGAN]
    MergedSpeech -.-> LipSync

    LipSync --> Mux[Mux with Video - FFmpeg]
    FinalAudio --> Mux

    Mux --> Output[Translated Output Video]
```

---

## 🤝 Contributing

Contributions are welcome! Areas where help is appreciated:
- Additional language support and voice models
- Performance optimizations
- Bug fixes and stability improvements
- Documentation and tutorials

---

## 🏷️ Citation

If you find this project useful, please cite it as follows:

```bibtex
@software{video_translator,
  author = {Ashok Gajjala},
  email = {gajjalaashok75@gmail.com},
  title = {AI Video Translator},
  url = {https://github.com/gajjalaashok75-UI/video-translator},
  version = {1.0.0},
  date = {2026-06-15}
}
```

---

## 📄 License

This project is for educational and personal use. Please respect the licenses of underlying models and technologies.

---

<div align="center">

**🌟 Star this repo if you find it useful! 🌟**

*Made with ❤️ for content creators worldwide*

</div>
