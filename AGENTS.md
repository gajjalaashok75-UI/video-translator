# AGENTS.md

Repository-specific knowledge for agents working on this codebase.

## Project

`video-translator` (gajjalaashok75-UI/video-translator) — a Gradio app that dubs a
video into another language: separate vocals, transcribe, diarize, translate,
synthesize speech in each speaker's voice, then re-sync and optionally lipsync.

## Environment notes

- Python 3.13. This is newer than several pinned dependencies expect.
- `python -m pytest` is the reliable invocation. The user site-packages bin
  directory (`/home/openhands/.local/bin`) is not on `PATH`.
- `tests/conftest.py` mocks heavy dependencies (`torch`, `ctranslate2`, etc.)
  only when they are not installed. This is why a plain
  `python -c "import src.utils.config"` can fail with `ModuleNotFoundError:
  ctranslate2` while the same import works fine under pytest. Verify
  import-level behaviour through pytest, not a bare interpreter.
- Install test/runtime deps with `pip install --user`; the environment is not
  writable system-wide in some sandboxes.

### Pre-existing test failures (not caused by your changes)

Establish a baseline with `git stash -u` before blaming yourself. As of
`8e91ebc` these fail in a CPU-only sandbox with no ffmpeg, and are all
environmental:

- `tests/test_utils.py::TestAudioUtils` (3) — libsndfile cannot open the
  generated temp WAV.
- `tests/test_tts_reference_fallback.py` (5) — depend on the same audio path.
- `tests/test_eq_matching.py` — collection error, scipy/torch array-api
  `issubclass()` incompatibility.
- `tests/test_smoke_install.py`, `test_smoke_pipeline.py` (ffmpeg version),
  `test_visual_translator.py`, `test_wav2lip_*`, `test_paddle_ocr_init.py`,
  `test_chunker.py` (needs ffmpeg).

Expected baseline: `9 failed, 150 passed, 30 skipped, 5 errors` when excluding
the three files above.


### Working CPU environment (validated end-to-end)

The system Python (3.13) is too new for the pinned stack. A working CPU
environment was built with `uv` at `.venv` (CPython 3.12.14):

- `uv venv --python 3.12 .venv`, then install per package with
  `uv pip install ...`. The full `requirements.txt` cannot resolve on 3.12:
  `pandas==1.5.3` fails to build (its build dep `pkg_resources` is gone).
  `pandas` is not imported anywhere in `src/`, so it is safe to omit.
- Pin `av==13.1.0`. `faster-whisper` 1.x calls
  `av.open(..., metadata_errors="ignore")`, which PyAV 14+ removed, so a newer
  `av` fails every transcription with
  `TypeError: open() got an unexpected keyword argument 'metadata_errors'`.
- `numpy<2` (1.26.4) is required by the older torchaudio/demucs stack.
- `torch`, `torchvision` and `torchaudio` must be installed as a matched
  release triple (`torch==2.5.1` / `torchvision==0.20.1` / `torchaudio==2.5.1`).
  A mismatched pair imports but fails at first use — e.g. `torchvision` newer
  than `torch` raises `RuntimeError: operator torchvision::nms does not exist`,
  and `torchaudio` newer than `torch` raises
  `undefined symbol: aoti_torch_abi_version`. `tests/test_smoke_install.py`
  catches both.
- `transformers` must stay `<4.52` while the CPU torch floor is 2.5.1.
  transformers 5.x requires `torch>=2.6` and 4.52.0 added a `torch.load()`
  CVE gate (CVE-2025-32434) that refuses to run on `torch<2.6`;
  `nllb-200-distilled-600M` ships only `pytorch_model.bin` (no safetensors),
  so either change silently disables the local NLLB backend
  (`TranslationRouter` then falls back to online Google). `4.51.3` is the
  newest release verified to load and translate NLLB on torch 2.5.1. Raise the
  cap only together with the torch floor.
- `pandas` is only referenced by `tests/test_smoke_install.py`, which accepts a
  major version of 1 or 2; the `pandas<2.0.0` pin cannot be satisfied on 3.12
  (see above), so 2.x is what this environment carries.
- ffmpeg/ffprobe come from apt. Their presence matters: `jobs.validate_video`
  switches from a size fallback to a real binary check, which changes which
  artifacts count as valid (see `tests/test_phase6_resume.py`).

### WebM sources: mux container/codec bug (fixed)

`VideoProcessor.replace_audio` derives the container from the output filename
(the pipeline keeps the source extension) but used to hardcode `h264`/`aac`.
That combination cannot be muxed into WebM, so any `.webm` source failed with
`Nothing was written into output file, because at least one of its streams
received no packets` and `replace_audio` returned `None` — which the caller
then recorded as a *completed* `final_output`, marking the manifest COMPLETE
with no video on disk. Fixed by choosing codecs from the output extension
(`vp9`/`opus` for `.webm`) and raising instead of returning `None`; the caller
now fails the stage on a falsy result, and `resume_job` re-runs rather than
trusting a COMPLETE manifest whose output is missing. Regression coverage:
`tests/test_replace_audio.py` (real ffmpeg, no mocks).

## Architecture

Entry point `app.py` (Gradio UI) → `src/core/video_translator.py`
(`VideoTranslator.process_video` → `_process_pipeline`), which runs staged
generators in `src/core/pipeline_stages.py` and keeps state in
`src/core/session.py`.

- `src/audio/` — `separator.py` (AudioSeparator), `transcription.py`
  (faster-whisper), `diarization.py` (PyAnnote).
- `src/translation/text_translator.py` — `LLMTranslator` (default model
  `tencent/HY-MT1.5-1.8B`) plus the `Translator` protocol.
- `src/synthesis/tts.py` — `TTSEngine`, the backend registry and the
  `generate_batch` entry point. Backends live in `src/synthesis/backends/` and
  all implement `TTSBackend.generate(..., **kwargs)`.
- `src/processing/` — `synchronization.py` (AudioSynchronizer, the duration
  fitting), `video.py`, `lipsync.py`, `voice_enhancement.py`.
- `src/utils/` — `languages.py` (see below), `config.py` (paths, `DEVICE`,
  dotenv, all tunable thresholds), `srt_generator.py`, `audio_utils.py`.

## Language architecture (Phase 1)

`src/utils/languages.py` is the single source of truth. Add a language there
and it appears in the UI dropdowns automatically (`app.py` calls
`languages.source_choices()` / `target_choices()`); nothing else needs editing.

Each entry carries: `name`, `aliases`, `flores` (NLLB-200 code), optional
`translate_to` (when the speech engine cannot read the native script),
`engine` (dedicated cloning backend, or `None`), optional `synth_lang`, `cps`
(rough chars/sec, used only as a sanity bound), `edge` voices, optional
`piper`, and `source`.

Two contracts matter:

- `get_language_code()` is deliberately lenient — unknown input returns `"en"`
  rather than raising. Existing callers and tests depend on this.
- `normalize()` is the strict resolver (accepts `te`, `te-IN`, `te_IN`,
  `Telugu`) and raises `UnsupportedLanguage`. New code should use it.

Derived views `LANGUAGE_CODE_MAP`, `EDGE_TTS_VOICE_MAP`, `PIPER_MODEL_MAP` are
kept so pre-existing call sites keep working.

Routing: `TTSEngine.route_model(target_lang, requested_model)` decides the
backend from the target language. A language with a dedicated engine (IndicF5
for the 11 Indian languages) uses it; anything else keeps the requested general
backend. An explicit request for a backend that cannot speak the language falls
back with a warning instead of failing mid-render.

## IndicF5 (Phase 2)

`src/synthesis/indicf5_engine.py` owns the model;
`src/synthesis/backends/indicf5.py` is the thin `TTSBackend` adapter.
`ai4bharat/IndicF5` is gated on Hugging Face — it needs an accepted licence and
`HF_TOKEN`.

Four non-obvious things, each of which fails silently if missed:

1. Checkpoint keys are prefixed `ema_model._orig_mod.…` because the model was
   saved from a `torch.compile()`-wrapped module. A plain `from_pretrained`
   matches zero tensors and leaves the model randomly initialised, producing
   silent audio of exactly the right length. `_load_real_weights` strips the
   prefix and refuses to continue unless every tensor matched and the weights
   are finite and non-degenerate.
2. The model's own `forward()` hardcodes `device="cpu"` and strips silence /
   re-normalises loudness, fighting duration control. We bypass it and call
   `f5_tts.infer.utils_infer.infer_batch_process` directly.
3. `torchaudio.load` on torchaudio ≥ 2.11 demands TorchCodec. Passing a
   pre-loaded tensor avoids that path; `src/utils/torch_compat.py` repairs
   `torchaudio.load` for libraries that call it internally (Coqui XTTS does).
4. `ref_text` (the reference clip's transcript) is required for a good clone.
   An empty one does not raise in the model, it just produces a worse voice, so
   the engine raises unless the caller sets `allow_degraded_ref_text=True`.

Duration control: `natural_duration()` mirrors the model's own byte-based
heuristic so a request that matches what the model already wants avoids
internal compression. `fix_duration` asks for a specific total length, which is
far better than stretching the output afterwards. Calibration uses the
reference's *voiced* fraction, not its wall time, so the clip's own pauses do
not make every byte look slower than it is; `clamp_seconds_per_byte` rejects
implausible rates.

## Gotchas

- `TTSEngine._sanitize_text` discards any text with no character from
  `_SPEAKABLE_CHARS`. That class must cover every script we speak. It
  originally omitted all Brahmic scripts, so every Indic segment was silently
  dropped and the dub came out silent with no error. Adding a language means
  checking this class.
- Several source files have trailing whitespace on lines. `file_editor`
  `str_replace` will fail to match blocks spanning those lines; patch with a
  line-based Python script instead.
- The batch TTS path puts `ref_text` and `fix_duration` in every task dict. All
  backends take `**kwargs` for this reason. A new backend that does not will
  break every language, not just Indic ones.
- Translation source and target are independent. `source_flores()` accepts
  source-only languages (`EXTRA_SOURCE_FLORES`, e.g. Urdu) that
  `target_flores()` deliberately rejects.
- `src/utils/config.py` imports `ctranslate2` at module scope and is not
  guarded. Any `python -c "from src.utils import config"` outside pytest fails
  with `ModuleNotFoundError` unless ctranslate2 is installed; conftest mocks it.
- "The stage ran" is not "the stage produced output". A translation backend
  that fails every line returns a full-length list of empty strings instead of
  raising; the pipeline used to write that list, mark the stage complete, dub
  nothing, and still report `COMPLETE`. Three guards now enforce the real
  success conditions before a stage is recorded: `_translation_invariant_error`
  (speech in => usable translation out), `_tts_invariant_error` (dialogue in =>
  speech out), and `_final_invariant_error` (last resort before `COMPLETE`).
  Do not weaken these to a "did the file get written" check. A speechless
  source is not an error at any of them. `"hi"` in `target_language` is Hindi
  (the ISO 639-1 code), not the greeting.

## Speaker references (Phase 3)

`src/audio/speaker_reference.py` picks one clean reference clip per diarized
speaker and returns its transcript.

The ordering constraint is the whole reason it lives where it does:
`_step_diarization` (Stage 3) runs *before* transcription, so the profiles it
extracts have no transcript and cannot be used as cloning references on their
own. `build_speaker_references` therefore runs in Stage 4b, after speaker
assignment and merging, and overrides `speaker_profiles` for speakers where it
found something better. It is called once per video; every segment reuses the
result.

Selection is two-stage on purpose. Transcript-only signals (word confidence,
speech density, internal gaps, other-speaker overlap, position) rank every
candidate cheaply; only the top `REFERENCE_AUDIO_PROBE_TOP_K` per speaker get a
waveform probe for loudness, clipping and silence. A two-hour film costs a
handful of audio reads instead of thousands.

Things to not break:

- Candidates are built only from words whose `speaker_id` matches, which is what
  stops a reference containing two voices. `TestAdjacentSpeakerContamination`
  fails if that filter is removed.
- Windows overlapped by another speaker by more than
  `REFERENCE_OVERLAP_REJECT` are refused, not merely penalised. Diarization does
  not separate voices, so overlapped audio is not usable — do not claim
  separation here.
- Scoring must not collapse to "longest wins". `TestDoesNotSimplyPickTheLongest`
  pins that.
- `_resolve_speaker_ref_text` drops the transcript when `_apply_reference_fallback`
  substituted different audio. A wrong transcript is worse than a missing one:
  it breaks the model's duration model as well as the voice.
- Extraction is timing-preserving (DC removal, edge fades, peak normalisation).
  Never add silence stripping — it would desynchronise the clip from `ref_text`.
  `test_clip_does_not_change_the_timeline` pins this.

## Translation architecture (Phase 4)

`src/translation/text_translator.py` holds two classes. `LLMTranslator` wraps one
causal LM (`tencent/HY-MT1.5-1.8B` default, or Llama-3.1-8B / ALMA-7B-R) and
builds a per-model prompt; `Translator` is the facade the pipeline calls
(`translate_segments`). `src/utils/languages.py` is the registry (Phase 1);
`flores` codes there are NLLB-200 / FLORES-200 codes.

Backends today: `google` (deep-translator), `hymt`, `llama`, `alma`. Selection
is by substring match on the UI model name inside `_step_translation`
(`"HY-MT" in name` -> `hymt`, etc.), so a new backend needs a branch there.

`translate_segments(segments, target_lang, model=..., source_lang=...)`:
- cache key is `"{src}|{tgt}|{text}|{model}"` in `TEMP_DIR/translation_cache.json`
- `optimize=True` runs `translate_context_aware` **sequentially**, threading only
  `prev_text` (previous *translated* line); `optimize=False` is stateless batch
  translation of chunked texts. There is no next-line or speaker context.
- segments are plain dicts; every backend returns a copy with `translated_text`
  added. `start`/`end`/`speaker_id` are carried through untouched.
- LLM backends are unloaded at the end of each `translate_segments` call.

Non-obvious:
- The source code comes from `config.get_language_code(source_lang)` which is
  lenient and returns `"en"` for anything unknown, so a mis-set source silently
  becomes English. Whisper's detected language is available in the pipeline
  (`detected_lang`) and already replaces `"auto"`.
- `source_code == target_code` short-circuits in `_step_translation` (copies the
  text through, no backend call).
- The IndicF5 TTS path needs the target in the *native* script; the registry's
  `target_flores(..., for_speech=True)` is the authority, and `translate_to`
  exists for Urdu (spoken from Devanagari).
- `TTSEngine.route_model` is the precedent for a language-driven router. Phase 4
  follows it for translation (`src/translation/router.py`).

### Why NLLB-200, not IndicTrans2 (Phase 4 investigation)

IndicTrans2 is gated on HF (401 without an accepted licence + token), ships as
2-3 separate checkpoints (en-indic, indic-en, indic-indic) that must all be
downloaded, needs `IndicTransToolkit` + `indicnlp` + `mosestokenizer` for
pre/post-processing (script unification to Devanagari, entity masking), and its
"Indic-Indic" model is a stitched encoder/decoder fine-tuned on synthetic
pivoted data (~1 chrF++ below pivoting via English). `facebook/nllb-200-*` is
ungated, one checkpoint, pure `transformers` (`M2M100ForConditionalGeneration`),
200 languages including all 11 priority Indic ones, and translates Indic-Indic
*directly* in one pass. NLLB-200 is CC-BY-NC-4.0 (non-commercial); IndicTrans2
is MIT. For a local movie-dubbing tool NLLB is the better fit; IndicTrans2 stays
an option behind the same router interface if a commercial licence is needed.

Phase 4 additions: `src/translation/router.py` (capability-based route choice)
and `src/translation/nllb_backend.py` (local NLLB-200). `Translator` now returns
each segment with a `translation` metadata block (`path`, `backend`, `pivot`,
`status`, `error`, `concise`, `context_applied`) and records unrecoverable
segments in `Translator.last_failures` instead of writing blank text.

Known limitation: NLLB-200 needs the source in its own script. Romanised input
(e.g. Hinglish "yaar yeh movie acchi hai" tagged as `hi`) passes through
unchanged because the tokenizer sees Latin text under a Devanagari source tag.
Whisper emits native script for Indic languages, so normal ASR input is fine;
this only bites hand-typed or transliterated text.

## Duration-aware timing (Phase 5)

Goal: generated speech should fit the ORIGINAL dialogue window naturally, and
the fix for a mismatch must not be "stretch the audio harder".

### One timing record, one owner
`src/processing/timing.py` defines `SegmentTiming`. `target_duration` is a
PROPERTY derived from `start`/`end` — it cannot be assigned, so no stage can
silently retarget a segment. `final_start` is always the original `start`: the
old `merge_segments` shifted starts forward to avoid overlap and clamped that
shift at 1.5s, which destroyed intentional overlap and let the timeline drift.
It also corrected against the NEXT SEGMENT'S START rather than the window, and
ran a second correction pass with a different clamp than `sync_segment`
([0.8, 1.35] vs [0.25, 4.0]) — so one line could be stretched twice.

### The layered strategy (cheapest first)
1. **Duration-aware translation** (`src/translation/duration_aware.py`). NLLB
   cannot be told to be concise, so shortening is a separate pass AFTER
   translation, triggered only when a line is predicted to overrun by more than
   `config.TIMING_SHORTEN_TRIGGER`. Every candidate is validated: rejected if it
   changes script, is not shorter, drops numerals, or contains words that were
   never in the original. A rejection keeps the original translation, so the
   worst case is "no improvement", never wrong dialogue. The original stays in
   `segment["translation"]["original_translated_text"]`.
2. **Native duration fit** (`src/processing/duration_model.py`). IndicF5's
   `fix_duration` is the only exact-duration control in the project and is
   accurate to ~10 ms, so it is asked for a length instead of stretching after.
   **`fix_duration` is the TOTAL (reference + generated) length** — a request
   must add the speaker's reference duration or every segment comes back one
   reference too long. See `measure_reference_durations()`. Only backends that
   can honour a duration are asked (`src/synthesis/capabilities.py`); everyone
   else gets `fix_duration=None`.
3. **Measurement** (`TimingCorrector.correct`). One decision per segment, from
   the ratio `generated/target` and its band.
4. **Bounded stretch** (`apply_stretch`). Rubberband if installed, else librosa,
   else nothing (reported as `uncorrected`, never as success). Clamped to
   `[TIMING_MIN_STRETCH, TIMING_MAX_STRETCH]`.
5. **Explicit fallback**. An overrun may extend into the following pause
   (`fit="extended"`, not a defect); only reaching the next speaker is cut
   (`fit="collision"`, reported as degraded).

### Two rules worth not breaking
- **Never stretch to fill.** A line that comes back SHORTER than its window is
  left alone; the gap becomes a pause, which is the original's own rhythm.
  Slowing speech to fill silence is what makes dubbing sound drugged.
- **Never correct twice.** A backend that fitted the audio sets
  `duration_honoured`, and the merge stage then skips it entirely. Without this
  the model's own timing work is undone by a phase vocoder.

### Diagnostics
`self.timing_diagnostics` (a `TimingDiagnostics`) is filled during the merge and
its `summary()` is stored on `session.timing_diagnostics` at the end of a
render: counts by action and backend, plus the worst ratio. Chunked processing
folds each chunk's records into a run-level report (`TimingDiagnostics.merge`).

### Tests
`tests/test_timing_model.py` (model/bands/corrector), `tests/test_duration_aware.py`
(shortening + validation), `tests/test_timing_sync_integration.py` (real-WAV
merge, correction behaviour, `fix_duration` wiring).

### Known limitations
- Layer 1 ships with the LLM path INACTIVE: `SpeechShortener()` is constructed
  without a model id, so only the rule-based filler removal runs. The filler
  lists are small, so most overruns fall through to the stretch layer. Wiring a
  model id is the highest-value follow-up; the plumbing and validation are
  already in place.
- Band thresholds and stretch clamps are heuristic, chosen from the ratio ranges
  the old code tolerated plus listening checks on the librosa path. They are not
  the result of a formal listening test.
- `src/processing/synchronization.py::sync_segment` is the older single-segment
  helper. It is no longer called by the pipeline (the merge path replaces it)
  and still probes with `ffmpeg` rather than `ffprobe`.
- Long overruns that exceed the stretch clamp are trimmed, which loses the tail
  of a line. This is reported as degraded rather than hidden, but it is still a
  loss; better translation (layer 1) is the real fix.

## Durable jobs (Phase 6)

Goal: a render that takes four hours must survive a crash, a cancel, or a
closed laptop at hour three, and resume without redoing hours one and two.

### The three pieces
- `src/core/jobs.py` — a job is a directory `jobs/<job_id>/` with a fixed set of
  subdirectories (`JOB_SUBDIRS`). All writes go through `atomic_write_*`, which
  writes to a temp file in the same directory then `os.replace`s it, so a reader
  never sees a half-written file and a crash leaves the old one intact. Every
  path is checked with `assert_within` before use.
- `src/core/manifest.py` — `manifest.json` is the authoritative record: source,
  configuration, per-stage status + input fingerprint, per-segment records, and
  (for chunked runs) per-chunk records. Written by `save()`, which keeps the
  previous copy as a backup, so a corrupt primary can be recovered.
- `src/core/cache.py` — content-addressed artifacts. **Never key on segment
  index.** `tts_key(...)` hashes text + language + model + reference identity +
  duration request, so a changed line misses the cache and an unchanged one
  hits. `reference_identity()` hashes the reference's path/size/mtime_ns.
- `src/core/retry.py` — `classify()` sorts a failure into TRANSIENT /
  SEGMENT_LOCAL / FATAL and `run_with_retry` retries only the first two.
  `CancelledError` is re-raised untouched, never retried.

### Resume is just a re-run
`VideoTranslator.resume_job(job_id)` loads the manifest, replays its recorded
`config` (filtered against `process_video`'s signature — against the CLASS's, so
an instance double cannot hide the real parameters), and calls `process_video`
with the same `job_id`. Stage fingerprints then decide what is skipped. There is
no per-stage resume logic, which is why a stage added later resumes correctly
for free. `hf_token` is deliberately never persisted, so a gated model needs it
passed to `resume_job` again.

### Rules worth not breaking
- **Chunks and segments are different namespaces.** They are both numbered from
  zero. `Manifest.chunks` is separate from `Manifest.segments`, and
  `_chunk_index` namespaces stage fingerprints and artifact paths per chunk.
  Sharing one dict lets chunk 3 overwrite speech segment 3.
- **Validate before trusting an artifact.** `jobs.validate_wav` checks the RIFF
  header's declared size against the file size, because libsndfile silently
  clamps a truncated file's frame count instead of failing. `is_silent` catches
  a backend that returned well-formed silence rather than raising.
  `ArtifactCache.put_tts` refuses both, so a bad result cannot become permanent.
- **A cancelled job is resumable, not failed.** `_check_cancel` sets status
  `CANCELLED` and marks the in-flight stage `PARTIAL`. `resume_job` refuses only
  a `COMPLETE` job.
- **Partial is a real outcome.** One failed TTS line is recorded and skipped;
  the merge lays silence there. Only a total TTS failure raises.
- `atomic_write_json` uses `sort_keys=True`, so identical state writes identical
  bytes and a diff shows what actually changed.
- **A job id embeds its creation time, and that is what orders it.** An id is
  `<source-stem>-YYYYMMDD-HHMMSS-<hash>` with a UTC stamp. `jobs.job_sort_key`
  parses the stamp from the right (the stem may itself contain hyphens) so
  `list_jobs` orders by real creation time. Never sort job ids as plain strings:
  the source filename would dominate, and with the API's list truncation
  (`list_jobs()[:scan_limit]`) a just-finished render can fall off `/api/jobs`.

### Environment note
`python -m pip install --user -r requirements.txt` can fail on `pandas` (needs
`setuptools` for `pkg_resources`) and the tree needs `numpy<2`, `pillow`,
`soundfile`, `librosa`. Install those individually if the bulk install breaks.

### Tests
`tests/test_phase6_jobs.py` (job/manifest/cache/retry units),
`tests/test_phase6_resume.py` (TTS cache reuse across a simulated restart, with
a real filesystem and a counting TTS stand-in),
`tests/test_phase6_resume_api.py` (`resume_job` contract),
`tests/test_phase6_translation_cache.py` (translation survives a reconstructed
translator/cache: translate -> persist -> destroy -> resume -> reuse, at 120-line
scale, plus invalidation on text/language/backend/setting change and the
config-drift fail-safe). All use the real
modules — no mocks — because the phase is entirely about bytes on disk surviving
a process boundary. A stand-in must implement the full surface the pipeline
calls: `route_model`, `validate_reference`, `get_available_voices`,
`generate_batch`, `unload_all`.

## Phase 6 audit (2026-10-21) — findings and known limits

A focused review of the four durable-state modules plus a sweep of `references/`
for resume-like patterns. One real bug was found and fixed; the rest are
documented limits, not defects, because the phase's contract is single-process.

### Fixed
- **Chunk-level resume was a no-op.** In `_process_chunked` the loop reset a
  chunk to `PENDING` *before* the reuse check, so the check never saw the
  `COMPLETE` record it tests for and every chunk was redone on every resume.
  The reset now happens only after the reuse check declines. Pinned by
  `TestChunkNamespacing::test_a_finished_chunk_is_not_reset_to_pending_before_the_reuse_check`.

### Verified sound (do not "fix" these)
- **`Manifest.save()` durability and its concurrency fix.** `save()` now writes
  the new primary *first* and then copies the previous bytes to `.bak` (it used
  to `os.replace` the primary to `.bak` before writing). Crash-durability is
  unchanged: if the process dies mid-save the primary is either the old file or
  the new one, never partial, because the write is `atomic_write_json`. What the
  old ordering got wrong only showed up under *concurrency*: `Manifest.load()`
  treats an absent primary as a reason to fall back to `.bak` and **repairs the
  primary from it**, so a status poll landing in the window where the primary had
  been moved away would read the stale `.bak` and resurrect it — permanently
  reverting a just-written terminal status or completed stage. Copying instead of
  moving means the primary is never absent. Pinned by
  `tests/test_api_recovery.py` (the resume/status tests go red if `.bak` handling
  regresses) and the `test_api_jobs.py` cancellation tests.
- **`clean()` safety.** Refuses everything when `active=True`; never removes
  `final` or the manifest except at `FINAL_ONLY`; containment is asserted per
  target before any `rmtree`.
- **Atomic writes and containment.** Every job write goes through
  `atomic_write_*` (temp file in the same dir + `os.replace`); job artifact
  paths are resolved and asserted inside the job directory.
- **Cancellation.** `_check_cancel` sets `CANCELLED` (never `FAILED`) and
  demotes the running stage to `PARTIAL`, which is what makes a cancel
  resumable. `run_with_retry` re-raises `CancelledError` untouched.
- **`validate_wav`** checks the RIFF-declared size against the real file size,
  because libsndfile silently clamps a truncated file's frame count.

### Known limits (not bugs; out of scope for this phase)
- **Config drift fails safe, does not reconcile.** `Manifest.load_or_create`
  keeps the *stored* `config`/`source` when a job is reopened, and
  `_stage_fingerprint` hashes that stored block. So a bare
  `process_video(..., resume=True, job_id=...)` with a *different*
  `target_lang`/`tts_model_name` would recompute the old fingerprint, match, and
  return the previous run's output. `process_video` now calls
  `Manifest.differs_from(request_config)` and raises `JobError` rather than
  continue. `resume_job(job_id)` is the supported way to continue a job: it
  replays the manifest's own config, so the two always agree and the check never
  fires. Reconciling (invalidating only the affected stages and continuing) is
  still a later hardening step, deliberately not implemented.
- **Single writer per job is required.** Nothing prevents two concurrent runs
  on one job from interleaving manifest saves. This is a documented constraint,
  not an accident: proper process locking is a later hardening item to do before
  multi-user/server deployment. A `JobLock` (atomic `O_CREAT|O_EXCL` with
  stale-takeover) was prototyped and deliberately left out of Phase 6.
- **`retry.py`: production-wired vs future-facing.** Wired: `run_with_retry`
  via `VideoTranslator._run_stage_with_retry` around extraction, separation,
  diarization, transcription and TTS. Future-facing, tested but with no
  production caller yet: `should_retry_segment` (per-segment retry policy),
  `summarise` (manifest-ready attempt summary), and `RetryOutcome.history`
  (populated, never persisted). Left in place on purpose — they are the natural
  hooks for the next hardening step; remove only if that plan changes.
- **Translation cache: one authoritative path (fixed).** The job-scoped
  `ArtifactCache` (`cache/translation.json`) is the production translation cache.
  `VideoTranslator.process_video` hands it to the translator
  (`self.translator.job_cache = self.cache`); the `Translator` reads and writes
  it through `_cache_get`/`_cache_put`/`_cache_flush` and persists per block.
  The old `temp/translation_cache.json` is only used when there is no job
  (standalone `Translator`, unit tests) and is now written atomically too. The
  two are never active at once.

### Translation persistence (the authoritative cache)
One store, not two. When a job is active, `ArtifactCache.translation.json` under
the job directory is the only translation cache:
- **Identity** is `translation_key(text, src, tgt, backend, settings)` — a
  content-addressed hash. It covers source text, both languages, backend, and
  `settings` (`optimize`, `concise`, plus the neighbouring source lines and
  speaker when the backend actually uses dialogue context). Segment index and
  timestamps are deliberately excluded: the same line is the same translation.
  Changing any covered input yields a new key, so a stale hit is impossible.
- **Persistence** is atomic (`atomic_write_json`). The stage checkpoints every
  `TRANSLATION_FLUSH_EVERY` (50) lines; context-aware translation is one call so
  it checkpoints once at the end. A crash at line 800 of 1200 keeps the 800.
- **Partial completion** is natural: a failed line is recorded in
  `last_failures` and left empty, never cached, so a resume retries exactly it.
- `Translator` reaches the store through `_cache_get`/`_cache_put`/`_cache_flush`,
  which fall back to the temp file only when `job_cache` is None.

### References
- `references/Multiva.Ai/engine/project.py` is the closest analogue (same
  "manifest never fails a finished render" contract, atomic write-beside-rename)
  but explicitly does **not** support resume: `scan()` skips any manifest whose
  status is not `done`. Phase 6 goes beyond it deliberately.
- `references/video-dubbing-system` has no resume or checkpoint machinery.
- Neither reference has cross-process locking; concurrency is new ground here.

## Scoped invalidation, lip-sync checkpointing, and mixing (Phase 7)

Phase 7 makes a re-render cheap and honest: it redoes only what a change
actually affects, it does not lose finished lip-sync work to a crash, and it
stops the dub from guessing about actors or drowning the background.

### Scoped invalidation — `src/utils/config.py`, `src/core/manifest.py`
- `STAGE_CONFIG_KEYS` lists the run-config keys each stage depends on.
  `config_keys_through(stage)` returns that stage's keys plus every upstream
  stage's, and `_stage_fingerprint` hashes only those (plus the source). An
  unknown stage falls back to the whole config, so it cannot *under*-invalidate.
- `RE_RENDERABLE_CONFIG_KEYS` is the allow-list of changes permitted on a
  finished job. Everything in it maps to a sticky stage (lip-sync, visual
  translation, the mux) or to the mix; changing anything else still refuses,
  because that work was made under the old settings.
- Every fingerprint folds in `config.FINGERPRINT_SCOPE` (stored in
  `manifest.source["fingerprint_scope"]`). This is what stops a manifest
  written before Phase 7 — whose fingerprints hashed the whole config — from
  matching a scoped one and handing back stale work.

### Lip-sync checkpointing — `src/core/video_translator.py`
- Lip-sync is a durable stage. The cache is read *before* `_begin_stage`, so a
  `COMPLETE` record is seen; a resume after a crash skips it instead of
  re-running the model. Do not move that read below the reset.
- `_valid_artifact` validates the file (a crash mid-write leaves a file, and
  treating that as a hit is how a dub gets a silent stretch). `_stage_done`
  requires both a valid artifact and a `COMPLETE` record.

### Multi-face safe fallback — `src/processing/wav2lip.py`
- PASS 1 records the box *and* the face count per frame (`_detect_frame_boxes`
  returns `(all_boxes, chosen_box)`; `chosen_box` is the largest face only when
  there is exactly one, else `None`).
- `get_smooth_box` / `_fill_missing_boxes` treat a multi-face frame as a gap
  and would fill it from neighbours — silently picking an actor. After gap
  filling, ambiguous frames are re-marked `None` using
  `_smooth_face_counts` (median, then max/dilate). Median alone is not enough:
  it can average a genuine two-face frame at a shot edge back to one, and
  moving a guessed actor's mouth is the failure we are preventing. Dilating
  only costs a skipped frame of lip-sync, so the filter errs safe.
- `_prepare_chunk_batches` gives `None` frames a placeholder crop so they keep
  their slot and the inference results stay aligned with the target frames;
  PASS 2 writes such frames through untouched.

### Levelled mixing — `src/audio/mix_loudness.py`, `video.py`, `synchronization.py`
- `estimate_mix_gains` places the background a set distance below the dialogue
  instead of a fixed `bg=0.8`. Two invariants, both load-bearing:
  the background is **never boosted** (boosting it raises its noise floor), and
  the dialogue gain is **bounded** so a quiet line is not flattened.
- `limit_peaks` corrects only a clip-risk peak, and only by the needed amount.
- `render_timeline` applies equal-power fades so segment onsets do not click,
  and sums overlapping segments with deferred writes so an intentional overlap
  keeps both speakers.

### Tests
- `tests/test_phase7_scoping_lipsync_mix.py` (18 tests) drives the real
  manifest, renderer, and mixer; only the ML models are not loaded.
- Full suite: 1059 passed, 14 skipped.

## Quality presets (P5)

`config.QUALITY_PRESETS` bundles quality settings behind a name; the caller
opts in with `process_video(..., quality_preset="FAST"|"BALANCED"|"HIGH")`.
Passing None (the default) changes nothing. `preset_config` raises on an
unknown name and refuses any preset that sets a non-quality key, so a preset
can never change what the dub says or whose voice says it.

- FAST: `enable_lipsync=False`, `lipsync_model_name="Wav2Lip-GAN (Low Quality
  - Fast)"`, `enable_audio_enhancement=False`.
- BALANCED: `enable_lipsync=True`, same Wav2Lip model, no enhancement.
- HIGH: `enable_lipsync=True`, `lipsync_model_name="LivePortrait (High
  Quality - Slow)"`, `live_portrait_mode="ort"`, `enable_audio_enhancement=True`.

Every key a preset sets belongs to the `sync`, `lipsync`, or `final` stage, so
FAST -> HIGH invalidates only lip-sync and the final mux; ASR, diarization,
references, translation and TTS stay cached.

## Real-sample validation (P7)

`tests/test_phase7_real_validation.py` runs the real mixer, synchronizer and
mux on the His Girl Friday job's cached stems and real frames. It needs the
job's `separation/` and `sync/` artifacts and `/tmp/sample/hgf_scene.webm`.
The face-detector tests download the S3FD/2DFAN4 weights on first run.

Measured on the real sample:
- Background is already ~42 dB below the dialogue, so `estimate_mix_gains`
  leaves it untouched (never boosts); a loud bed is pulled to ~18 dB below the
  dialogue or clamped at the 24 dB cap.
- No onset clicks; the dub is unclipped (peak 0.455).
- No cumulative A/V drift: onset offset is +0.17 s at the start, +0.18 s in the
  middle, +0.15 s at the end; duration 96.694 s against 96.6935 s.
- Residual English in the accompaniment is ~37 dB below the dialogue
  (NEGLIGIBLE); the speech band is quieter during dialogue than outside it and
  the vocals/accompaniment correlation is +0.08, so the limit is Demucs
  separation, not the mixer.
- MP4 -> H.264 + AAC, WebM -> VP9 + Opus, audio-only replacement stream-copies
  the video.
- On real frames: 16/49 single-face (eligible), 33 multi-face (untouched),
  0 no-face. Inference is NOT run: this environment has no Wav2Lip/LivePortrait
  weights.

`Manifest.reconcile` writes the manifest, so any test that drives it on the
real job must first copy the job; mutating the real one corrupts the fixture.

## Read-only HTTP API (F1)

`src/api/` is an additive FastAPI layer. It reports what the installation can do
and runs no dubbing. `app.py` (Gradio, port 7860) is unchanged and remains the
UI of record; the API runs separately on port 8000.

```
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Endpoints (all GET): `/api/health`, `/api/system`, `/api/models`,
`/api/languages`, `/api/capabilities`, `/api/options`, `/api/voices`.

These are the read-only endpoints. F2 adds uploads, execution and events on the
same app — see the F2 section below.

### Import order is load-bearing

`src/api/bootstrap.py` reproduces the order `app.py` establishes: encoding patch
-> `config`/`languages` -> CUDA/DLL setup -> late patches. `src.utils.config`
must be imported before `torch`/`transformers`, or Windows dies with
`WinError 127`. `bootstrap()` is idempotent and thread-safe, and `main.py` calls
it before importing any route module.

### Two traps worth remembering

**`src/translation/__init__.py` eagerly imports `text_translator`**, which
imports torch, transformers, sklearn and deep-translator. So `from
src.translation.router import ...` pulls in the whole ML stack (~3 s) and fails
outright when transformers is absent. `discovery.router_module()` loads
`router.py` directly from its file instead, registering the module in
`sys.modules` *before* `exec_module` — `dataclasses` resolves annotations via
`sys.modules[cls.__module__]` and raises `AttributeError` otherwise. There is one
`_ROUTER_MODULE` cache, so the backend list is not duplicated.

**`importlib.util.find_spec("pyannote.audio")` imports `pyannote`**, which drags
in transformers and sklearn. `capabilities._find_spec_without_import()` walks the
search locations by hand to avoid it. `module_available()` is import-free by
design; keep it that way.

### Capability states, not booleans

`ComponentState` distinguishes `available`, `not_installed`,
`model_not_downloaded`, `token_required`, `unsupported_platform`, `unavailable`
and `unknown`, because each has a different remedy. IndicF5 reports its
*implementation* availability separately from its *weight/access* availability:
runtime missing -> `not_installed`; runtime present but no token and no cache ->
`token_required`; token present but no cache -> `model_not_downloaded`; cached ->
`available`.

`cloning_supported_by_architecture` and
`cloning_model_available_on_this_machine` are deliberately separate in
`/api/languages`: the first is a property of the language/engine pairing, the
second of this installation. Conflating them would let the UI promise a cloned
voice the machine cannot produce.

### Probes never download and never infer

Every probe is a filesystem check, an import-availability check or a config
read. `/api/models` must stay network-free and safe on a CPU-only box with no
`HF_TOKEN`. `tests/test_api_readonly.py` traps `hf_hub_download` /
`snapshot_download` to enforce this. No endpoint returns `HF_TOKEN` or an
absolute filesystem path.

`/api/capabilities?include_routes=true` builds the source/target route matrix
(702 routes). It is opt-in and cached per set of available backends, because
building it loads the translation package.

### Options are duplicated on purpose

`src/api/options.py` reproduces the dropdown literals from `app.py` rather than
sharing them, because `app.py` must not be modified. `OptionsResponse.source`
records that provenance, and tests assert the lists still match the `app.py`
literals and `src/synthesis/capabilities.CAPABILITIES`. When migration happens,
collapse the duplication rather than growing it.

### Tests

`tests/test_api_readonly.py` — 43 tests driving the real app through
`TestClient` (no mocks; only the environment is faked). Three tests run in a
subprocess, because an earlier import in the same session would otherwise mask
what they check: `/api/health` and `/api/capabilities` must not import
transformers, and the router must load without the translation package.

Deps: the `api` extra in `pyproject.toml`, or `fastapi` + `httpx` for tests.
`TestClient` needs `httpx`.

## Upload, execution and events (F2)

F2 adds the write half of the API on top of F1's read-only layer. The pipeline
is not modified and `app.py` is untouched; a job runs the same generator the
Gradio UI consumes.

Endpoints: `POST /api/uploads`, `GET|DELETE /api/uploads/{id}`,
`POST|GET /api/jobs`, `GET /api/jobs/{id}`, `POST /api/jobs/{id}/cancel`,
`GET /api/jobs/{id}/events`, `GET /api/jobs/{id}/output`,
`GET /api/jobs/{id}/output/download`.

```
curl -X POST 'localhost:8000/api/uploads?filename=clip.mp4' \
     --data-binary @clip.mp4
curl -X POST localhost:8000/api/jobs \
     -H 'content-type: application/json' \
     -d '{"upload_id":"up-...","target_lang":"hi"}'
curl -N localhost:8000/api/jobs/<job-id>/events      # SSE
```

### Module layout and the one seam

`settings` (env-tunable limits) -> `uploads` (staging, one-shot claim) ->
`media` (reuses the pipeline's `validate_video`) -> `validation` (request
rules) -> `manager` (queue) -> `runner` (one job) -> `events` (contract +
buffer) -> `jobstate` (read model) -> `routes/{uploads,jobs}`.

`JobRunner._build_translator` is the *single* point where F2 touches the
pipeline. Everything else — scheduling, isolation, event mapping, cancellation,
durability — is F2's own code and is tested for real. Tests patch only that
method, which is why the F2 suite needs no model weights.

### Isolation is the reason JobRunner exists

The Gradio UI builds one module-level `VideoTranslator` and reuses it. That
instance holds the loaded model, the session, the manifest and the cancel flag,
so two jobs sharing it would cancel each other and overwrite each other's
durable state. Each `JobRunner` constructs its own instance and nothing outside
the runner holds a reference. Cancellation calls `abort()` on *that* job's
translator only.

The generator is consumed on a dedicated thread, never on the event loop, or a
render would freeze `/api/health`. The thread touches asyncio only through
`EventBuffer.publish`, which marshals with `call_soon_threadsafe`.

### Blocking work never runs on the event loop

`create_job` does its directory creation, source move and manifest write in
`asyncio.to_thread`; `create_upload` probes with ffprobe the same way. A
multi-gigabyte `os.replace` or a slow ffprobe would otherwise stall every
concurrent request.

The upload body is streamed to disk chunk by chunk rather than parsed as
multipart, so `MAX_UPLOAD_BYTES` is enforced *during* the transfer and a
rejected upload costs one chunk, not a spooled copy of the whole body.

### Two sources of truth, kept separate

Durable state is the `Job` directory and its `Manifest`, written by the
pipeline; it is what a status read uses and it survives a restart. Live state is
the `JobRunner` in this process: queue position, current fraction, cancel state.
A job with a manifest but no runner is normal (finished before this process
started) and is reported from the manifest alone.

**A manifest left `RUNNING` with no runner reports `interrupted`, not
`running`.** The process that wrote it is gone; saying `running` would leave a
client polling forever. `jobstate._manifest_state` is the only place that
decision is made.

### Cancellation and durability

`runner._notify_finished` is exactly-once, guarded by a lock, so the terminal
transition (and the manager's completion callback) cannot fire twice.

A job cancelled *while queued* never reaches the pipeline, so the pipeline's own
`CANCELLED` write never happens. `_finish_cancelled` therefore writes the
terminal status itself via `_record_manifest_terminal`. Without that the
manifest would keep claiming `RUNNING` and the status read would report
`interrupted` — a different and wrong answer for a deliberate cancellation.

A completed or failed render has *already* written its own status through the
manifest it owns, so the runner does not rewrite it: doing so would race the
pipeline's write for no gain.

`main._lifespan` calls `JobManager.shutdown` on server exit, so a render reaches
a checkpoint and writes its durable state instead of being abandoned mid-stage
with the manifest still claiming `RUNNING`.

### Config is resolved once, in the API

`routes/jobs._run_config` mirrors `VideoTranslator._run_config`. The manifest is
written with it before the render starts and the pipeline recomputes and
compares it, so the two must agree *exactly* — a mismatch is reported as
configuration drift and the job refuses to run. `test_pipeline_contract` pins
the key set against the real method.

A quality preset is therefore applied by the API, not forwarded: forwarding it
*and* recording the config would make the recorded and effective configurations
disagree and the drift check would reject the job the API had just created.
Languages are normalised to codes for the same reason — the UI sends "Hindi",
the pipeline hashes "hi".

### The event stream

`events.py` defines the contract; the transport is SSE. The first frame is a
`snapshot` of durable state so a client can render immediately. `Last-Event-ID`
is honoured (header or `last_event_id` query param, since `EventSource` cannot
set headers) so a browser's automatic reconnect resumes without gaps; if the
position has rotated out of the bounded buffer, a `resync` frame says so rather
than silently dropping events. Heartbeats (`: keepalive`) every
`HEARTBEAT_SECONDS` keep a long silent model stage from timing out.

Subscription happens *before* replay, so an event published between the two is
not lost; a duplicate is harmless because clients key on `event_id`. The buffer
is bounded (`maxlen`), which is the leak the phase brief forbids, and the
subscriber is always removed in a `finally`.

A job from a previous process has no buffer and no runner: the stream sends the
snapshot, then a `resync` explaining that live events are unavailable, then
closes. Without that a client would hang on a stream that can never produce an
event.

### Request validation is stricter than the pipeline

`validation.py` checks languages, the TTS/language pairing (via the pipeline's
own `validate_tts_pair`), presets and every enumerated option against the lists
`/api/options` publishes — all before a job directory exists. Availability is
*not* enforced: a model whose weights are not downloaded can still be requested,
because the pipeline downloads on first use and refusing it would make the API
stricter than the product. Only the *name* has to be real, so a typo is a 422
rather than a failed render minutes later.

### Tests

`tests/test_api_jobs.py` — 43 tests driving the real app through `TestClient`
against the real `Job`, `Manifest`, `JobManager` and `EventBuffer`. Only
`JobRunner._build_translator` is replaced, by a fake that yields the same
log/progress/result tuples and writes genuine durable state through the real
`Manifest` API. `test_pipeline_contract` pins that fake against the real
signature so it cannot drift. One test runs in a subprocess to prove `/api/health`
and `/api/jobs` do not import the pipeline.

Must pass CPU-only, offline, with no weights and no `HF_TOKEN`.

`tests/test_api_recovery.py` — the recovery/resume suite (F3). Same setup as
`test_api_jobs.py`; it drives the real app, the real `Manifest` and the real
`JobManager` and replaces only `_build_translator`. Two conventions are load
bearing and easy to get wrong:

- **Simulate a restart with `_restart_manager()`.** A finished runner is retained
  for late SSE subscribers and its state is merged into status responses, so a
  test that wants the *recovery* view — what the API reports from disk alone —
  must drop the runners first. Without it, an "interrupted" job still answers
  `completed` from the retained runner.
- **Wait on the durable manifest, not the response.** `_wait_for_terminal` reads
  `Manifest.load()` because these tests then mutate the manifest to stage a
  scenario; waiting on the response would race the thing they are about to
  change. After a *resume* the manifest may already be terminal from the earlier
  stop, so `_wait_for_run` waits on the runner instead.

The suite also pins the cross-process durability rule from
`src/core/manifest.py`: a status poll concurrent with a manifest write must not
resurrect the previous state. That is what the `save()` copy-don't-move ordering
exists for.


## Frontend (`frontend/`)

A React 19 + Vite + TypeScript SPA that talks to the FastAPI service over the
same origin (`/api/*`). Vite proxies `/api` to `127.0.0.1:8000` in dev, so no
component ever holds an absolute backend URL and the dev and production shapes
match (FastAPI serving the built bundle). Plain CSS with design tokens in
`src/styles/tokens.css`; no CSS framework.

### Commands

```bash
cd frontend
npm install
npm test              # vitest, 231 tests
npm run typecheck     # tsc -b --noEmit
npm run build         # tsc -b && vite build
npm run dev           # dev server on :5173, proxies /api to :8000
```

`VITE_ALLOWED_HOSTS` (comma-separated) is needed only when reaching the dev
server through a tunnel or proxy; Vite blocks unknown `Host` headers by default.
No hostname is committed.

### Layout

- `src/api/endpoints.ts` is the single source of every URL path. Add a call
  there, not inline in a hook.
- `src/api/types.ts` is a hand-written mirror of `src/api/schemas.py`.
- `src/api/client.ts` normalizes every failure into an `ApiError`; a network
  failure is flagged `isOffline`. Callers never see a raw rejection or an HTTP
  status.
- `src/hooks/` wrap `useAsync`, which aborts on unmount and ignores the abort.
- `src/utils/` hold the pure rules (state wording, language filtering, model
  grouping) so components stay thin.

### Rules that matter

- **The backend is the source of truth.** Capability flags, language support,
  route availability, whether a job can resume and the wording of why not all
  come from the API. The frontend never re-derives support from a hard-coded
  list. In particular, a job's resume affordance reads `resume.can_resume` and
  its explanation is the backend's own `resume.message`.
- **Absence is not failure.** A missing route matrix (`routes: []`) means the
  matrix was not requested, not that a pair is unsupported. The UI says so
  rather than claiming a route is missing.
- **Normalize legacy job language values.** The durable manifest holds both
  display names (`"English"`, from older jobs) and codes (`"en"`). Render them
  through `displayLanguage()` in `src/utils/languages.ts` so the same field
  looks the same on every row.
- **`useHealth` is `.tsx`.** It exports a provider component, so the extension
  must be `.tsx`; `tsc` fails with a parse error otherwise.

### The OpenAPI contract test

`src/test/contract.test.ts` reads `src/test/contract/openapi.json` (a snapshot
of `app.openapi()`) and the text of `src/api/types.ts`, and fails when the
hand-written mirror and the backend disagree — a field renamed, added or
removed. It parses the `.ts` source because types do not exist at runtime.

After changing `src/api/schemas.py`, regenerate the snapshot and reconcile the
types:

```bash
.venv/bin/python frontend/scripts/export_openapi.py   # or: npm run openapi:export
```

The snapshot is committed on purpose: that is what makes the check run without
importing the Python app.

### Testing notes

Tests render real pages against the real hooks and the real client, stubbing
only `fetch` (`src/test/fetchStub.ts`). Nothing above the transport is mocked,
so a change to error parsing is caught. An undeclared request is a test failure
rather than a silent 404.

Two traps when adding component tests:

- "Hindi" appears both in a language card and in the route explorer's
  `<option>` elements. Scope assertions with `within(card)` or `getAllByRole`
  rather than `getByText`, or the query matches twice.
- `BackendIndicator` renders only inside `BackendHealthProvider`; render it
  directly to assert on the indicator text.

The upload path uses `XMLHttpRequest`, not `fetch`, because upload progress is
only observable through XHR. `src/test/xhrStub.ts` provides `FakeXhr`, whose
constructor records each instance in `FakeXhr.instances`; a test asserts on
`FakeXhr.instances` (e.g. its length and the recorded body). If the constructor
stops recording, those assertions pass vacuously against an empty array — so a
change to `FakeXhr` must keep recording in the constructor, and the workflow
test that checks `FakeXhr.instances` must stay.

## Guided demo (F6)

`/demo` runs a real dub on a bundled public-domain clip so a first-time visitor
does not have to find a video before seeing the pipeline work. It is not a
simulation and has no separate runner:

1. `POST /api/samples/hgf_scene/claim` stages the clip as an ordinary upload
   (`GET /api/samples` lists it, `GET /api/samples/{id}/media` previews it with
   byte ranges).
2. `POST /api/jobs` creates a normal durable job from that upload.
3. The browser lands on the ordinary `/jobs/{id}` page for progress, SSE,
   cancel and result.

Nothing about the route or defaults is decided in the page: `src/utils/demo.ts`
reads the backend's own `features`, language lists and `priority_indic`, so the
demo cannot advertise a configuration the machine cannot run and no readiness
check is hard-coded green. The default route is EN→HI with local NLLB and Edge
TTS, each used only when available.

`src/api/samples.py` is a curated allow-list, not a directory listing: a client
names a sample by `id` and every path is derived from the table, so `samples/`
sitting next to source and credentials is never served by name. `stage_file()`
in `src/api/uploads.py` copies (never moves) the file into the upload staging
area and probes it exactly as a client upload is probed, so the job that follows
cannot tell the difference. Adding a sample means adding an entry to the
allow-list; nothing else makes it reachable.

## Production serving, launcher and runtime dirs (F7)

### One command, one origin

`python run.py` is the production entry point. It is deliberately thin: it
checks the interpreter (3.10–3.12) and the web deps, prints a banner, then calls
`uvicorn.run(app, workers=1)` **in-process**. No subprocess tree, so Ctrl+C
reaches the server and no orphan holds the port — the failure mode that makes
Ctrl+C unreliable on Windows. It reimplements nothing: the heavy import-order
bootstrap is `src/api/bootstrap.py`, the same one `src/api/main.py` runs.

Flags/env: `--host`/`API_HOST`, `--port`/`API_PORT`, `--open`/`API_OPEN_BROWSER`,
`--log-level`/`API_LOG_LEVEL`, `--check`. A missing frontend build is a banner
warning, not a fatal error, unless `API_REQUIRE_FRONTEND=1`.

### How FastAPI serves the React build

`src/api/frontend.py`. The rule that is easy to get wrong: the SPA fallback is
**ASGI middleware** (`FrontendFallback`), not a `@app.get("/{path}")` catch-all
route. A catch-all route makes an unmatched path a *partial* match for every
other method, so `POST /api/jobs/..%2F..%2Fetc/resume` would return `405`
instead of the API's `404`, and `PUT /api/jobs/<id>` would lose its `405`. The
middleware lets Starlette route exactly as before and only substitutes a
response for a genuine `404`.

What it substitutes, and only for a GET/HEAD to a non-`/api` path:

- a real build file → served as itself;
- a path in `CLIENT_ROUTES` → `index.html` (deep link / refresh);
- anything else → a plain 404. It is deliberately **not** a blanket
  "unknown → shell": that would answer `GET /etc/passwd/media` with 200 + HTML
  and make a typo look like a success. `tests/test_api_frontend.py` pins this.

`CLIENT_ROUTES` mirrors the `<Route>` table in `frontend/src/App.tsx`; add a
route in both places or a direct load of it 404s. `/assets` is a `StaticFiles`
mount so it streams and answers byte ranges. Config lives in `src/api/settings.py`
(`FRONTEND_DIST`, `SERVE_FRONTEND`, `CORS_DEV_ORIGINS`, `HOST`, `PORT`,
`OPEN_BROWSER`), all env-overridable with the `API_` prefix.

### Runtime-directory policy

A job owns everything it writes. `Job.scratch_dir()` is `jobs/<id>/scratch/`
(added to `JOB_SUBDIRS`), and the pipeline writes its non-durable intermediates
there through `VideoTranslator._scratch_path()` instead of the shared
`config.TEMP_DIR`. This fixes two real bugs: two jobs with the same source stem
collided, and a run whose stem matched a committed fixture (e.g.
`temp/vocals.wav`, `output/translated_hgf_scene.webm`) overwrote a tracked file.
`_scratch_path()` falls back to `config.TEMP_DIR` when there is no job (the
legacy path). `AudioSeparator.separate()` takes `output_dir` for the same reason.
`clean()` always drops `scratch` — it is working space, never a resume input.

`.gitignore` ignores new files under `/jobs`, `/temp` and `/output`
(`/jobs/*` etc.). The already-committed Phase 7 evidence stays tracked — git
keeps tracked files regardless of ignore rules — but a fresh run can no longer
bury real changes under gigabytes of intermediates. Do not "fix" this by
untracking the evidence; the samples and rendered output are deliberate.

### Subtitle modes

`subtitle_mode` (`none`, `external`, `embedded`, `burned`) is plumbed from the
API through `_run_config` into the `subtitles` stage and the final mux. The
registry lives in `src/utils/config.py` (`SUBTITLE_MODES`,
`DEFAULT_SUBTITLE_MODE`), and `config.STAGE_CONFIG_KEYS["subtitles"]` names it,
so changing the mode invalidates only the subtitle stage and everything after
it. The API rejects an unknown value with a 422; the pipeline falls back to
`external` for an unknown value so a typo still yields subtitles.

`none` records the stage as `SKIPPED` (terminal-OK, but not a claim that a file
exists), which is why `Manifest.skip_stage` exists alongside `complete_stage`.
`embedded` and `burned` run `VideoProcessor.add_subtitles` on the finished mux
and record the subtitle-bearing file as `final_output`. Burn-in goes through an
explicit argv rather than ffmpeg-python: the builder escapes the already-escaped
filter path a second time, and libass then finds no file while ffmpeg exits 0.

Subtitle artifacts resolve through `src/api/artifacts.py` (`subtitles`,
`subtitles_vtt`), whose `manifest_key` makes the recorded path authoritative
over the glob candidates. `tests/test_subtitles.py` covers all three layers,
including a real burn-in whose pixels must change.

## Preflight and the durable result summary (F8)

Two features that share one idea: the backend already knows something the user
needs before and after a render, and both should be *reported*, not re-derived.

### Preflight is advice; `createJob` is the authority

`POST /api/jobs/preflight` (`src/api/preflight.py`) answers "is this render
ready, and why not" for a `JobCreateRequest` **before** a job directory exists.
It is advisory on purpose. Every check is computed from the same registries
`POST /api/jobs` validates against, but creating a job re-runs all of them, so
the panel can never be the reason a valid render is refused.

The distinction that makes severities meaningful:

- **blocking** — the render cannot be *correct* as configured: a TTS backend
  that cannot speak the target, a required backend that is unavailable, a file
  with no audio stream, `min_speakers > max_speakers`.
- **warning** — something is off but the render is still valid: an optional
  feature that is unavailable *while it is switched off*.

The rule for optional features is: unavailable is a **warning while the feature
is off**, and becomes **blocking the moment the user selects it**. That is what
keeps the panel from being a permanent amber light. `_option_checks` encodes it
with a `(field, label, enabled, required)` plan; `_feature_checks` handles the
flags that have no model choice of their own.

Two implementation notes that were bugs first:

- `OptionItem` has **`description`**, not `reason`. Reading `.reason` raised
  `AttributeError` on the pydantic model.
- Severities are **plain strings** (`OK`/`WARNING`/`BLOCKING`), not an enum, so
  a new severity is additive rather than a breaking wire change. The frontend
  maps the string; an unknown string falls back to a neutral tone and prints
  itself rather than being hidden.

`build_preflight(request, states=None)` takes injectable component states so a
test (or a caller that already probed) does not pay for a second probe round;
when omitted it uses the same `caps.component_states()` the `/api/models`
dashboard uses, so the two cannot disagree. `tests/test_preflight.py` (22 tests)
pins every branch, including that the panel reports *all* problems at once
(`validate_request` stops at the first; a user fixing a form wants the list).

### The result page summarizes how the render was made

`src/api/summary.py` builds `JobSummary` on the job status response: the
language pair, translation/context backends, voice mode, TTS backend and voice,
subtitle mode and cue count, lip-sync and diarization facts, and duration. It is
**durable** — recorded at job creation, not recomputed from the form — so the
result page describes the render that actually ran even after a resume or a
reload.

`_name(code)` is the subtle part. It checks `languages.is_known(code)` **first**
and returns `None` for an unknown code; only a known code is passed to
`get_language_name`. The previous order let an unknown code fall through to the
display-name lookup's `"English"` default, so a bogus language rendered as
English. `auto` maps to `"Auto-detect"`, and `source_language_name` /
`target_language_name` are `null` when unknown rather than a wrong name.

The frontend mirrors this in `summaryRows()` (`frontend/src/utils/dubbing.ts`),
which returns labelled lines and **omits** a line it has no fact for rather than
printing "unknown". Off features (no lip-sync, no diarization) are omitted, not
shown as "off". The panel is `JobResultPage.tsx` ("How this was made").
`frontend/src/test/dubbing.test.ts` covers the omission rules directly;
`workflow.test.tsx` covers the rendered panel, including a job that predates the
summary fields.

### Frontend wiring notes

- `usePreflight` (`frontend/src/hooks/usePreflight.ts`) is deliberately **not**
  built on `useAsync`: its request body changes on nearly every keystroke, and
  `useAsync` compares deps by reference. It debounces (350 ms), serializes the
  body to compare it, and aborts in flight so a stale verdict cannot overwrite
  the current one. A preflight *failure* renders as "could not check readiness"
  and leaves Start enabled — only a settled `ready: false` disables Start, and
  even then `createJob` would have caught it.
- `DubPage` builds the request body **once** (`formValues` → `buildJobRequest`)
  and shares it between Start and preflight, so the verdict is always about the
  render that would actually be submitted.
- Test trap: `baseRoutes()` declares a default `/api/jobs/preflight` route, and
  `installFetchStub` matches the **first** declared route. A test asserting on
  the verdict must declare its own preflight route **before** `...baseRoutes()`.

## Hosted deployment: Colab, Kaggle, Docker (A9)

### One runtime module, two thin notebooks

The notebooks do the same thing and differ only in the platform's opening cell
(clone directory, which proxy to use, what to tell the user about the URL). The
shared logic is **not** in the notebooks — it is `src/deploy/runtime.py`, which
is importable, testable without a GPU, and the single place the two variants
cannot drift apart.

The notebooks themselves are build products: `scripts/build_notebooks.py` holds
their content as plain Python data and writes both `.ipynb` files. Edit the
script, not the notebook. `--check` re-renders and compares against disk, and
`tests/test_deploy_notebooks.py` runs it, so a notebook edited in the browser
without updating the script fails the suite. Cell ids are derived from cell
content (sha1), so a re-render is byte-identical.

### Ordering rule: configure the runtime *before* importing config

`src/utils/config.py` reads `TEMP_DIR`, `OUTPUT_DIR` and `JOBS_DIR` from the
environment **at import time** and creates them. So `configure_runtime()` must
run first; redirecting afterwards is too late. `tests/test_deploy_notebooks.py`
asserts that no notebook cell before the configure cell imports `src.utils.config`
(directly or via `src.api.main`) or torch.

This is why `TEMP_DIR` and `OUTPUT_DIR` are now env-overridable like `JOBS_DIR`
already was (A9 change; defaults unchanged). `configure_runtime()` also sets
`API_UPLOADS_DIR`.

### One process, one worker — the notebooks obey it too

`start_server()` runs uvicorn in a **daemon thread of the current process**, not
a subprocess, for the same reason `run.py` uses `workers=1`: the pipeline holds
multi-gigabyte models and an in-process job registry, so a second worker doubles
memory and splits the registry. A notebook also must keep its kernel responsive,
which rules out a subprocess tree.

`ServerHandle.stop()` sets uvicorn's `should_exit` and joins with a timeout, and
is **idempotent** — a notebook cell is re-run, so the start cell stops any
previous `server` before starting a new one. A leaked server thread holding the
port is the failure this prevents.

### Reaching the app from a browser you are not sitting at

On a notebook the VM's `127.0.0.1:8000` is not reachable from the user's
browser. `notebook_public_url(port)` resolves in order:

1. Colab's built-in `google.colab.kernel.proxyPort` (no account or token);
2. `PUBLIC_HOST` (+ `PUBLIC_SCHEME`, default `https`) — a tunnel;
3. loopback, with the notebook telling the user it is VM-local.

Kaggle has no built-in proxy, so a tunnel is required there. `start_server`
binds `0.0.0.0` because a proxy connects from outside loopback.

### Preset recommendation is a documented judgement

`hardware_report()` maps free VRAM to a preset: `>= 16 GB` → `HIGH`, `>= 8 GB` →
`BALANCED`, else `FAST` (no lip-sync). It imports torch lazily so it works before
the pipeline is set up, and reports a missing torch as a note rather than
raising. The thresholds are duplicated in `docs/DEPLOYMENT.md`, and
`tests/test_deploy_docs.py` fails if the two disagree.

### The deliverable archive follows git's view

`scripts/build_archive.py` writes the distributable `video-translator-full.zip`.
It collects files through `git ls-files --cached --others --exclude-standard`,
not by walking the tree: the runtime directories (`jobs/`, `temp/`, `output/`)
are git-ignored, and a test run creates hundreds of transient `jobs/` directories
there. Walking would put that noise in every build after a test run; following
git means the archive is exactly the repository's contents — the deliberately
committed evidence stays in, the noise stays out. The walk is kept as a fallback
for a non-git tree (which is what the synthetic-tree tests exercise).

The exclusion set is data (`EXCLUDED_DIR_NAMES`, `EXCLUDED_PATHS`,
`EXCLUDED_FILE_NAMES`, `EXCLUDED_SUFFIXES`), so a reviewer sees exactly what
leaves the machine. `references/` and `models/` are the two with promises
attached and are asserted explicitly. Entries carry a fixed timestamp, so two
builds of one tree are byte-identical. `tests/test_build_archive.py` (31) builds
real zips and checks contents, exclusions, the single top-level directory, and
reproducibility.

### Docs are tested against the code

`docs/DEPLOYMENT.md` names environment variables and runtime directories. The
`A9` doc tests extract the setting-shaped names from the document and assert
each is actually read by `src/`, `scripts/` or `run.py`, and that the key ones
are all present. A renamed variable or a removed setting therefore fails the
suite instead of shipping a doc that lies. `_src_text()` scans `run.py` and
`scripts/` too, because `API_LOG_LEVEL` is read by the launcher, not by `src/`.

### Tests

- `tests/test_deploy_runtime.py` (26) — platform detection from the marker each
  platform really sets, the runtime layout and the env vars it sets, URL rules,
  the VRAM→preset mapping (with a fake GPU), FFmpeg provisioning (including that
  a plain local host never runs `apt-get`), and a **real** uvicorn
  start/health/stop cycle on an ephemeral port.
- `tests/test_deploy_notebooks.py` (29) — both notebooks are valid v4 JSON, every
  code cell parses, they agree except for platform specifics, they configure the
  runtime before importing config, and every `runtime.<name>` they call exists.
- `tests/test_deploy_docs.py` (22) — the doc's variable, directory and preset
  claims match the code.
