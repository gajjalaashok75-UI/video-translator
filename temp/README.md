# Intermediates for the His Girl Friday run

Scratch audio the pipeline produced for `jobs/hgf_scene-*`. Kept because the
Phase 7 validation measures against exactly these files.

- `hgf_scene_full.wav` — audio extracted from the sample video.
- `vocals.wav`, `accompaniment.wav` — Demucs stems (speech / music+effects).
- `hgf_scene_merged_speech.wav` — TTS lines placed on the timeline (the dub).
- `hgf_scene_final_mix.wav` — dub + background after the levelled mix.
- `v/`, `video_merged_speech.wav`, `hgf_scene_profiles/` — per-segment TTS
  takes and speaker profiles from earlier test runs.

These are regenerable; they are here for inspection, not as inputs.
