import ffmpeg
import logging
import os
from pathlib import Path
import torchaudio
from src.utils import config
import soundfile as sf
import numpy as np

from src.processing.timing import classify_ratio
try:
    import pyrubberband as pyrb
    # Configure pyrubberband to use 'rubberband-program' on Windows
    import sys
    if sys.platform == 'win32':
        pyrb.pyrb.__RUBBERBAND_UTIL = 'rubberband-program'
except ImportError:
    pyrb = None

try:
    import librosa
except ImportError:
    librosa = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_crossfade_window(length: int, fade_type: str = 'linear') -> np.ndarray:
    """
    Generate a crossfade window for smooth audio transitions.
    
    Args:
        length: Number of samples for the fade
        fade_type: Type of fade curve ('linear', 'cosine', 'exponential')
        
    Returns:
        Fade-in curve as numpy array (multiply by audio for fade effect)
    """
    if length <= 0:
        return np.array([])
    
    if fade_type == 'cosine':
        # Smoother S-curve using cosine interpolation
        t = np.linspace(0, np.pi, length)
        return (1 - np.cos(t)) / 2
    elif fade_type == 'exponential':
        # Faster rise at the start, slower at the end
        t = np.linspace(0, 1, length)
        return t ** 0.5
    else:
        # Linear fade
        return np.linspace(0, 1, length)


class AudioSynchronizer:
    # The dub track is assembled at one rate. Every segment is resampled to it,
    # so placement in samples is exact and cannot drift over a long timeline.
    sample_rate = 24000

    def __init__(self, sample_rate: int | None = None):
        if sample_rate:
            self.sample_rate = int(sample_rate)

    def sync_segment(self, audio_path: str | Path, target_duration, output_path: str | Path | None = None):
        """
        Stretches or squeezes the audio at audio_path to fit target_duration.
        """
        if not output_path:
            output_path = str(Path(audio_path).parent / f"synced_{Path(audio_path).name}")
        
        # Get current duration
        try:
            probe = ffmpeg.probe(audio_path)
            audio_info = next(s for s in probe['streams'] if s['codec_type'] == 'audio')
            current_duration = float(audio_info['duration'])
        except Exception as e:
            logger.error(f"Failed to probe audio {audio_path}: {e}")
            return audio_path # Return original if fail
        
        if current_duration <= 0 or target_duration <= 0:
            return audio_path

        speed_factor = current_duration / target_duration
        
        logger.info(f"Syncing {audio_path}: current={current_duration}s, target={target_duration}s, speed={speed_factor:.2f}")

        # 1. Close enough check (0.95 - 1.05) - Just copy
        if 0.95 < speed_factor < 1.05:
            import shutil
            shutil.copy(audio_path, output_path)
            return output_path

        # 2. [Safety Check] Extreme stretching (Fallback to Padding/Trimming)
        if speed_factor < 0.25 or speed_factor > 4.0:
            logger.warning(f"Speed factor {speed_factor:.2f} is extreme. Fallback to padding/trimming.")
            
            w, sr = sf.read(str(audio_path))
            target_samples = int(target_duration * sr)
            
            if w.ndim == 1:
                w = w[:, np.newaxis]
            
            current_samples = w.shape[0]
            
            if current_samples < target_samples:
                # Pad with silence
                padding = target_samples - current_samples
                new_w = np.vstack([w, np.zeros((padding, w.shape[1]))])
            else:
                # Trim
                new_w = w[:target_samples, :]
                
            sf.write(output_path, new_w, sr)
            return output_path

        # 3. [Optimization] Use pyrubberband for natural stretching
        if pyrb:
            try:
                # Load with soundfile
                y, sr = sf.read(str(audio_path))
                if y.ndim > 1:
                     if y.shape[1] > 1:
                        y = y[:, 0] # Force mono
                
                # Ensure float64 for pyrubberband
                y = y.astype(np.float64)

                y_stretched = pyrb.time_stretch(y, sr, speed_factor)
                
                sf.write(output_path, y_stretched, sr)
                return output_path
            except Exception as e:
                logger.error(f"Pyrubberband failed: {e}. Falling back to FFmpeg.")

        # 4. [Fallback] FFmpeg 'atempo' filter
        atempo_filters = []
        remaining_factor = speed_factor
        
        while remaining_factor > 2.0:
            atempo_filters.append("atempo=2.0")
            remaining_factor /= 2.0
        while remaining_factor < 0.5:
            atempo_filters.append("atempo=0.5")
            remaining_factor /= 0.5
             
        if remaining_factor != 1.0:
             atempo_filters.append(f"atempo={remaining_factor}")
        
        try:
            stream = ffmpeg.input(audio_path)
            for f in atempo_filters:
                 stream = stream.filter('atempo', f.split('=')[1])
            
            stream.output(output_path).run(overwrite_output=True, quiet=True)
            return output_path
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg sync failed: {e.stderr.decode() if e.stderr else str(e)}")
            return audio_path

    # ------------------------------------------------------------------
    # Phase 5: authoritative timeline assembly
    # ------------------------------------------------------------------
    def merge_segments(self, segments, total_duration, output_path: str | Path,
                       enable_time_stretch=False, diagnostics=None):
        """
        Lay segments onto a track of exactly `total_duration` seconds.

        Backwards-compatible entry point: accepts the plain
        `{audio_path, start, end}` dicts the pipeline has always passed, and
        also accepts dicts that already carry a `SegmentTiming` under `timing`.

        Correctness properties, each of which the previous implementation
        lacked:
          * audio is placed at its ORIGINAL start, so two speakers who were
            mixed together in the source stay mixed together;
          * a segment is stretched at most once, and never if its backend
            already fitted it;
          * the returned track is exactly `total_duration` long.
        """
        from src.processing.timing import SegmentTiming, build_timings

        ordered = sorted(segments, key=lambda x: x.get('start', 0.0))

        timings: list[SegmentTiming] = []
        paths: dict[int, str] = {}
        for i, seg in enumerate(ordered):
            existing = seg.get('timing')
            if isinstance(existing, SegmentTiming):
                t = existing
            else:
                t = build_timings([seg])[0]
                t.segment_id = seg.get('segment_id', i)
            timings.append(t)
            paths[t.segment_id] = seg['audio_path']

        return self.render_timeline(
            timings, paths, total_duration, output_path,
            enable_time_stretch=enable_time_stretch, diagnostics=diagnostics,
        )

    def render_timeline(self, timings, paths, total_duration, output_path,
                        enable_time_stretch=True, diagnostics=None):
        """
        Render `timings` to a track. Owns placement, correction and diagnostics.

        `paths` maps segment_id to its generated audio file.
        """
        from src.processing.timing_correction import (
            TimingCorrector, allowed_end, apply_stretch, classify_fit,
            trim_with_fade,
        )
        from src.processing.timing import TimingDiagnostics
        from src.audio import mix_loudness

        diagnostics = diagnostics if diagnostics is not None else TimingDiagnostics()
        corrector = TimingCorrector(enable_stretch=enable_time_stretch)

        ordered = sorted(timings, key=lambda t: t.start)
        total_samples = max(1, int(round(total_duration * self.sample_rate)))
        canvas = np.zeros(total_samples, dtype=np.float64)
        silent: list[int] = []
        # One entry per placed segment: (start_sample, level_dbfs, audio).
        # Held until after the loop so each line can be conformed to a shared
        # dialogue level before the segments are summed together.
        placed: list[tuple[int, float, np.ndarray]] = []

        for i, t in enumerate(ordered):
            path = paths.get(t.segment_id)
            wave, sr, err = self._load_segment_audio(path)

            if wave is None or wave.size == 0:
                t.timing_action = "failed"
                t.error = err or "empty_or_unreadable"
                diagnostics.record(t)
                continue

            if sr != self.sample_rate:
                import librosa
                wave = librosa.resample(np.asarray(wave, dtype=np.float32),
                                        orig_sr=sr, target_sr=self.sample_rate)
            wave = np.asarray(wave, dtype=np.float64)
            if wave.ndim > 1:
                wave = wave[:, 0]

            t.generated_duration = len(wave) / float(self.sample_rate)

            # Silent output is worse than no output: it leaves a hole in the dub
            # while looking like success. Flag it and keep going.
            if float(np.max(np.abs(wave))) < 1e-4:
                silent.append(t.segment_id)
                t.timing_action = "silent"
                t.error = "silent_output"
                diagnostics.record(t)
                continue

            next_timing = ordered[i + 1] if i + 1 < len(ordered) else None
            limit = allowed_end(t, next_timing, total_duration)

            # Layer 4: at most one stretch, and none at all if the backend
            # already fitted the segment to its window.
            action, stretch, final_dur = corrector.correct(t, t.generated_duration)
            if abs(stretch - 1.0) > 1e-3:
                wave, method = apply_stretch(wave, self.sample_rate, stretch)
                if method == "none":
                    # The ratio was not achieved; say so instead of pretending.
                    action = "uncorrected"
                else:
                    t.stretch_ratio = stretch
                final_dur = len(wave) / float(self.sample_rate)

            # Layer 5: an overrun may extend into the pause that follows (fine),
            # but it may not reach the next speaker. Only a collision is cut,
            # and only after the stretch has already done what it can.
            #
            # `limit` is an absolute position on the timeline and the wave is a
            # length, so the comparison is on the segment's END.
            at = int(round(t.final_start * self.sample_rate))
            limit_samples = max(1, int(round(limit * self.sample_rate)))
            room = limit_samples - at
            if room > 0 and len(wave) > room:
                t.overflow = (len(wave) - room) / float(self.sample_rate)
                wave = trim_with_fade(wave, room)
                final_dur = len(wave) / float(self.sample_rate)
                t.fit = "collision"
                if action in ("none", "native_fit"):
                    action = "trimmed"
            else:
                t.fit = classify_fit(t.target_duration, final_dur, limit - t.final_start)

            t.final_duration = final_dur
            t.timing_action = action
            t.ratio = (t.generated_duration / t.target_duration) if t.target_duration > 0 else 1.0
            t.band = classify_ratio(t.ratio)

            if at >= total_samples:
                # Starts past the end of the video: nothing to place, but the
                # segment is accounted for rather than silently dropped.
                t.timing_action = "overflow"
                t.fit = "not_placed"
                t.error = "starts_after_video_end"
                diagnostics.record(t)
                continue

            end = min(total_samples, at + len(wave))
            placed_wave = wave[: end - at]
            seg_level = mix_loudness.active_dbfs(placed_wave, sample_rate=self.sample_rate)
            # Equal-power fades stop a click where a cut meets the background.
            # Kept to a few milliseconds so rapid dialogue is not blurred.
            if len(placed_wave) > 3:
                placed_wave = placed_wave * mix_loudness.equal_power_fades(
                    len(placed_wave), self.sample_rate)
            # Held, not summed yet: each segment gets its own loudness gain
            # first, so an overlapping neighbour is not scaled along with it.
            placed.append((at, seg_level, placed_wave))
            diagnostics.record(t)

        if silent:
            logger.warning("%d segment(s) produced silent audio: %s",
                           len(silent), silent[:8])

        # Bound per-line loudness spread without flattening dynamics: the gains
        # are clamped, so a shout stays louder than a whisper while the
        # loud/quiet alternation between two speakers stops being a defect.
        if placed:
            levels = [lvl for _, lvl, _ in placed]
            target = float(np.median(levels))
            gains = mix_loudness.match_dialogue_levels(levels, target)
            adjusted = 0
            for (at, level, seg_wave), gain_db in zip(placed, gains):
                if abs(gain_db) >= 0.05:
                    seg_wave = seg_wave * (10.0 ** (gain_db / 20.0))
                    adjusted += 1
                # Sum now that this segment's own gain is applied. Overlapping
                # speakers simply add, which is what intentional overlap is.
                end = min(total_samples, at + len(seg_wave))
                canvas[at:end] += seg_wave[: end - at]
            logger.info("Dialogue loudness matched to %.1f dB: %d/%d segment(s) adjusted",
                        target, adjusted, len(placed))

        peak_before = float(np.max(np.abs(canvas))) if canvas.size else 0.0
        canvas = mix_loudness.limit_peaks(canvas, mix_loudness.DEFAULT_CEILING)
        if peak_before > mix_loudness.DEFAULT_CEILING:
            logger.info("Peak limiter: %.3f -> %.3f", peak_before,
                        float(np.max(np.abs(canvas))))

        sf.write(str(output_path), canvas.astype(np.float32), self.sample_rate)
        return str(output_path)

    def _load_segment_audio(self, path):
        """
        Read a segment's audio, reporting why it failed rather than raising.

        Returns `(wave, sample_rate, error)`. A missing file, a zero-byte file
        and a corrupt WAV are all recoverable: the segment is dropped from the
        mix and recorded, and the rest of an hour-long render continues.
        """
        if not path:
            return None, 0, "no_path"
        try:
            p = Path(path)
            if not p.exists():
                return None, 0, "missing_file"
            if p.stat().st_size < 100:
                return None, 0, "file_too_small"
        except Exception as exc:  # noqa: BLE001
            return None, 0, f"stat_failed:{exc}"

        try:
            wave, sr = sf.read(str(path), dtype="float32", always_2d=False)
        except Exception as exc:  # noqa: BLE001
            return None, 0, f"unreadable:{type(exc).__name__}"

        return wave, sr, None



if __name__ == "__main__":
    pass
