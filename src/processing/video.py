import ffmpeg
import logging
import subprocess
from pathlib import Path
from src.utils import config
import torchaudio
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoProcessor:
    def extract_audio(self, video_path: str | Path, output_path: str | Path | None = None):
        if not output_path:
            output_path = str(Path(video_path).with_suffix(".wav"))
        
        try:
            (
                ffmpeg
                .input(video_path)
                .output(output_path, ac=1, ar=24000) # Mono 24k for consistency with TTS
                .overwrite_output()
                .run(quiet=True)
            )
            return output_path
        except ffmpeg.Error as e:
            logger.error(f"Extract audio failed: {e.stderr.decode() if e.stderr else str(e)}")
            return None

    def mix_tracks(self, vocal_track: str | Path, background_track: str | Path,
                   output_path: str | Path,
                   vocal_volume: float | None = None, bg_volume: float | None = None):
        """
        Mix the dubbed dialogue over the original background (music + effects).

        Levels are chosen, not hard-coded: the background is placed a set
        distance below the dialogue (see `src.audio.mix_loudness`) so the
        speech stays intelligible without burying the music, and only a
        clip-risk peak is corrected. The previous fixed `vocal=1.0, bg=0.8`
        left the mix ~32 dB lopsided in favour of the generated speech.

        Passing explicit `vocal_volume`/`bg_volume` still works and overrides
        the automatic choice, so existing callers keep their behaviour.
        """
        try:
            import soundfile as sf
            import numpy as np
            from src.audio import mix_loudness

            v_np, v_sr = sf.read(str(vocal_track))
            v_np = v_np[np.newaxis, :] if v_np.ndim == 1 else v_np.T
            v_wav = torch.from_numpy(v_np).float()

            b_np, b_sr = sf.read(str(background_track))
            b_np = b_np[np.newaxis, :] if b_np.ndim == 1 else b_np.T
            b_wav = torch.from_numpy(b_np).float()

            if v_sr != b_sr:
                resampler = torchaudio.transforms.Resample(b_sr, v_sr)
                b_wav = resampler(b_wav)
                b_sr = v_sr

            # Pad to a common length; the background is the reference timeline.
            max_len = max(v_wav.shape[1], b_wav.shape[1])
            if v_wav.shape[1] < max_len:
                v_wav = torch.cat([v_wav, torch.zeros(v_wav.shape[0], max_len - v_wav.shape[1])], dim=1)
            if b_wav.shape[1] < max_len:
                b_wav = torch.cat([b_wav, torch.zeros(b_wav.shape[0], max_len - b_wav.shape[1])], dim=1)

            # Per-channel levels differ; average to mono for the level decision.
            d_mono = v_wav.mean(0).numpy()
            b_mono = b_wav.mean(0).numpy()
            auto_d_gain, auto_b_gain = mix_loudness.estimate_mix_gains(d_mono, b_mono)
            d_gain_db = auto_d_gain if vocal_volume is None else 20.0 * np.log10(max(vocal_volume, 1e-6))
            b_gain_db = auto_b_gain if bg_volume is None else 20.0 * np.log10(max(bg_volume, 1e-6))

            final_mix = (v_wav * (10.0 ** (d_gain_db / 20.0))) + (b_wav * (10.0 ** (b_gain_db / 20.0)))

            # Correct only a clip-risk peak, and only by the amount needed.
            mix_np = final_mix.detach().cpu().numpy()
            mix_np = mix_loudness.limit_peaks(mix_np, mix_loudness.DEFAULT_CEILING)

            logger.info(
                "Mix levels: dialogue %+.1f dB, background %+.1f dB (dialogue %.1f dB, bg %.1f dB)",
                d_gain_db, b_gain_db, mix_loudness.active_dbfs(d_mono), mix_loudness.active_dbfs(b_mono))

            sf.write(str(output_path), mix_np.T, v_sr)
            return output_path
        except Exception as e:
            logger.error(f"Mixing failed: {e}")
            return None

    def _subtitle_codec(self, container_ext: str) -> str:
        """
        The subtitle codec a container accepts.

        There is no universal answer: MP4/MOV want `mov_text`, WebM wants
        `webvtt`, Matroska takes `srt`. Naming the wrong one is not a warning —
        the mux fails outright — so the choice is made from the container
        extension the same way the audio/video codecs already are.
        """
        if container_ext == ".webm":
            return "webvtt"
        if container_ext in (".mkv",):
            return "srt"
        return "mov_text"

    def _escape_subtitle_path(self, path: str | Path) -> str:
        """
        A path that is safe to interpolate into a `subtitles=` filtergraph.

        The filter parser treats `:` and `\\` as syntax and `'` as quoting, so a
        path containing any of them would otherwise be read as filter options —
        at best the render fails, at worst a crafted filename alters the graph.
        The value is wrapped in single quotes with the escapes ffmpeg documents.
        """
        text = str(path)
        text = text.replace("\\", "\\\\")
        text = text.replace(":", "\\:")
        text = text.replace("'", "'\\''")
        return f"'{text}'"

    def add_subtitles(self, video_path: str | Path, subtitle_path: str | Path,
                      output_video_path: str | Path, burn: bool = False):
        """
        Add a subtitle track to a video, either soft (toggleable) or burned in.

        Soft subtitles are muxed as a selectable track: the picture is copied
        untouched and the viewer can turn the track on or off. Burned subtitles
        are rendered into the frames by libass, which means the video must be
        re-encoded and the result can never be turned off — a deliberate,
        visible change to the picture.

        Raises `ffmpeg.Error` on failure rather than returning None: the caller
        records a completed artifact from a returned path, so a silent failure
        would publish a video that has no subtitles while claiming it does.
        """
        out_ext = Path(str(output_video_path)).suffix.lower()
        sub_path = self._escape_subtitle_path(subtitle_path)

        if burn:
            # Burn-in re-encodes the picture through the `subtitles` filter, so
            # it goes through an explicit argv rather than the ffmpeg-python
            # builder. The builder escapes the already-escaped path a second
            # time (its filtergraph writer doubles the backslashes this method
            # added), and libass then looks for a file whose name really
            # contains them. One layer of escaping, applied once, is what the
            # filter parser expects.
            if out_ext == ".webm":
                codec_args = ["-c:v", "libvpx-vp9", "-crf", "30", "-b:v", "0",
                              "-c:a", "libopus"]
            else:
                codec_args = ["-c:v", "libx264", "-preset", "fast", "-c:a", "copy"]
            cmd = [
                "ffmpeg", "-y", "-v", "error",
                "-i", str(video_path),
                "-vf", f"subtitles={sub_path}",
                *codec_args,
                str(output_video_path),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise ffmpeg.Error(
                    "ffmpeg", cmd,
                    (proc.stderr or "").encode("utf-8", "replace"))
            return output_video_path

        # Soft track: stream-copy video, re-encode only the audio if needed, and
        # add the subtitle stream.
        codec = self._subtitle_codec(out_ext)
        audio_args = ({"acodec": "libopus"} if out_ext == ".webm"
                      else {"acodec": "aac", "strict": "experimental"})
        (
            ffmpeg
            .output(
                ffmpeg.input(video_path).video,
                ffmpeg.input(video_path).audio,
                ffmpeg.input(subtitle_path),
                # The builder only accepts a str filename; a Path here raises
                # "A filename must be provided" from inside the library.
                str(output_video_path),
                vcodec="copy",
                **audio_args,
                **{"scodec": codec},
            )
            .overwrite_output()
            .run(quiet=True)
        )
        return output_video_path

    def replace_audio(self, video_path: str | Path, new_audio_path: str | Path, output_video_path: str | Path):
        # Codecs must match the *output* container, which is derived from the
        # output filename (the pipeline keeps the source extension). h264/aac
        # are invalid inside WebM, so a .webm source would otherwise fail to mux.
        ext = Path(str(output_video_path)).suffix.lower()
        if ext == ".webm":
            copy_args = {"vcodec": "copy", "acodec": "libopus"}
            reencode_args = {"vcodec": "libvpx-vp9", "acodec": "libopus", "crf": "30", "b:v": "0"}
        else:
            copy_args = {"vcodec": "copy", "acodec": "aac", "strict": "experimental"}
            reencode_args = {"vcodec": "libx264", "acodec": "aac", "strict": "experimental", "preset": "fast"}

        try:
            video = ffmpeg.input(video_path)
            audio = ffmpeg.input(new_audio_path)

            # Attempt 1: Copy video stream (Fastest)
            try:
                (
                    ffmpeg
                    .output(video.video, audio, output_video_path, **copy_args)
                    .overwrite_output()
                    .run(quiet=True)
                )
            except ffmpeg.Error as e:
                logger.warning(f"Replace audio (copy) failed: {e.stderr.decode() if e.stderr else str(e)}. Retrying with re-encode...")
                # Attempt 2: Re-encode video (Compatible)
                (
                    ffmpeg
                    .output(video.video, audio, output_video_path, **reencode_args)
                    .overwrite_output()
                    .run(quiet=True)
                )

            return output_video_path
        except ffmpeg.Error as e:
             logger.error(f"Replace audio failed (both copy and re-encode): {e.stderr.decode() if e.stderr else str(e)}")
             # Raise rather than return None: a None result must not be recorded
             # as a completed final artifact by the caller.
             raise

if __name__ == "__main__":
    pass
