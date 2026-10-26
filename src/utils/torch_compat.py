"""
torchaudio / torch compatibility shims.

Two independent problems are handled here, both of which only bite on recent
PyTorch releases and both of which produce confusing failures far from the
actual cause:

1. `torchaudio.load` routing through TorchCodec. From torchaudio 2.11 the
   bundled decoding backends were removed and `load` was rerouted through
   TorchCodec, which is not a dependency of this project. Any third-party
   library that calls `torchaudio.load` internally (Coqui XTTS does) then dies
   with "TorchCodec is required for load_with_torchcodec". We hand back the
   soundfile-backed implementation torchaudio used to ship, which covers every
   format this pipeline feeds it.

2. `torch.load` defaulting to `weights_only=True`. PyTorch 2.6 flipped the
   default, so checkpoints that pickle their own config dataclasses (Coqui's
   `XttsConfig`, pyannote's `TorchVersion`) raise `UnsupportedGlobal`. The fix
   is a narrow allowlist of the specific classes, not disabling the protection
   globally.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_TORCHAUDIO_PATCHED = False


def patch_torchaudio_load() -> bool:
    """
    Give `torchaudio.load` a soundfile-backed implementation when torchaudio
    has no working decoding backend of its own.

    Idempotent. Returns True if a patch is (or was already) in place.
    """
    global _TORCHAUDIO_PATCHED
    if _TORCHAUDIO_PATCHED:
        return True

    try:
        import torch  # noqa: F401  (must load before torchaudio)
        import torchaudio
    except ImportError:
        return False

    # Probe: does the installed torchaudio have a real decoding backend?
    try:
        torchaudio.load(__file__)
        _TORCHAUDIO_PATCHED = True  # it errored for a non-TorchCodec reason
        return True
    except Exception as exc:  # noqa: BLE001 - we only care about the message
        if "TorchCodec" not in str(exc):
            _TORCHAUDIO_PATCHED = True
            return True

    try:
        import soundfile as sf
    except ImportError:
        logger.warning(
            "torchaudio has no decoding backend and soundfile is unavailable, "
            "so torchaudio.load cannot be repaired. Install soundfile."
        )
        return False

    def _load(uri, frame_offset=0, num_frames=-1, normalize=True,
              channels_first=True, **_):
        data, sr = sf.read(
            str(uri), dtype="float32", always_2d=True,
            start=int(frame_offset),
            frames=(-1 if num_frames in (-1, None) else int(num_frames)),
        )
        tensor = torch.from_numpy(data)  # (frames, channels)
        return (tensor.T.contiguous() if channels_first else tensor), sr

    torchaudio.load = _load
    _TORCHAUDIO_PATCHED = True
    logger.info("Patched torchaudio.load to use soundfile (TorchCodec not installed).")
    return True


def allow_torch_load_globals() -> None:
    """
    Allowlist checkpoint config classes that PyTorch >= 2.6 refuses to unpickle
    under the new `weights_only=True` default.

    Only classes from checkpoints this project already trusts (downloaded into
    the local HF cache) are added.
    """
    try:
        import torch
    except ImportError:
        return

    if not hasattr(torch.serialization, "add_safe_globals"):
        return  # older torch: weights_only was not the default

    allowed = []
    for module, names in (
        ("TTS.tts.configs.xtts_config", ["XttsConfig"]),
        ("TTS.tts.models.xtts", ["XttsAudioConfig", "XttsArgs"]),
        ("TTS.config.shared_configs", ["BaseDatasetConfig", "BaseAudioConfig"]),
    ):
        try:
            mod = __import__(module, fromlist=names)
            allowed += [getattr(mod, n) for n in names if hasattr(mod, n)]
        except Exception:  # noqa: BLE001 - module absent is the normal case
            continue

    if allowed:
        try:
            torch.serialization.add_safe_globals(allowed)
            logger.debug("Allowlisted %d torch.load global(s) for Coqui configs.", len(allowed))
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not allowlist torch.load globals: %s", exc)
