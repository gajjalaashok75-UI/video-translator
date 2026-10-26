"""
Central language registry — the single source of truth for the dubbing pipeline.

Every language the tool can consume or produce is described once, here:

    name         human-readable label used by the UI
    aliases      additional display names that resolve to this code
    flores       NLLB-200 / flores-200 code, used by translation backends
    translate_to flores code we actually translate INTO when it differs from
                 `flores` because the synthesis engine cannot read the native
                 script (e.g. IndicF5 has no Arabic script, so Urdu is spoken
                 from Devanagari Hindi text)
    engine       the TTS backend that can clone a reference voice for this
                 language (None means "any general backend the user picks")
    synth_lang   language code handed to that engine, when it differs from the
                 UI code
    cps          rough characters-per-second of natural speech; used only as a
                 sanity bound by duration fitting, never as a hard rule
    edge         Edge-TTS voices by gender, for the non-cloning fallback path
    piper        Piper model name, when one exists
    source       True if the language is usable as a transcription source

The derived views (`LANGUAGE_CODE_MAP`, `EDGE_TTS_VOICE_MAP`, `PIPER_MODEL_MAP`)
exist so existing call sites keep working unchanged. New code should use the
helper functions (`engine_for`, `synth_lang`, `source_flores`, ...) instead of
reading the raw dictionaries.

IndicF5 (ai4bharat/IndicF5) natively covers 11 Indian languages and is the
strongest reference-cloning backend available for them. Its vocabulary holds
all 11 Indic scripts plus Latin, but no Arabic script — so Urdu is translated
to Hindi and spoken as Hindi. The audio a listener hears is correct Urdu; the
caveat is lexical, since heavily Persianised formal Urdu drifts toward Hindi
vocabulary.
"""

from __future__ import annotations

from typing import Optional

# ---------------------------------------------------------------------------
# Engine identifiers
# ---------------------------------------------------------------------------
ENGINE_INDICF5 = "indicf5"
ENGINE_XTTS = "xtts"
ENGINE_EDGE = "edge"
ENGINE_PIPER = "piper"
ENGINE_F5 = "f5"
ENGINE_VIBEVOICE = "vibevoice"

# Languages IndicF5 was trained on.
INDICF5_NATIVE = {"as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"}

# Every Indic language we can transcribe or translate, whether or not we can
# speak it. This is the set the translation router treats as "Indic": it decides
# which backend owns a pair, not which engine can render it.
INDIC_CODES = INDICF5_NATIVE | {"ur", "sa", "ne", "sd", "mai", "sat", "kok", "doi", "brx", "mni"}

# Languages Coqui XTTS-v2 supports (used to validate a cloning request).
XTTS_NATIVE = {
    "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru",
    "nl", "cs", "ar", "zh", "ja", "hu", "ko", "hi",
}

# Languages Edge-TTS can speak (filled from the registry below).
EDGE_NATIVE: set[str] = set()

# The priority target languages, in the order the UI should show them.
PRIORITY_INDIC = ["te", "hi", "ta", "kn", "ml", "bn", "mr", "gu", "pa", "or", "as"]


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------
# Order matters: `get_language_name()` returns the first registered display name
# for a code, so keep the historical ordering of the pre-existing languages and
# append the new Indic entries after them.
LANGUAGES: dict[str, dict] = {
    "en": dict(
        name="English", aliases=["English"], flores="eng_Latn",
        engine=None, cps=15.5, source=True,
        edge={"Female": ["en-US-AriaNeural", "en-US-JennyNeural", "en-GB-SoniaNeural"],
              "Male": ["en-US-GuyNeural", "en-US-ChristopherNeural", "en-GB-RyanNeural"]},
        piper="en_US-lessac-high",
    ),
    "es": dict(
        name="Spanish", aliases=["Spanish"], flores="spa_Latn",
        engine=None, cps=15.0, source=True,
        edge={"Female": ["es-ES-ElviraNeural", "es-MX-DaliaNeural"],
              "Male": ["es-ES-AlvaroNeural", "es-MX-JorgeNeural"]},
        piper="es_ES-sharvard-medium",
    ),
    "fr": dict(
        name="French", aliases=["French"], flores="fra_Latn",
        engine=None, cps=15.0, source=True,
        edge={"Female": ["fr-FR-DeniseNeural", "fr-CA-SylvieNeural"],
              "Male": ["fr-FR-HenriNeural", "fr-CA-JeanNeural"]},
        piper="fr_FR-siwis-medium",
    ),
    "de": dict(
        name="German", aliases=["German"], flores="deu_Latn",
        engine=None, cps=14.5, source=True,
        edge={"Female": ["de-DE-KatjaNeural", "de-AT-IngridNeural"],
              "Male": ["de-DE-ConradNeural", "de-AT-JonasNeural"]},
        piper="de_DE-thorsten-medium",
    ),
    "it": dict(
        name="Italian", aliases=["Italian"], flores="ita_Latn",
        engine=None, cps=15.0, source=True,
        edge={"Female": ["it-IT-ElsaNeural", "it-IT-IsabellaNeural"],
              "Male": ["it-IT-DiegoNeural", "it-IT-GiuseppeNeural"]},
        piper="it_IT-riccardo-x_low",
    ),
    "pt": dict(
        name="Portuguese", aliases=["Portuguese"], flores="por_Latn",
        engine=None, cps=15.0, source=True,
        edge={"Female": ["pt-BR-FranciscaNeural", "pt-PT-RaquelNeural"],
              "Male": ["pt-BR-AntonioNeural", "pt-PT-DuarteNeural"]},
    ),
    "pl": dict(
        name="Polish", aliases=["Polish"], flores="pol_Latn",
        engine=None, cps=14.0, source=True,
        edge={"Female": ["pl-PL-ZofiaNeural", "pl-PL-AgnieszkaNeural"],
              "Male": ["pl-PL-MarekNeural"]},
    ),
    "tr": dict(
        name="Turkish", aliases=["Turkish"], flores="tur_Latn",
        engine=None, cps=14.0, source=True,
        edge={"Female": ["tr-TR-EmelNeural"], "Male": ["tr-TR-AhmetNeural"]},
    ),
    "ru": dict(
        name="Russian", aliases=["Russian"], flores="rus_Cyrl",
        engine=None, cps=14.0, source=True,
        edge={"Female": ["ru-RU-SvetlanaNeural", "ru-RU-DariyaNeural"],
              "Male": ["ru-RU-DmitryNeural"]},
    ),
    "nl": dict(
        name="Dutch", aliases=["Dutch"], flores="nld_Latn",
        engine=None, cps=14.5, source=True,
        edge={"Female": ["nl-NL-ColetteNeural", "nl-NL-FennaNeural"],
              "Male": ["nl-NL-MaartenNeural"]},
    ),
    "cs": dict(
        name="Czech", aliases=["Czech"], flores="ces_Latn",
        engine=None, cps=14.0, source=True,
        edge={"Female": ["cs-CZ-VlastaNeural"], "Male": ["cs-CZ-AntoninNeural"]},
    ),
    "ar": dict(
        name="Arabic", aliases=["Arabic"], flores="arb_Arab",
        engine=None, cps=13.0, source=True,
        edge={"Female": ["ar-SA-ZariyahNeural", "ar-EG-SalmaNeural"],
              "Male": ["ar-SA-HamedNeural", "ar-EG-ShakirNeural"]},
    ),
    "zh": dict(
        name="Chinese (Simplified)", aliases=["Chinese (Simplified)", "Chinese"],
        flores="zho_Hans", engine=None, synth_lang="zh", cps=6.5, source=True,
        edge={"Female": ["zh-CN-XiaoxiaoNeural", "zh-CN-XiaoyiNeural", "zh-CN-XiaochenNeural"],
              "Male": ["zh-CN-YunxiNeural", "zh-CN-YunjianNeural", "zh-CN-YunyeNeural"]},
    ),
    "ja": dict(
        name="Japanese", aliases=["Japanese"], flores="jpn_Jpan",
        engine=None, cps=7.0, source=True,
        edge={"Female": ["ja-JP-NanamiNeural", "ja-JP-AoiNeural"],
              "Male": ["ja-JP-KeitaNeural", "ja-JP-DaichiNeural"]},
    ),
    "ko": dict(
        name="Korean", aliases=["Korean"], flores="kor_Hang",
        engine=None, cps=8.0, source=True,
        edge={"Female": ["ko-KR-SunHiNeural", "ko-KR-JiMinNeural"],
              "Male": ["ko-KR-InJoonNeural", "ko-KR-BongJinNeural"]},
    ),
    "hi": dict(
        name="Hindi", aliases=["Hindi"], flores="hin_Deva",
        engine=ENGINE_INDICF5, cps=13.5, source=True,
        edge={"Female": ["hi-IN-SwaraNeural"], "Male": ["hi-IN-MadhurNeural"]},
    ),

    # ---- Priority Indian languages: reference-cloned by IndicF5 ----
    "te": dict(
        name="Telugu", aliases=["Telugu"], flores="tel_Telu",
        engine=ENGINE_INDICF5, cps=12.5, source=True,
        edge={"Female": ["te-IN-ShrutiNeural"], "Male": ["te-IN-MohanNeural"]},
    ),
    "ta": dict(
        name="Tamil", aliases=["Tamil"], flores="tam_Taml",
        engine=ENGINE_INDICF5, cps=12.0, source=True,
        edge={"Female": ["ta-IN-PallaviNeural"], "Male": ["ta-IN-ValluvarNeural"]},
    ),
    "kn": dict(
        name="Kannada", aliases=["Kannada"], flores="kan_Knda",
        engine=ENGINE_INDICF5, cps=12.5, source=True,
        edge={"Female": ["kn-IN-SapnaNeural"], "Male": ["kn-IN-GaganNeural"]},
    ),
    "ml": dict(
        name="Malayalam", aliases=["Malayalam"], flores="mal_Mlym",
        engine=ENGINE_INDICF5, cps=12.0, source=True,
        edge={"Female": ["ml-IN-SobhanaNeural"], "Male": ["ml-IN-MidhunNeural"]},
    ),
    "bn": dict(
        name="Bengali", aliases=["Bengali"], flores="ben_Beng",
        engine=ENGINE_INDICF5, cps=13.0, source=True,
        edge={"Female": ["bn-IN-TanishaaNeural"], "Male": ["bn-IN-BashkarNeural"]},
    ),
    "mr": dict(
        name="Marathi", aliases=["Marathi"], flores="mar_Deva",
        engine=ENGINE_INDICF5, cps=13.0, source=True,
        edge={"Female": ["mr-IN-AarohiNeural"], "Male": ["mr-IN-ManoharNeural"]},
    ),
    "gu": dict(
        name="Gujarati", aliases=["Gujarati"], flores="guj_Gujr",
        engine=ENGINE_INDICF5, cps=13.0, source=True,
        edge={"Female": ["gu-IN-DhwaniNeural"], "Male": ["gu-IN-NiranjanNeural"]},
    ),
    "pa": dict(
        name="Punjabi", aliases=["Punjabi"], flores="pan_Guru",
        engine=ENGINE_INDICF5, cps=13.0, source=True,
        edge={"Female": ["pa-IN-GurpreetNeural"], "Male": ["pa-IN-OjasNeural"]},
    ),
    "or": dict(
        name="Odia", aliases=["Odia"], flores="ory_Orya",
        engine=ENGINE_INDICF5, cps=12.5, source=True,
        edge={"Female": ["or-IN-SubhasiniNeural"], "Male": ["or-IN-SukantNeural"]},
    ),
    "as": dict(
        name="Assamese", aliases=["Assamese"], flores="asm_Beng",
        engine=ENGINE_INDICF5, cps=13.0, source=True,
        edge={"Female": ["as-IN-YashicaNeural"], "Male": ["as-IN-PriyomNeural"]},
    ),
}

# Source-side only: Whisper may detect these even though we cannot synthesize
# them. Used to resolve a translation source code, never a synthesis target.
EXTRA_SOURCE_FLORES = {
    "ur": "urd_Arab",
    "th": "tha_Thai", "vi": "vie_Latn", "id": "ind_Latn", "ms": "zsm_Latn",
    "sw": "swh_Latn", "uk": "ukr_Cyrl", "ro": "ron_Latn", "el": "ell_Grek",
    "hu": "hun_Latn", "sv": "swe_Latn", "da": "dan_Latn", "fi": "fin_Latn",
    "no": "nob_Latn", "he": "heb_Hebr", "fa": "pes_Arab", "ne": "npi_Deva",
    "si": "sin_Sinh",
}

# Extra source-only display names, so the UI can offer them as a source.
EXTRA_SOURCE_NAMES = {
    "ur": "Urdu",
}


class UnsupportedLanguage(ValueError):
    """Raised when a language is requested for a stage that cannot handle it."""


# ---------------------------------------------------------------------------
# Derived views (kept for backwards compatibility with existing call sites)
# ---------------------------------------------------------------------------
LANGUAGE_CODE_MAP: dict[str, str] = {"Auto Detect": "auto"}
for _code, _entry in LANGUAGES.items():
    for _alias in _entry["aliases"]:
        LANGUAGE_CODE_MAP[_alias] = _code
del _code, _entry, _alias

EDGE_TTS_VOICE_MAP: dict[str, dict] = {}
for _code, _entry in LANGUAGES.items():
    if _entry.get("edge"):
        EDGE_TTS_VOICE_MAP[_code] = _entry["edge"]
        EDGE_NATIVE.add(_code)
del _code, _entry

PIPER_MODEL_MAP: dict[str, str] = {
    code: entry["piper"] for code, entry in LANGUAGES.items() if entry.get("piper")
}


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------
def normalize(code: str) -> str:
    """
    Resolve a language code to a registry key.

    Accepts short codes ('hi'), region-tagged codes ('hi-IN'), underscored
    codes ('hi_IN') and full display names ('Hindi'). Returns 'auto' for
    auto-detect. Raises `UnsupportedLanguage` for anything unknown.
    """
    if not code:
        raise UnsupportedLanguage("empty language code")

    c = str(code).strip()
    if c.lower() == "auto":
        return "auto"

    # Display name first, so "Chinese (Simplified)" wins over code splitting.
    if c in LANGUAGE_CODE_MAP:
        return LANGUAGE_CODE_MAP[c]

    low = c.lower()
    if low in LANGUAGES or low in EXTRA_SOURCE_FLORES:
        return low

    base = low.replace("_", "-").split("-")[0]
    if base in LANGUAGES or base in EXTRA_SOURCE_FLORES:
        return base

    raise UnsupportedLanguage(f"unknown language code: {code!r}")


def get_language_code(name: str) -> str:
    """
    Returns the internal language code for a given display name.

    Preserves the original lenient contract: an already-valid code passes
    through, and anything unrecognised falls back to 'en' rather than raising.
    """
    if name in LANGUAGE_CODE_MAP.values():
        return name
    return LANGUAGE_CODE_MAP.get(name, "en")


def get_language_name(code: str) -> str:
    """Returns the display name for a given language code (reverse lookup)."""
    for name, lang_code in LANGUAGE_CODE_MAP.items():
        if lang_code == code:
            return name
    return "English"  # Default fallback


def is_known(code: str) -> bool:
    """True if the code names a language we can translate or synthesize."""
    try:
        return normalize(code) != "auto"
    except UnsupportedLanguage:
        return False


def is_target(code: str) -> bool:
    """True if the code can be used as a dubbing target."""
    try:
        return normalize(code) in LANGUAGES
    except UnsupportedLanguage:
        return False


def is_indic(code: str) -> bool:
    """
    True if the code names an Indic language.

    Used by the translation router to pick a backend, not to decide whether we
    can synthesize the language. `is_target()` is the synthesis question.
    """
    try:
        return normalize(code) in INDIC_CODES
    except UnsupportedLanguage:
        return False


def engine_for(code: str) -> Optional[str]:
    """
    The reference-cloning engine that owns this language, or None.

    None means "no dedicated engine": the user's chosen general backend
    (Edge / Piper / XTTS / F5 / VibeVoice) handles it.
    """
    return LANGUAGES[normalize(code)].get("engine")


def synth_lang(code: str) -> str:
    """Language code handed to the TTS engine (e.g. Chinese -> 'zh')."""
    entry = LANGUAGES[normalize(code)]
    return entry.get("synth_lang", normalize(code))


def chars_per_second(code: str) -> float:
    """Rough natural speaking rate, used as a sanity bound on duration fitting."""
    entry = LANGUAGES.get(normalize(code))
    return entry.get("cps", 14.0) if entry else 14.0


def chars(text: str) -> int:
    """
    Character count used for speech-length estimation.

    Deliberately CHARACTERS, not UTF-8 bytes. A Devanagari character is 3 bytes
    against 1 for Latin, so a byte count calibrated on an English reference
    over-estimates every Hindi line by roughly 3x. The `cps` table is also in
    characters, so both sides of the comparison must agree.
    """
    return max(1, len((text or "").strip()))


def source_flores(code: str) -> str:
    """flores-200 code to use as the TRANSLATION SOURCE."""
    c = normalize(code)
    if c in LANGUAGES:
        return LANGUAGES[c]["flores"]
    return EXTRA_SOURCE_FLORES[c]


def target_flores(code: str, for_speech: bool = True) -> str:
    """
    flores-200 code to translate INTO for this target.

    `for_speech` picks the script the voice model can read, which for a
    language like Urdu is not its own. Text that will only ever be read —
    subtitles, a translated script — wants the language's own script.
    """
    c = normalize(code)
    if c not in LANGUAGES:
        raise UnsupportedLanguage(f"{code!r} is not a supported dubbing target")
    entry = LANGUAGES[c]
    return entry.get("translate_to", entry["flores"]) if for_speech else entry["flores"]


def display_name(code: str) -> str:
    c = normalize(code)
    if c in LANGUAGES:
        return LANGUAGES[c]["name"]
    if c in EXTRA_SOURCE_NAMES:
        return EXTRA_SOURCE_NAMES[c]
    return get_language_name(code)


def supports_edge(code: str) -> bool:
    return normalize(code) in EDGE_NATIVE


def supported_targets() -> list[dict]:
    """UI-facing target list: priority Indian languages first, then the rest."""
    def order(code: str) -> tuple:
        if code in PRIORITY_INDIC:
            return (0, PRIORITY_INDIC.index(code))
        return (1, list(LANGUAGES).index(code))

    return [
        {"code": c, "name": LANGUAGES[c]["name"], "engine": LANGUAGES[c].get("engine")}
        for c in sorted(LANGUAGES, key=order)
    ]


def supported_sources() -> list[dict]:
    """UI-facing source list: auto-detect, then every registered language."""
    out = [{"code": "auto", "name": "Auto Detect"}]
    out += [
        {"code": c, "name": LANGUAGES[c]["name"]}
        for c in LANGUAGES if LANGUAGES[c].get("source")
    ]
    out += [
        {"code": c, "name": name} for c, name in EXTRA_SOURCE_NAMES.items()
    ]
    return out


def source_choices() -> list[str]:
    return [entry["name"] for entry in supported_sources()]


def target_choices() -> list[str]:
    return [entry["name"] for entry in supported_targets()]


def validate_tts_pair(target_code: str, model: str) -> None:
    """
    Fail early with a readable message when a target language cannot be spoken
    by the requested TTS backend.

    Raises `UnsupportedLanguage`; callers decide whether to fall back or abort.
    """
    c = normalize(target_code)
    if model == ENGINE_INDICF5:
        if c not in INDICF5_NATIVE:
            raise UnsupportedLanguage(
                f"IndicF5 does not support target language '{c}' "
                f"({display_name(c)}). It speaks: "
                f"{', '.join(sorted(INDICF5_NATIVE))}. "
                f"Use a general backend (edge/xtts/f5) for this language."
            )
    elif model == ENGINE_XTTS:
        if c not in XTTS_NATIVE:
            raise UnsupportedLanguage(
                f"XTTS-v2 does not support target language '{c}' "
                f"({display_name(c)})."
            )
    elif model == ENGINE_PIPER:
        if c not in PIPER_MODEL_MAP:
            raise UnsupportedLanguage(
                f"No Piper voice is configured for target language '{c}' "
                f"({display_name(c)})."
            )
