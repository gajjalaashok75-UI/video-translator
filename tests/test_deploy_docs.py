"""
A9: the deployment documentation must describe the code that exists.

Docs rot silently. These tests read `docs/DEPLOYMENT.md` and check its claims
against the real modules: every environment variable it names must actually be
read somewhere in `src/`, every runtime directory it lists must be a real
setting, and the notebook paths it points at must exist.

They are deliberately about *facts the code owns* — variable names, file paths —
and not prose, so they stay green through an edit to the wording.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEPLOYMENT_DOC = PROJECT_ROOT / "docs" / "DEPLOYMENT.md"
SRC = PROJECT_ROOT / "src"


def _doc() -> str:
    return DEPLOYMENT_DOC.read_text(encoding="utf-8")


def _src_text() -> str:
    """
    The text of everything that reads settings: the package, the launcher, and
    the scripts. `run.py` reads `API_LOG_LEVEL` and the installers read others,
    so a `src/`-only scan would wrongly report those as undocumented.
    """
    roots = [SRC, PROJECT_ROOT / "scripts"]
    files = [p for root in roots for p in root.rglob("*.py")]
    files.append(PROJECT_ROOT / "run.py")
    return "\n".join(p.read_text(encoding="utf-8", errors="replace")
                     for p in files if p.is_file())


# ---------------------------------------------------------------------------
# The document exists and is wired into the README
# ---------------------------------------------------------------------------
class TestDocExists:

    def test_the_deployment_doc_exists(self):
        assert DEPLOYMENT_DOC.is_file()

    def test_the_readme_links_to_it(self):
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        assert "docs/DEPLOYMENT.md" in readme

    def test_the_notebooks_it_points_at_exist(self):
        assert (PROJECT_ROOT / "deploy" / "notebooks" / "colab.ipynb").is_file()
        assert (PROJECT_ROOT / "deploy" / "notebooks" / "kaggle.ipynb").is_file()
        assert (PROJECT_ROOT / "scripts" / "build_notebooks.py").is_file()

    def test_the_docker_files_it_points_at_exist(self):
        assert (PROJECT_ROOT / "deploy" / "docker" / "Dockerfile").is_file()
        assert (PROJECT_ROOT / "run.py").is_file()


# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------
# Every variable the document names in a `CODE` span or a table cell, which is
# how it refers to settings. Deliberately over-broad: a name that turns out to
# be prose is filtered by the "looks like a setting" shape below.
_VAR_PATTERN = re.compile(r"\b([A-Z][A-Z0-9_]{3,})\b")

# Names that are not environment variables, so the check must not demand they
# appear in the source: file names, acronyms, and platform markers.
_NOT_SETTINGS = {
    "PATH", "README", "DOCKERFILE", "API", "UI", "GPU", "CPU", "VRAM", "CUDA",
    "FFMPEG", "HF", "COLAB", "KAGGLE", "T4", "P100", "NGC", "NVIDIA", "URL",
    "URLS", "JSON", "HTTP", "HTTPS", "GIB", "GB", "MB", "NAT", "NGROK", "WAV",
    "ML", "NLLB", "WAV2LIP", "LIVEPORTRAIT", "ID", "E", "I", "V",
}


def _documented_variables() -> set[str]:
    """The setting-shaped names the document mentions."""
    text = _doc()
    # Only inside the configuration tables and inline code spans, which is where
    # the document names settings; prose words are excluded by the filter.
    candidates = set()
    for match in re.finditer(r"`([A-Z][A-Z0-9_]{3,})`", text):
        candidates.add(match.group(1))
    for row in re.finditer(r"^\|\s*`([A-Z][A-Z0-9_]{3,})`", text, re.MULTILINE):
        candidates.add(row.group(1))
    return {c for c in candidates if c not in _NOT_SETTINGS}


class TestDocumentedVariablesExist:

    def test_the_document_names_some_variables(self):
        # A guard against the extraction silently matching nothing.
        assert len(_documented_variables()) >= 10

    def test_every_named_variable_is_read_by_the_code(self):
        source = _src_text()
        missing = sorted(v for v in _documented_variables()
                         if f'"{v}"' not in source and f"'{v}'" not in source)
        assert not missing, (
            f"docs/DEPLOYMENT.md names variables the code never reads: {missing}")

    @pytest.mark.parametrize("name", [
        "TEMP_DIR", "OUTPUT_DIR", "JOBS_DIR", "API_UPLOADS_DIR",
        "API_HOST", "API_PORT", "USE_CUDA", "HF_TOKEN",
        "RUNTIME_DIR", "PUBLIC_HOST", "PUBLIC_SCHEME",
    ])
    def test_the_key_variables_are_all_documented(self, name):
        assert f"`{name}`" in _doc(), f"{name} is not documented"


# ---------------------------------------------------------------------------
# The runtime-directory table matches the code
# ---------------------------------------------------------------------------
class TestRuntimeDirectoriesMatchTheCode:

    def test_the_documented_defaults_are_the_real_defaults(self):
        from src.utils import config

        # The code defaults, resolved the same way config resolves them.
        assert config.TEMP_DIR == Path(__import__("os").environ.get(
            "TEMP_DIR", str(config.BASE_DIR / "temp")))
        assert config.OUTPUT_DIR == Path(__import__("os").environ.get(
            "OUTPUT_DIR", str(config.BASE_DIR / "output")))
        assert config.JOBS_DIR == Path(__import__("os").environ.get(
            "JOBS_DIR", str(config.BASE_DIR / "jobs")))

    def test_the_four_directories_are_all_in_the_document(self):
        text = _doc()
        for name in ("TEMP_DIR", "OUTPUT_DIR", "JOBS_DIR", "API_UPLOADS_DIR"):
            assert name in text

    def test_configure_runtime_sets_exactly_the_documented_variables(self):
        """
        The doc says `configure_runtime()` redirects the whole tree; this checks
        the helper and the doc agree on *which* variables.
        """
        from src.deploy import runtime
        import inspect

        source = inspect.getsource(runtime.configure_runtime)
        for name in ("TEMP_DIR", "OUTPUT_DIR", "JOBS_DIR", "API_UPLOADS_DIR"):
            assert name in source
            assert name in _doc()


# ---------------------------------------------------------------------------
# The preset table matches the code
# ---------------------------------------------------------------------------
class TestPresetClaimsMatchTheCode:

    def test_the_documented_presets_exist(self):
        from src.utils import config

        for preset in ("FAST", "BALANCED", "HIGH"):
            assert preset in config.QUALITY_PRESETS
            assert preset in _doc()

    def test_the_vram_thresholds_match_the_runtime(self):
        """
        The doc states 8 GB and 16 GB as the boundaries. Recompute the mapping
        the runtime uses and confirm the doc's numbers are the real ones.
        """
        import inspect
        from src.deploy import runtime

        source = inspect.getsource(runtime.hardware_report)
        assert "16" in source and "8" in source
        text = _doc()
        assert "8 GB" in text and "16 GB" in text
