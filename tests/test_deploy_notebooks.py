"""
A9: the Colab and Kaggle notebooks, and the script that builds them.

A notebook is a JSON file that is easy to break and hard to review, so these
tests treat the shipped `.ipynb` files as build products: every code cell must
be valid Python, the two notebooks must agree on everything except the
platform-specific opening, and `scripts/build_notebooks.py --check` must confirm
the files on disk match the script (so a notebook edited in the browser without
updating the source fails here).

The last class actually runs the notebook's own logic against the real runtime
module, so a cell that calls a function that no longer exists is caught without
executing a GPU notebook.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_DIR = PROJECT_ROOT / "deploy" / "notebooks"
NOTEBOOKS = {"colab": NOTEBOOK_DIR / "colab.ipynb",
             "kaggle": NOTEBOOK_DIR / "kaggle.ipynb"}


def _load(name: str) -> dict:
    return json.loads(NOTEBOOKS[name].read_text(encoding="utf-8"))


def _sources(nb: dict, cell_type: str | None = None) -> list[str]:
    return ["".join(c["source"]) for c in nb["cells"]
            if cell_type is None or c["cell_type"] == cell_type]


@pytest.fixture(params=sorted(NOTEBOOKS))
def notebook(request) -> dict:
    return _load(request.param)


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------
class TestNotebookShape:

    def test_the_file_is_a_valid_v4_notebook(self, notebook):
        assert notebook["nbformat"] == 4
        assert notebook["nbformat_minor"] >= 5
        assert notebook["cells"]

    def test_every_cell_has_an_id(self, notebook):
        # nbformat 4.5 requires it, and a browser rewrites a missing one.
        assert all(c.get("id") for c in notebook["cells"])

    def test_every_code_cell_is_valid_python(self, notebook):
        for index, source in enumerate(_sources(notebook, "code")):
            try:
                ast.parse(source)
            except SyntaxError as exc:  # pragma: no cover - failure path
                pytest.fail(f"code cell {index} is not valid Python: {exc}")

    def test_code_cells_have_no_stored_output(self, notebook):
        # A committed output would be a stale traceback waiting to confuse a
        # reader, and it bloats the file.
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                assert cell.get("outputs") == []
                assert cell.get("execution_count") is None


# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------
class TestNotebookContent:

    def test_it_starts_the_server_on_a_reachable_interface(self, notebook):
        code = "\n".join(_sources(notebook, "code"))
        assert 'start_server(host="0.0.0.0"' in code

    def test_it_configures_the_runtime_before_importing_the_app(self, notebook):
        """
        The ordering rule: `configure_runtime()` must run before anything that
        imports `src.utils.config`, or the redirect is too late.
        """
        code_cells = _sources(notebook, "code")
        configure_at = next(
            (i for i, s in enumerate(code_cells) if "configure_runtime(" in s),
            None)
        assert configure_at is not None, "the notebook never configures the runtime"

        # No earlier cell may import the config module (directly or via the app).
        earlier = "\n".join(code_cells[:configure_at])
        for forbidden in ("src.utils.config", "src.api.main", "import torch"):
            assert forbidden not in earlier, (
                f"{forbidden!r} is imported before configure_runtime()")

    def test_it_stops_a_previous_server_before_starting_a_new_one(self, notebook):
        start = next(s for s in _sources(notebook, "code")
                     if "start_server(" in s)
        assert "_previous" in start and ".stop()" in start

    def test_it_stops_the_server_at_the_end(self, notebook):
        code = "\n".join(_sources(notebook, "code"))
        assert "server.stop()" in code

    def test_it_tells_the_user_where_the_url_came_from(self, notebook):
        code = "\n".join(_sources(notebook, "code"))
        assert "notebook_public_url(" in code

    def test_it_installs_the_package_with_the_api_and_gpu_extras(self, notebook):
        code = "\n".join(_sources(notebook, "code"))
        assert "[gpu-cu124" in code and "api" in code
        assert "pip" in code and "install" in code

    def test_the_install_targets_the_editable_repo(self, notebook):
        install = next(s for s in _sources(notebook, "code")
                       if "pip" in s and "install" in s)
        assert "-e" in install and "REPO_DIR" in install


# ---------------------------------------------------------------------------
# The two notebooks are the same notebook
# ---------------------------------------------------------------------------
class TestColabAndKaggleAgree:

    def test_they_have_the_same_cell_count_and_types(self):
        colab, kaggle = _load("colab"), _load("kaggle")
        assert len(colab["cells"]) == len(kaggle["cells"])
        assert ([c["cell_type"] for c in colab["cells"]] ==
                [c["cell_type"] for c in kaggle["cells"]])

    def test_the_only_differences_are_platform_specific(self):
        """
        Every difference between the two notebooks must be explained by the
        platform: its clone directory, its heading, or its proxy note. Anything
        else means the two have drifted.
        """
        colab = _sources(_load("colab"))
        kaggle = _sources(_load("kaggle"))
        differing = [i for i, (a, b) in enumerate(zip(colab, kaggle)) if a != b]

        markers = ("/content", "/kaggle", "colab", "kaggle", "proxied", "tunnel")
        for index in differing:
            joined = (colab[index] + kaggle[index]).lower()
            assert any(m in joined for m in markers), (
                f"cell {index} differs for no platform reason")

        # The shared bulk of the notebook (install, start, use, stop) must match,
        # so the count stays small.
        assert len(differing) <= 6, f"too many differing cells: {differing}"

    def test_neither_notebook_hardcodes_a_localhost_url_as_the_result(self):
        for name in NOTEBOOKS:
            code = "\n".join(_sources(_load(name), "code"))
            # The URL must be produced by the runtime helper, not written by hand.
            assert 'url = runtime.notebook_public_url(' in code


# ---------------------------------------------------------------------------
# The builder and the files on disk agree
# ---------------------------------------------------------------------------
class TestBuilderIsTheSourceOfTruth:

    def test_check_mode_passes(self):
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "build_notebooks.py"),
             "--check"],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
            env={**__import__("os").environ, "PYTHONPATH": str(PROJECT_ROOT)},
        )
        assert result.returncode == 0, (
            "the notebooks on disk do not match scripts/build_notebooks.py; "
            f"run it to regenerate them.\n{result.stdout}\n{result.stderr}")

    def test_the_builder_renders_both_notebooks(self):
        sys.path.insert(0, str(PROJECT_ROOT))
        try:
            import scripts.build_notebooks as builder  # type: ignore
        finally:
            sys.path.pop(0)

        for name in ("colab", "kaggle"):
            rendered = builder.render(name)
            assert rendered.endswith("\n")
            parsed = json.loads(rendered)
            assert parsed["cells"]


# ---------------------------------------------------------------------------
# The cells call functions that exist
# ---------------------------------------------------------------------------
class TestNotebookCallsRealApis:

    def test_every_runtime_attribute_used_exists(self):
        """
        Scan the notebooks for `runtime.<name>` references and assert each is a
        real attribute of the deployment module. This catches a renamed helper
        without running the notebook.
        """
        from src.deploy import runtime

        names = set()
        for name in NOTEBOOKS:
            code = "\n".join(_sources(_load(name), "code"))
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if (isinstance(node, ast.Attribute)
                        and isinstance(node.value, ast.Name)
                        and node.value.id == "runtime"):
                    names.add(node.attr)

        assert names, "the notebooks never call into the runtime module"
        missing = sorted(n for n in names if not hasattr(runtime, n))
        assert not missing, f"the notebooks call missing runtime helpers: {missing}"

    def test_the_configure_and_url_helpers_behave_as_the_notebooks_assume(
            self, tmp_path, monkeypatch):
        """A light integration check of the two helpers the cells rely on."""
        from src.deploy import runtime

        for var in ("TEMP_DIR", "OUTPUT_DIR", "JOBS_DIR", "API_UPLOADS_DIR"):
            monkeypatch.delenv(var, raising=False)

        dirs = runtime.configure_runtime(tmp_path / "rt")
        assert all(p.is_dir() for p in dirs.values())
        assert runtime.notebook_public_url(8000).startswith("http")
