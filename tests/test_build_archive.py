"""
A9: the distributable archive builder.

`scripts/build_archive.py` decides what leaves the machine, so the exclusions
are the part worth testing: a leaked `references/` directory would ship design
input that was meant to stay out, and a leaked `.venv/` would produce a
multi-gigabyte zip.

The tests build real zips from a synthetic tree, so they check the actual
written archive — its contents, its single top-level directory, and its
reproducibility — rather than the filter in isolation.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import build_archive  # noqa: E402


def _make_tree(root: Path) -> None:
    """A miniature project with one of every include/exclude category."""
    (root / "src").mkdir(parents=True)
    (root / "src" / "main.py").write_text("print('hi')\n")
    (root / "tests").mkdir()
    (root / "tests" / "test_x.py").write_text("def test_x():\n    pass\n")
    (root / "deploy" / "notebooks").mkdir(parents=True)
    (root / "deploy" / "notebooks" / "colab.ipynb").write_text("{}\n")
    (root / "docs").mkdir()
    (root / "docs" / "DEPLOYMENT.md").write_text("# deploy\n")

    # Excluded: regenerable or private.
    (root / "references").mkdir()
    (root / "references" / "secret.txt").write_text("design input\n")
    (root / "models").mkdir()
    (root / "models" / "big.pth").write_bytes(b"0" * 1024)
    (root / ".venv" / "lib").mkdir(parents=True)
    (root / ".venv" / "lib" / "thing.py").write_text("x = 1\n")
    (root / "node_modules" / "pkg").mkdir(parents=True)
    (root / "node_modules" / "pkg" / "index.js").write_text("//\n")
    (root / "frontend" / "dist").mkdir(parents=True)
    (root / "frontend" / "dist" / "bundle.js").write_text("//\n")
    (root / "frontend" / "src").mkdir(parents=True)
    (root / "frontend" / "src" / "App.tsx").write_text("export {}\n")
    (root / "src" / "__pycache__").mkdir()
    (root / "src" / "__pycache__" / "main.cpython-312.pyc").write_bytes(b"\x00")
    (root / "video_translator.egg-info").mkdir()
    (root / "video_translator.egg-info" / "PKG-INFO").write_text("Name: x\n")

    # Excluded: noise files.
    (root / "debug_crash.log").write_text("traceback\n")
    (root / ".DS_Store").write_bytes(b"\x00")
    (root / "frontend" / "tsconfig.app.tsbuildinfo").write_text("{}\n")


@pytest.fixture
def tree(tmp_path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    _make_tree(root)
    return root


# ---------------------------------------------------------------------------
# What is left out
# ---------------------------------------------------------------------------
class TestExclusions:

    @pytest.mark.parametrize("path", [
        "references/secret.txt",
        "models/big.pth",
        ".venv/lib/thing.py",
        "node_modules/pkg/index.js",
        "frontend/dist/bundle.js",
        "src/__pycache__/main.cpython-312.pyc",
        "video_translator.egg-info/PKG-INFO",
        "debug_crash.log",
        ".DS_Store",
        "frontend/tsconfig.app.tsbuildinfo",
    ])
    def test_the_path_is_excluded(self, path):
        assert build_archive._is_excluded(Path(path))

    @pytest.mark.parametrize("path", [
        "src/main.py",
        "tests/test_x.py",
        "deploy/notebooks/colab.ipynb",
        "docs/DEPLOYMENT.md",
        "frontend/src/App.tsx",
        "README.md",
    ])
    def test_the_path_is_kept(self, path):
        assert not build_archive._is_excluded(Path(path))

    def test_exclusions_are_declared_not_scattered(self):
        # The sets are the reviewable record of what leaves the machine.
        assert "references" in build_archive.EXCLUDED_PATHS
        assert "frontend/dist" in build_archive.EXCLUDED_PATHS
        for name in ("models", "node_modules", ".venv", "__pycache__", ".git"):
            assert name in build_archive.EXCLUDED_DIR_NAMES


# ---------------------------------------------------------------------------
# What is collected
# ---------------------------------------------------------------------------
class TestCollection:

    def test_only_the_kept_files_are_collected(self, tree):
        files = {p.as_posix() for p in build_archive.collect_files(tree)}
        assert files == {
            "src/main.py",
            "tests/test_x.py",
            "deploy/notebooks/colab.ipynb",
            "docs/DEPLOYMENT.md",
            "frontend/src/App.tsx",
        }

    def test_collection_is_sorted(self, tree):
        files = build_archive.collect_files(tree)
        assert files == sorted(files)

    def test_it_does_not_descend_into_an_excluded_directory(self, tree, monkeypatch):
        """
        The walk must prune, not filter: descending a real `.venv` (gigabytes)
        just to discard it would dominate the build.
        """
        visited = []
        real_walk = build_archive.os.walk

        def _spy_walk(root):
            for dirpath, dirnames, filenames in real_walk(root):
                visited.append(Path(dirpath).name)
                yield dirpath, dirnames, filenames

        monkeypatch.setattr(build_archive.os, "walk", _spy_walk)
        build_archive.collect_files(tree)

        for excluded in (".venv", "node_modules", "models", "references"):
            assert excluded not in visited, f"walked into {excluded}"


# ---------------------------------------------------------------------------
# The archive itself
# ---------------------------------------------------------------------------
class TestBuildArchive:

    def test_it_writes_a_zip_with_one_top_level_directory(self, tree, tmp_path):
        target, count = build_archive.build_archive(
            tmp_path / "out", root=tree, name="x.zip")
        assert target.is_file()
        assert count == 5

        with zipfile.ZipFile(target) as archive:
            names = archive.namelist()
            assert all(n.startswith("video-translator/") for n in names)
            assert "video-translator/src/main.py" in names
            assert "video-translator/docs/DEPLOYMENT.md" in names

    def test_no_excluded_path_reaches_the_archive(self, tree, tmp_path):
        target, _ = build_archive.build_archive(
            tmp_path / "out", root=tree, name="x.zip")
        with zipfile.ZipFile(target) as archive:
            names = archive.namelist()
        for probe in ("references/", "models/", ".venv/", "node_modules/",
                      "frontend/dist/", "__pycache__/", "egg-info/"):
            assert not any(probe in n for n in names), f"{probe} leaked"

    def test_the_archived_bytes_match_the_source(self, tree, tmp_path):
        target, _ = build_archive.build_archive(
            tmp_path / "out", root=tree, name="x.zip")
        with zipfile.ZipFile(target) as archive:
            assert archive.read("video-translator/src/main.py") == b"print('hi')\n"

    def test_it_is_reproducible(self, tree, tmp_path):
        """Two builds of the same tree must be byte-identical."""
        first, _ = build_archive.build_archive(tmp_path / "a", root=tree, name="x.zip")
        second, _ = build_archive.build_archive(tmp_path / "b", root=tree, name="x.zip")
        assert first.read_bytes() == second.read_bytes()

    def test_it_creates_the_output_directory(self, tree, tmp_path):
        target, _ = build_archive.build_archive(
            tmp_path / "does" / "not" / "exist", root=tree, name="x.zip")
        assert target.is_file()


# ---------------------------------------------------------------------------
# The real project
# ---------------------------------------------------------------------------
class TestTheRealProject:

    def test_the_reference_repos_are_never_archived(self):
        """
        `references/` is the one exclusion with a promise attached: those repos
        were design input and are not part of the deliverable.
        """
        files = build_archive.collect_files(PROJECT_ROOT)
        assert not any(f.parts[0] == "references" for f in files)

    def test_the_expected_top_level_entries_are_present(self):
        files = build_archive.collect_files(PROJECT_ROOT)
        tops = {f.parts[0] for f in files}
        for expected in ("src", "tests", "scripts", "deploy", "docs",
                         "frontend", "samples", "run.py", "README.md"):
            assert expected in tops, f"{expected} is missing from the archive"

    def test_the_a9_artifacts_are_in_the_archive(self):
        files = {f.as_posix() for f in build_archive.collect_files(PROJECT_ROOT)}
        assert "deploy/notebooks/colab.ipynb" in files
        assert "deploy/notebooks/kaggle.ipynb" in files
        assert "docs/DEPLOYMENT.md" in files
        assert "scripts/build_notebooks.py" in files
        assert "src/deploy/runtime.py" in files

    def test_git_ignored_runtime_noise_is_left_out(self):
        """
        A test run writes transient `jobs/` directories that git ignores. The
        archive must follow git's view, or every build after a test run would
        differ from the repository's contents.
        """
        import subprocess

        files = build_archive.collect_files(PROJECT_ROOT)
        archived_jobs = {f.parts[1] for f in files
                         if len(f.parts) > 1 and f.parts[0] == "jobs"}
        tracked_jobs = {
            line.split("/")[1]
            for line in subprocess.run(
                ["git", "-C", str(PROJECT_ROOT), "ls-files", "jobs"],
                capture_output=True, text=True).stdout.splitlines()
            if "/" in line
        }
        # Every archived job directory is a tracked one.
        assert archived_jobs <= tracked_jobs

    def test_it_uses_gits_view_when_available(self):
        """The collection is git-aware in a real checkout."""
        assert build_archive._collect_tracked(PROJECT_ROOT) is not None

    def test_it_falls_back_to_walking_without_git(self, tree):
        """A non-git tree still archives, using the exclusion rules."""
        assert build_archive._collect_tracked(tree) is None
        files = {f.as_posix() for f in build_archive.collect_files(tree)}
        assert "src/main.py" in files
        assert not any(f.startswith("references/") for f in files)
