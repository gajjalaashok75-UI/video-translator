#!/usr/bin/env python3
"""
Build the distributable codebase archive.

Produces `video-translator-full.zip`: the complete working tree — source, tests,
scripts, the deployment files, the frontend sources, and the committed sample
inputs, rendered output and pipeline evidence under `samples/`, `output/`,
`temp/` and `jobs/` — with everything that is regenerable or private left out.

Excluded, and why:

* `.git/`, `.venv/`, `node_modules/`, `__pycache__/`, `models/` — regenerable.
  The installer and `scripts/download_models.py` rebuild them, and shipping a
  multi-gigabyte venv or weight set in a zip helps nobody.
* `references/` — the Multiva.Ai and video-dubbing-system repos were design
  input only and are deliberately not part of the deliverable.
* `frontend/dist/` — a build artifact; `npm run build` recreates it.
* caches and editor/OS noise (`debug_crash.log`, `*.tsbuildinfo`, `.DS_Store`).

The exclusion set is data, not scattered `if`s, so it is testable and a reviewer
can see exactly what leaves the machine.

Usage:
    python scripts/build_archive.py                     # writes ./dist/<name>.zip
    python scripts/build_archive.py --out /tmp/archive  # choose the directory
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ARCHIVE_NAME = "video-translator-full.zip"
# The zip's single top-level directory, so it unpacks into one folder rather
# than scattering the tree into the current directory.
ARCHIVE_ROOT = "video-translator"

# Directories skipped wherever they appear, by name.
EXCLUDED_DIR_NAMES = frozenset({
    ".git", ".venv", "venv", "env",
    "node_modules", "__pycache__",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".ipynb_checkpoints",
    "models",
})

# Paths skipped by position relative to the project root. `references/` is the
# design-input repos; `frontend/dist` is a build artifact.
EXCLUDED_PATHS = frozenset({
    "references",
    "frontend/dist",
    "dist",
    ".agent_tmp",
})

# File names skipped wherever they appear.
EXCLUDED_FILE_NAMES = frozenset({
    "debug_crash.log", ".DS_Store", "Thumbs.db",
})

# Suffixes skipped wherever they appear.
EXCLUDED_SUFFIXES = (".pyc", ".pyo", ".tsbuildinfo", ".swp", ".orig")


def _is_excluded(rel: Path) -> bool:
    """
    Whether a path (relative to the project root) is left out of the archive.

    Checks the directory names anywhere in the path, the exact relative paths,
    the file's name and suffix, and the `*.egg-info` setuptools build metadata
    that an editable install leaves behind.
    """
    if any(part in EXCLUDED_DIR_NAMES or part.endswith(".egg-info")
           for part in rel.parts):
        return True
    rel_posix = rel.as_posix()
    if any(rel_posix == p or rel_posix.startswith(p + "/") for p in EXCLUDED_PATHS):
        return True
    if rel.name in EXCLUDED_FILE_NAMES:
        return True
    return rel.name.endswith(EXCLUDED_SUFFIXES)


def _collect_tracked(root: Path) -> list[Path] | None:
    """
    Every file git would keep: tracked files plus untracked files that are not
    ignored, via `git ls-files --cached --others --exclude-standard`.

    This is the precise definition of "the repository's contents", and it is what
    keeps the archive honest: the runtime trees (`jobs/`, `temp/`, `output/`) are
    git-ignored, so the transient directories a test run creates are left out
    while the evidence that was deliberately committed stays in.

    Returns `None` when `root` is not a git checkout or git is unavailable, so
    the caller can fall back to walking the tree.
    """
    import subprocess

    try:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--cached", "--others",
             "--exclude-standard", "-z"],
            capture_output=True, timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None

    files = [Path(p.decode("utf-8", "surrogateescape"))
             for p in result.stdout.split(b"\x00") if p]
    return sorted(f for f in files if not _is_excluded(f))


def collect_files(root: Path = PROJECT_ROOT) -> list[Path]:
    """
    The files to archive, as paths relative to `root`, sorted for a stable zip.

    Prefers git's view of the repository (tracked plus non-ignored) so the
    archive is exactly the project's contents and no git-ignored runtime noise
    leaks in. Falls back to walking the tree with the exclusion rules when git is
    not available — which is also the path the tests exercise on a synthetic
    tree.

    The walk prunes excluded directories in place, so a skipped directory
    (`.venv`, `node_modules`) is never descended into: walking a 2 GB venv to
    discard it would dominate the runtime.
    """
    tracked = _collect_tracked(root)
    if tracked is not None:
        return tracked

    collected: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        rel_dir = here.relative_to(root)

        # Prune in place: this is what stops the walk entering a huge directory.
        dirnames[:] = sorted(
            d for d in dirnames
            if not _is_excluded(rel_dir / d)
        )

        for name in sorted(filenames):
            rel = rel_dir / name
            if rel_dir == Path("."):
                rel = Path(name)
            if not _is_excluded(rel):
                collected.append(rel)

    return sorted(collected)


def build_archive(out_dir: Path, root: Path = PROJECT_ROOT,
                  name: str = ARCHIVE_NAME) -> tuple[Path, int]:
    """
    Write the archive and return its path and the number of files it holds.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / name
    files = collect_files(root)

    # A fixed timestamp for every entry makes the zip reproducible: two builds
    # of the same tree differ only if the tree changed.
    fixed_time = (2026, 1, 1, 0, 0, 0)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED,
                         compresslevel=6) as archive:
        for rel in files:
            source = root / rel
            info = zipfile.ZipInfo(
                filename=f"{ARCHIVE_ROOT}/{rel.as_posix()}", date_time=fixed_time)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            try:
                archive.writestr(info, source.read_bytes())
            except OSError as exc:  # a file that vanished or is unreadable
                print(f"skipping {rel}: {exc}", file=sys.stderr)

    return target, len(files)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--out", default=str(PROJECT_ROOT / "dist"),
                        help="directory to write the zip into (default: ./dist)")
    parser.add_argument("--name", default=ARCHIVE_NAME,
                        help=f"archive file name (default: {ARCHIVE_NAME})")
    args = parser.parse_args(argv)

    target, count = build_archive(Path(args.out), name=args.name)
    size_mb = target.stat().st_size / (1024 * 1024)
    print(f"wrote {target} ({count} files, {size_mb:.1f} MiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
