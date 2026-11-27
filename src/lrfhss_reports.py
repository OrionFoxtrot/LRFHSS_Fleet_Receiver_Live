"""Portable external reports/evidence resolver for the LR-FHSS receiver.

The code tree is intentionally kept small. Generated benchmark results,
historical evidence, narrative reports, and analysis packages live in a sibling
``LRFHSS_Fleet_GENERATED_REPORTS`` directory. For compatibility with historical
scripts, this module can create code-tree links named ``generated`` and
``evidence`` that point into that external reports tree.

Resolution order:
  1. explicit path supplied by the caller;
  2. ``LRFHSS_REPORTS_ROOT`` environment variable;
  3. ``.lrfhss_reports_path`` saved beside this file;
  4. a recognized sibling reports directory containing the marker file.

No machine-specific path is shipped in the release package.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
MARKER = "LRFHSS_REPORTS_MANIFEST.json"
CONFIG = ROOT / ".lrfhss_reports_path"
DEFAULT_DIRECTORY_NAME = "LRFHSS_Fleet_GENERATED_REPORTS"
LINKS = {
    ROOT / "generated": Path("generated"),
    ROOT / "evidence": Path("evidence"),
}


def initialize_reports_root(root: str | os.PathLike) -> Path:
    """Create a new external reports root with the required portable layout."""
    resolved = Path(root).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    for relative in (
        "generated",
        "evidence",
        "reports",
        "analysis",
        "docs/history",
        "runtime_logs",
        "consolidated",
        "manifests",
    ):
        (resolved / relative).mkdir(parents=True, exist_ok=True)
    marker = resolved / MARKER
    if not marker.exists():
        marker.write_text(
            json.dumps(
                {
                    "schema": "LRFHSS_external_reports_v1",
                    "directory": resolved.name,
                    "purpose": "External generated evidence and historical reports for LR-FHSS receiver batches",
                    "required_directories": ["generated", "evidence"],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return resolved


def _valid(root: Path) -> bool:
    try:
        resolved = root.expanduser().resolve()
    except OSError:
        return False
    return (
        (resolved / MARKER).is_file()
        and (resolved / "generated").is_dir()
        and (resolved / "evidence").is_dir()
    )


def _candidate_roots(explicit: str | os.PathLike | None = None) -> Iterable[Path]:
    if explicit:
        yield Path(explicit)
    env = os.environ.get("LRFHSS_REPORTS_ROOT")
    if env:
        yield Path(env)
    if CONFIG.is_file():
        saved = CONFIG.read_text(encoding="utf-8").strip()
        if saved:
            yield Path(saved)

    parent = ROOT.parent
    for name in (
        DEFAULT_DIRECTORY_NAME,
        "LRFHSS_GENERATED_REPORTS",
        "Generated_Reports",
        "generated_reports",
    ):
        yield parent / name
    try:
        for child in parent.iterdir():
            if child.is_dir() and (child / MARKER).is_file():
                yield child
    except OSError:
        pass


def resolve_reports_root(explicit: str | os.PathLike | None = None) -> Path:
    """Resolve an existing reports root without modifying the filesystem."""
    candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in _candidate_roots(explicit):
        candidates.append(candidate)
        try:
            resolved = candidate.expanduser().resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if _valid(resolved):
            return resolved

    checked = "\n  - ".join(str(p) for p in candidates) or "(none)"
    raise FileNotFoundError(
        "LR-FHSS generated-reports root was not found. Extract or create "
        f"{DEFAULT_DIRECTORY_NAME} beside this code tree, set "
        "LRFHSS_REPORTS_ROOT, or run:\n\n"
        "  python3 tools/configure_reports.py --reports-root <path> --create --save\n\n"
        f"Checked:\n  - {checked}"
    )


def _same_target(link: Path, target: Path) -> bool:
    try:
        return link.resolve() == target.resolve()
    except OSError:
        return False


def _create_dir_link(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        if _same_target(link, target):
            return
        raise RuntimeError(
            f"Refusing to replace existing path {link}. Expected a link to {target}. "
            "Move any old in-tree generated/evidence directory into the external "
            "reports root, then remove the old path manually."
        )

    if os.name == "nt":
        proc = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            text=True,
            capture_output=True,
        )
        if proc.returncode:
            raise RuntimeError(
                f"Could not create Windows directory junction {link} -> {target}:\n"
                f"{proc.stdout}{proc.stderr}"
            )
    else:
        link.symlink_to(target, target_is_directory=True)


def ensure_reports_links(
    reports_root: str | os.PathLike | None = None,
    *,
    save: bool = False,
    create: bool = False,
) -> Path:
    """Resolve the reports root and create compatibility links in the code tree."""
    if create:
        if reports_root is None:
            reports_root = ROOT.parent / DEFAULT_DIRECTORY_NAME
        root = initialize_reports_root(reports_root)
    else:
        root = resolve_reports_root(reports_root)
    for link, relative_target in LINKS.items():
        target = root / relative_target
        if not target.is_dir():
            raise FileNotFoundError(f"Required reports path is missing: {target}")
        _create_dir_link(link, target)
    if save:
        CONFIG.write_text(str(root) + "\n", encoding="utf-8")
    return root


def reports_path(relative: str | os.PathLike, reports_root: str | os.PathLike | None = None) -> Path:
    """Return a path below the resolved external reports root."""
    return resolve_reports_root(reports_root) / Path(relative)


def unlink_reports() -> None:
    """Remove only compatibility links/junctions created by this module."""
    for link in LINKS:
        if not (link.exists() or link.is_symlink()):
            continue
        if os.name != "nt" and link.is_symlink():
            link.unlink()
            continue
        if os.name == "nt":
            proc = subprocess.run(["cmd", "/c", "rmdir", str(link)], capture_output=True, text=True)
            if proc.returncode:
                raise RuntimeError(f"Could not remove junction {link}: {proc.stdout}{proc.stderr}")
            continue
        raise RuntimeError(f"Refusing to remove non-link directory: {link}")
