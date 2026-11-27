"""External test-data resolver for the LR-FHSS receiver.

Batch 7.5 deliberately separates changing receiver source from large, reusable
IQ fixtures and validation evidence.  The code package contains no test data.
For compatibility with existing tests/tools, this module creates directory
links from the historic in-tree paths to one external data bundle.

Resolution order:
  1. explicit path supplied by caller;
  2. LRFHSS_TEST_DATA_ROOT environment variable;
  3. .lrfhss_test_data_path saved beside this file;
  4. a sibling directory containing TEST_DATA_MANIFEST.json.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MARKER = "TEST_DATA_MANIFEST.json"
CONFIG = ROOT / ".lrfhss_test_data_path"
LINKS = {
    ROOT / "tests" / "fixtures": Path("tests/fixtures"),
    ROOT / "tests" / "reference": Path("tests/reference"),
    ROOT / "validation": Path("validation"),
}


def _valid(root: Path) -> bool:
    root = root.expanduser().resolve()
    return (root / MARKER).is_file() and all((root / rel).exists() for rel in LINKS.values())


def resolve_test_data_root(explicit: str | os.PathLike | None = None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.environ.get("LRFHSS_TEST_DATA_ROOT")
    if env:
        candidates.append(Path(env))
    if CONFIG.is_file():
        saved = CONFIG.read_text(encoding="utf-8").strip()
        if saved:
            candidates.append(Path(saved))

    parent = ROOT.parent
    for name in ("LRFHSS_Fleet_TEST_DATA_v1", "LRFHSS_TEST_DATA_v1", "test_data"):
        candidates.append(parent / name)
    try:
        for child in parent.iterdir():
            if child.is_dir() and (child / MARKER).is_file():
                candidates.append(child)
    except OSError:
        pass

    seen: set[Path] = set()
    for candidate in candidates:
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
        "LR-FHSS external test data was not found. Extract the reusable "
        "LRFHSS_Fleet_TEST_DATA_v1 package beside this code tree, set "
        "LRFHSS_TEST_DATA_ROOT, or run:\n\n"
        "  python tools/configure_test_data.py --data-root <path> --save\n\n"
        f"Checked:\n  - {checked}"
    )


def _same_target(link: Path, target: Path) -> bool:
    try:
        return link.resolve() == target.resolve()
    except OSError:
        return False


def _is_link(path: Path) -> bool:
    """A symlink or (Windows) directory junction, whether or not its target exists."""
    is_junction = getattr(path, "is_junction", None)
    return path.is_symlink() or bool(is_junction and is_junction())


def _remove_dangling_link(link: Path) -> bool:
    """Remove a link/junction whose target no longer exists (e.g. after the code,
    data and reports folders were moved together). A dangling link holds no data;
    a real directory or a link to an existing target is never touched."""
    if link.exists() or not _is_link(link):
        return False
    if os.name == "nt":
        os.rmdir(link)            # removes the junction/symlink entry only
    else:
        link.unlink()
    return True


def _create_dir_link(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    _remove_dangling_link(link)
    if link.exists() or link.is_symlink():
        if _same_target(link, target):
            return
        raise RuntimeError(
            f"Refusing to replace existing path {link}. Expected a link to {target}. "
            "Move/remove the existing path manually if it is obsolete."
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


def ensure_test_data_links(
    data_root: str | os.PathLike | None = None,
    *,
    save: bool = False,
) -> Path:
    root = resolve_test_data_root(data_root)
    for link, relative_target in LINKS.items():
        target = root / relative_target
        if not target.exists():
            raise FileNotFoundError(f"Required test-data path is missing: {target}")
        _create_dir_link(link, target)
    if save:
        CONFIG.write_text(str(root) + "\n", encoding="utf-8")
    return root


def unlink_test_data() -> None:
    """Remove only links/junctions created by this compatibility layer."""
    for link in LINKS:
        if not (link.exists() or link.is_symlink()):
            continue
        # Never remove a real directory containing data.
        if os.name != "nt" and link.is_symlink():
            link.unlink()
            continue
        if os.name == "nt":
            # A junction is a reparse point and rmdir removes the junction only.
            proc = subprocess.run(["cmd", "/c", "rmdir", str(link)], capture_output=True, text=True)
            if proc.returncode:
                raise RuntimeError(f"Could not remove junction {link}: {proc.stdout}{proc.stderr}")
            continue
        raise RuntimeError(f"Refusing to remove non-link directory: {link}")
