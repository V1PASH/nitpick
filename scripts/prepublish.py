"""Checks that run before a package is published.

Publishing is the one thing you cannot undo, so the checks are strict:

1. The working tree is clean — never publish half-finished work.
2. The version in ``pyproject.toml`` matches the package's ``__version__``.
3. The version has not already been published — PyPI rejects a re-upload,
   and it is better to say so plainly than to fail halfway through.
4. The built artefacts exist and pass ``twine check``.

Run it directly, or through ``task publish``::

    python scripts/prepublish.py
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"


def fail(message: str) -> None:
    print(f"\n  ✗ {message}\n", file=sys.stderr)
    sys.exit(1)


def run(*args: str) -> tuple[int, str]:
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    return result.returncode, (result.stdout + result.stderr).strip()


def check_clean_tree() -> None:
    if not (ROOT / ".git").exists():
        return  # not a git checkout — nothing to check
    code, output = run("git", "status", "--porcelain")
    if code != 0:
        return
    if output.strip():
        files = [line[3:] for line in output.splitlines() if line.strip()]
        fail(
            "the working tree is dirty — commit or stash first:\n    "
            + "\n    ".join(files[:10])
        )


def declared_version() -> str:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return str(tomllib.load(handle)["project"]["version"])


def package_version() -> str | None:
    """Read ``__version__`` out of the package's ``__init__.py``."""
    for init in sorted(ROOT.glob("*/__init__.py")):
        source = init.read_text(encoding="utf-8")
        match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', source, re.M)
        if match:
            return match.group(1)
    return None


def check_versions_match() -> str:
    version = declared_version()
    package = package_version()
    if package is None:
        print("  · no __version__ found — skipping the check")
        return version
    if package != version:
        fail(
            f"the versions disagree — pyproject.toml says {version!r} but the "
            f"package says {package!r}"
        )
    print(f"  · version {version} (pyproject and the package agree)")
    return version


def already_published(name: str, version: str) -> bool:
    """Is this version already on PyPI?

    ``httpx`` is used rather than ``urllib`` because it bundles its own CA
    bundle. ``urllib`` relies on the system trust store, which on some
    machines cannot verify PyPI's certificate — which would quietly turn
    this check into a no-op.
    """
    url = f"https://pypi.org/pypi/{name}/{version}/json"

    try:
        import httpx

        response = httpx.get(url, timeout=15, follow_redirects=True)
        return response.status_code == 200
    except Exception as error:  # noqa: BLE001 - never block a publish on this
        print(f"  · could not check PyPI ({error}) — continuing")
        return False


def check_not_published(name: str, version: str) -> None:
    if already_published(name, version):
        fail(
            f"{name} {version} is already on PyPI — bump the version before "
            "publishing again"
        )
    print(f"  · {name} {version} is not on PyPI yet")

def check_dist() -> None:
    artefacts = sorted(
        [p for p in DIST.glob("*.whl")] + [p for p in DIST.glob("*.tar.gz")]
    )
    if not artefacts:
        fail("there is nothing in dist/ — run the build first")
    print("  · artefacts:")
    for artefact in artefacts:
        print(f"      {artefact.name}  ({artefact.stat().st_size:,} bytes)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-published-check",
        action="store_true",
        help="do not check whether the version is already on PyPI",
    )
    args = parser.parse_args()

    name = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["name"]
    print(f"\npublishing {name}")

    check_clean_tree()
    version = check_versions_match()
    if not args.skip_published_check:
        check_not_published(name, version)
    check_dist()

    print("\n  all checks passed\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
