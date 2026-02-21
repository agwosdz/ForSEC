#!/usr/bin/env python3
"""
req2toml — Convert a requirements.txt to a uv-compatible pyproject.toml.

Handles:
  • Standard PEP 508 dependencies (versions, extras, markers)
  • --index-url / --extra-index-url  →  [[tool.uv.index]]
  • --find-links                     →  [[tool.uv.index]]
  • PyTorch index auto-detection     →  explicit = true + [tool.uv.sources]
  • -r / --requirement recursive includes
  • -e / --editable VCS installs

Usage:
    python req2toml.py <requirements.txt> [options]

Options:
    -o, --output <path>       Output path (default: pyproject.toml in cwd)
    -n, --name <name>         Project name (default: derived from directory)
    -v, --version <ver>       Project version (default: 0.1.0)
    --description <desc>      Project description
    --python <constraint>     Python version constraint (e.g. ">=3.10")
    --torch-index <variant>   Force a PyTorch index variant, e.g. cu121, cpu
    --overwrite               Overwrite existing pyproject.toml
    --dry-run                 Print generated TOML to stdout instead of writing
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Packages that should be routed to a PyTorch index via [tool.uv.sources]
_PYTORCH_PACKAGES = {
    "torch", "torchvision", "torchaudio", "torchtext",
    "pytorch-triton-rocm", "pytorch-triton-xpu",
}

# Map of known PyTorch index variants → URL path suffix
_PYTORCH_VARIANTS = {
    "cpu":      "cpu",
    "cu118":    "cu118",
    "cu121":    "cu121",
    "cu124":    "cu124",
    "cu126":    "cu126",
    "cu128":    "cu128",
    "cu130":    "cu130",
    "rocm6.4":  "rocm6.4",
    "xpu":      "xpu",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class IndexEntry:
    """Represents a [[tool.uv.index]] entry."""
    name: str
    url: str
    explicit: bool = False
    default: bool = False


@dataclass
class ParseResult:
    """Everything extracted from a requirements file."""
    deps: list[str] = field(default_factory=list)
    indexes: list[IndexEntry] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

# -r / --requirement
_REQ_FILE_RE = re.compile(r"^\s*(?:-r|--requirement)\s+(.+)$")

# -e / --editable
_EDITABLE_RE = re.compile(r"^\s*(?:-e|--editable)\s+(.+)$")

# --index-url / -i
_INDEX_URL_RE = re.compile(r"^\s*(?:-i|--index-url)\s+(.+)$")

# --extra-index-url
_EXTRA_INDEX_RE = re.compile(r"^\s*--extra-index-url\s+(.+)$")

# --find-links / -f
_FIND_LINKS_RE = re.compile(r"^\s*(?:-f|--find-links)\s+(.+)$")

# Pip options we truly cannot represent (skip silently)
_PIP_OPTION_RE = re.compile(
    r"^\s*(--trusted-host\s|--no-binary\s|--only-binary\s|--prefer-binary"
    r"|--no-deps|--pre|--require-hashes|--hash\s|--constraint\s|-c\s)"
)

# Environment markers after a semicolon
_MARKER_RE = re.compile(r";\s*(.+)$")

# Standard PEP 508 dependency
_VERSION_SPEC_RE = re.compile(
    r"^([A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?)"  # package name
    r"(\[.*?\])?"                                      # optional extras
    r"\s*(.*?)$"                                       # version constraint
)


def _clean_line(line: str) -> str:
    """Strip comments and whitespace."""
    idx = line.find("#")
    if idx >= 0:
        line = line[:idx]
    return line.strip()


def _derive_index_name(url: str) -> str:
    """Derive a short, unique name for an index URL."""
    parsed = urlparse(url.rstrip("/"))
    host = parsed.hostname or "index"
    path = parsed.path.strip("/")

    # PyTorch: https://download.pytorch.org/whl/cu121 → pytorch-cu121
    if "pytorch.org" in host and path.startswith("whl/"):
        variant = path.split("/", 1)[1] if "/" in path else path[4:]
        return f"pytorch-{variant}" if variant else "pytorch"

    # Generic: use the hostname, stripping common prefixes
    name = host.replace("www.", "").replace(".", "-")
    if path:
        # Append last meaningful path segment
        segments = [s for s in path.split("/") if s and s != "simple"]
        if segments:
            name += "-" + segments[-1]
    return name


def _is_pytorch_index(url: str) -> bool:
    """Check if a URL is a PyTorch download index."""
    return "download.pytorch.org/whl" in url


def parse_requirements(
    req_path: Path,
    *,
    _visited: set[Path] | None = None,
) -> ParseResult:
    """
    Parse a requirements.txt into dependencies, indexes, and skipped lines.

    Recursively follows ``-r`` / ``--requirement`` includes.
    """
    if _visited is None:
        _visited = set()

    req_path = req_path.resolve()
    if req_path in _visited:
        return ParseResult()
    _visited.add(req_path)

    result = ParseResult()

    with open(req_path, encoding="utf-8") as fh:
        for raw_line in fh:
            raw_stripped = raw_line.strip()

            # Editable installs must be checked BEFORE _clean_line,
            # because #egg=name contains a '#' that would be stripped.
            m = _EDITABLE_RE.match(raw_stripped)
            if m:
                url = m.group(1).strip()
                egg_match = re.search(r"#egg=([A-Za-z0-9._-]+)", url)
                if egg_match:
                    pkg = egg_match.group(1)
                    clean_url = re.sub(r"#egg=.*", "", url)
                    result.deps.append(f"{pkg} @ {clean_url}")
                else:
                    result.skipped.append(raw_stripped)
                continue

            line = _clean_line(raw_line)
            if not line:
                continue

            # Line continuations (trailing \)
            while line.endswith("\\"):
                line = line[:-1].rstrip()
                next_line = next(fh, "")
                line += " " + _clean_line(next_line)

            # -r / --requirement  →  recurse
            m = _REQ_FILE_RE.match(line)
            if m:
                ref = m.group(1).strip()
                sub_path = (req_path.parent / ref).resolve()
                sub = parse_requirements(sub_path, _visited=_visited)
                result.deps.extend(sub.deps)
                result.indexes.extend(sub.indexes)
                result.skipped.extend(sub.skipped)
                continue

            # --index-url / -i  →  default index (replaces PyPI)
            m = _INDEX_URL_RE.match(line)
            if m:
                url = m.group(1).strip()
                result.indexes.append(IndexEntry(
                    name=_derive_index_name(url),
                    url=url,
                    explicit=_is_pytorch_index(url),
                    default=True,
                ))
                continue

            # --extra-index-url  →  additional index
            m = _EXTRA_INDEX_RE.match(line)
            if m:
                url = m.group(1).strip()
                result.indexes.append(IndexEntry(
                    name=_derive_index_name(url),
                    url=url,
                    explicit=_is_pytorch_index(url),
                ))
                continue

            # --find-links / -f  →  flat index
            m = _FIND_LINKS_RE.match(line)
            if m:
                url = m.group(1).strip()
                result.indexes.append(IndexEntry(
                    name=_derive_index_name(url),
                    url=url,
                    explicit=_is_pytorch_index(url),
                ))
                continue

            # Other pip options we can't convert
            if _PIP_OPTION_RE.match(line):
                result.skipped.append(line)
                continue

            # Normal dependency line
            m = _VERSION_SPEC_RE.match(line)
            if m:
                name = m.group(1)
                extras = m.group(3) or ""
                version = m.group(4).strip()

                # Environment markers
                marker = ""
                marker_match = _MARKER_RE.search(version)
                if marker_match:
                    marker = " ; " + marker_match.group(1).strip()
                    version = version[: marker_match.start()].strip()

                dep = f"{name}{extras}{version}{marker}" if version else f"{name}{extras}{marker}"
                result.deps.append(dep)
            else:
                result.skipped.append(line)

    return result


# ---------------------------------------------------------------------------
# PyTorch helpers
# ---------------------------------------------------------------------------

def _normalise_pkg_name(name: str) -> str:
    """Normalise a package name for comparison (PEP 503)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _find_pytorch_packages(deps: list[str]) -> list[str]:
    """Return the original dependency names that are PyTorch-family packages."""
    found: list[str] = []
    for dep in deps:
        m = _VERSION_SPEC_RE.match(dep)
        if m:
            norm = _normalise_pkg_name(m.group(1))
            if norm in _PYTORCH_PACKAGES:
                found.append(m.group(1))
    return found


def _get_pytorch_index(indexes: list[IndexEntry]) -> IndexEntry | None:
    """Return the first PyTorch index from the index list."""
    for idx in indexes:
        if _is_pytorch_index(idx.url):
            return idx
    return None


def ensure_pytorch_index(
    indexes: list[IndexEntry],
    variant: str,
) -> IndexEntry:
    """
    Ensure a PyTorch index entry exists for the given variant.
    Returns the existing or newly-created IndexEntry.
    """
    url = f"https://download.pytorch.org/whl/{variant}"
    for idx in indexes:
        if idx.url.rstrip("/") == url.rstrip("/"):
            idx.explicit = True
            return idx

    entry = IndexEntry(
        name=f"pytorch-{variant}",
        url=url,
        explicit=True,
    )
    indexes.append(entry)
    return entry


# ---------------------------------------------------------------------------
# TOML generation  (no third-party lib needed)
# ---------------------------------------------------------------------------

def _toml_string(s: str) -> str:
    """Escape a string for TOML."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _toml_list(items: list[str], *, indent: str = "    ") -> str:
    """Format a TOML array of strings, one item per line."""
    if not items:
        return "[]"
    lines = ["["]
    for item in items:
        lines.append(f"{indent}{_toml_string(item)},")
    lines.append("]")
    return "\n".join(lines)


def generate_toml(
    deps: list[str],
    indexes: list[IndexEntry] | None = None,
    pytorch_sources: dict[str, str] | None = None,
    *,
    name: str = "my-project",
    version: str = "0.1.0",
    description: str = "",
    python_requires: str = ">=3.10",
) -> str:
    """Build the pyproject.toml content string."""
    lines: list[str] = []

    # -- [project] --
    lines.append("[project]")
    lines.append(f"name = {_toml_string(name)}")
    lines.append(f"version = {_toml_string(version)}")
    lines.append(f"description = {_toml_string(description)}")
    lines.append(f"requires-python = {_toml_string(python_requires)}")
    lines.append(f"dependencies = {_toml_list(deps)}")
    lines.append("")

    # -- [build-system] --
    lines.append("[build-system]")
    lines.append('requires = ["hatchling"]')
    lines.append('build-backend = "hatchling.build"')
    lines.append("")

    # -- [tool.hatch.metadata] -- allow direct references if needed
    has_direct_refs = any(re.search(r"\s*@\s+https?://", dep) for dep in deps)
    if has_direct_refs:
        lines.append("[tool.hatch.metadata]")
        lines.append("allow-direct-references = true")
        lines.append("")

    # -- [[tool.uv.index]] --
    if indexes:
        for idx in indexes:
            lines.append("[[tool.uv.index]]")
            lines.append(f"name = {_toml_string(idx.name)}")
            lines.append(f"url = {_toml_string(idx.url)}")
            if idx.explicit:
                lines.append("explicit = true")
            if idx.default:
                lines.append("default = true")
            lines.append("")

    # -- [tool.uv.sources] --
    if pytorch_sources:
        lines.append("[tool.uv.sources]")
        for pkg, index_name in pytorch_sources.items():
            lines.append(f'{pkg} = [{{ index = {_toml_string(index_name)} }}]')
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Commands file parsing
# ---------------------------------------------------------------------------

# Matches: uv pip install ... / pip install ...
_PIP_INSTALL_RE = re.compile(
    r"^(?:uv\s+)?pip\s+install\s+(.+)$", re.IGNORECASE
)


def parse_pip_install_args(arg_string: str) -> tuple[list[str], list[IndexEntry]]:
    """
    Parse the arguments of a pip install command line.
    Returns (deps, indexes) extracted from the command.
    """
    deps: list[str] = []
    indexes: list[IndexEntry] = []
    tokens = arg_string.split()
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        # --extra-index-url URL
        if tok == "--extra-index-url" and i + 1 < len(tokens):
            url = tokens[i + 1]
            indexes.append(IndexEntry(
                name=_derive_index_name(url),
                url=url,
                explicit=_is_pytorch_index(url),
            ))
            i += 2
            continue
        # --index-url / -i URL
        if tok in ("--index-url", "-i") and i + 1 < len(tokens):
            url = tokens[i + 1]
            indexes.append(IndexEntry(
                name=_derive_index_name(url),
                url=url,
                explicit=_is_pytorch_index(url),
                default=True,
            ))
            i += 2
            continue
        # --find-links / -f URL
        if tok in ("--find-links", "-f") and i + 1 < len(tokens):
            url = tokens[i + 1]
            indexes.append(IndexEntry(
                name=_derive_index_name(url),
                url=url,
            ))
            i += 2
            continue
        # Skip flags we can't represent but don't need to keep
        if tok.startswith("--") or (tok.startswith("-") and len(tok) == 2):
            # Known value-bearing flags to skip over
            if tok in ("--target", "-t", "--prefix", "--root", "--src"):
                i += 2
            else:
                i += 1  # boolean flags like --no-deps, --force-reinstall
            continue
        # Everything else is a dependency specifier
        m = _VERSION_SPEC_RE.match(tok)
        if m:
            deps.append(tok)
        i += 1
    return deps, indexes


def parse_commands_file(
    cmd_path: Path,
) -> tuple[list[str], list[IndexEntry], list[str]]:
    """
    Parse a commands file.
    Returns (deps, indexes, remaining_commands).

    - ``uv pip install`` / ``pip install`` lines are decomposed into deps + indexes.
    - All other non-empty, non-comment lines are returned as remaining commands.
    """
    deps: list[str] = []
    indexes: list[IndexEntry] = []
    remaining: list[str] = []

    for raw_line in cmd_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        m = _PIP_INSTALL_RE.match(line)
        if m:
            cmd_deps, cmd_indexes = parse_pip_install_args(m.group(1))
            deps.extend(cmd_deps)
            indexes.extend(cmd_indexes)
        else:
            remaining.append(line)

    return deps, indexes, remaining


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert requirements.txt → uv-compatible pyproject.toml",
    )
    parser.add_argument("requirements", type=Path, help="Path to requirements.txt")
    parser.add_argument("-o", "--output", type=Path, default=Path("pyproject.toml"))
    parser.add_argument("-n", "--name", default=None, help="Project name")
    parser.add_argument("-v", "--version", default="0.1.0")
    parser.add_argument("--description", default="")
    parser.add_argument("--python", default=">=3.10", dest="python_requires")
    parser.add_argument(
        "--torch-index",
        default=None,
        metavar="VARIANT",
        help=f"PyTorch index variant: {', '.join(sorted(_PYTORCH_VARIANTS))}",
    )
    parser.add_argument(
        "--commands",
        type=Path,
        default=None,
        metavar="FILE",
        help="File with extra commands to run after uv sync (generates post_install.cmd)",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)

    if not args.requirements.exists():
        print(f"Error: {args.requirements} not found.", file=sys.stderr)
        return 1

    # Validate --torch-index
    if args.torch_index and args.torch_index not in _PYTORCH_VARIANTS:
        print(
            f"Error: Unknown PyTorch variant '{args.torch_index}'. "
            f"Choose from: {', '.join(sorted(_PYTORCH_VARIANTS))}",
            file=sys.stderr,
        )
        return 1

    # Validate --commands
    if args.commands and not args.commands.exists():
        print(f"Error: Commands file {args.commands} not found.", file=sys.stderr)
        return 1

    # Default project name from cwd
    if args.name is None:
        args.name = Path.cwd().name.lower().replace(" ", "-")

    # ---- Parse requirements ----
    result = parse_requirements(args.requirements)

    # ---- Parse commands file (extract pip install deps) ----
    remaining_commands: list[str] = []
    if args.commands:
        cmd_deps, cmd_indexes, remaining_commands = parse_commands_file(args.commands)
        result.deps.extend(cmd_deps)
        # Merge indexes, avoiding duplicates by URL
        existing_urls = {idx.url.rstrip("/") for idx in result.indexes}
        for idx in cmd_indexes:
            if idx.url.rstrip("/") not in existing_urls:
                result.indexes.append(idx)
                existing_urls.add(idx.url.rstrip("/"))

    # ---- PyTorch index handling ----
    pytorch_sources: dict[str, str] = {}

    # If --torch-index was given, ensure the index exists
    if args.torch_index:
        variant = _PYTORCH_VARIANTS[args.torch_index]
        pt_index = ensure_pytorch_index(result.indexes, variant)
    else:
        # Check if a PyTorch index was already captured from the requirements
        pt_index = _get_pytorch_index(result.indexes)
        if pt_index:
            pt_index.explicit = True

    # Build [tool.uv.sources] for PyTorch-family packages
    if pt_index:
        pt_pkgs = _find_pytorch_packages(result.deps)
        for pkg in pt_pkgs:
            pytorch_sources[pkg] = pt_index.name

    # ---- Generate ----
    toml = generate_toml(
        result.deps,
        indexes=result.indexes or None,
        pytorch_sources=pytorch_sources or None,
        name=args.name,
        version=args.version,
        description=args.description,
        python_requires=args.python_requires,
    )

    if args.dry_run:
        print(toml)
    else:
        if args.output.exists() and not args.overwrite:
            print(
                f"Error: {args.output} already exists. Use --overwrite to replace it.",
                file=sys.stderr,
            )
            return 1
        args.output.write_text(toml, encoding="utf-8")
        print(f"Wrote {args.output}")

        # Generate post_install.cmd for remaining (non-pip) commands
        if remaining_commands:
            post_install = args.output.parent / "post_install.cmd"
            lines_out = ["@echo off", "REM Auto-generated by req2toml from " + args.commands.name, ""]
            for cmd_line in remaining_commands:
                # Rewrite "python ..." → "uv run python ..." to use managed env
                if re.match(r"^python\s", cmd_line, re.IGNORECASE):
                    cmd_line = "uv run " + cmd_line
                lines_out.append(f'echo ^> {cmd_line}')
                lines_out.append(cmd_line)
                lines_out.append('if %ERRORLEVEL% neq 0 echo [WARNING] Command exited with code %ERRORLEVEL%')
                lines_out.append('')
            post_install.write_text("\n".join(lines_out), encoding="utf-8")
            print(f"Wrote {post_install}")

    if result.skipped:
        print(
            f"\n⚠  {len(result.skipped)} line(s) were skipped (pip flags / unsupported):",
            file=sys.stderr,
        )
        for s in result.skipped:
            print(f"   • {s}", file=sys.stderr)

    print(f"\n✓ {len(result.deps)} dependencies converted.", end="")
    if result.indexes:
        print(f"  {len(result.indexes)} index(es) configured.", end="")
    if pytorch_sources:
        print(f"  {len(pytorch_sources)} PyTorch source(s) routed.", end="")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
