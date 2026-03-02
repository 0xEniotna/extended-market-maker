#!/usr/bin/env python3
"""Format a .env file to match the structure/ordering of a reference .env.

Usage:
    python scripts/tools/format_env.py .env.eth .env.lit
    python scripts/tools/format_env.py .env.eth .env.*        # format all
    python scripts/tools/format_env.py .env.eth .env.lit --dry-run
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def _parse_key(line: str) -> str | None:
    """Extract the key from an env line, or None if it's a comment/blank."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if "=" not in stripped:
        return None
    key = stripped.split("=", 1)[0].strip()
    if key.startswith("export "):
        key = key[7:].strip()
    return key


def _parse_value(line: str) -> str:
    """Extract the raw value (everything after first =)."""
    if "=" not in line:
        return ""
    return line.split("=", 1)[1].rstrip("\n")


def _build_reference_skeleton(ref_lines: list[str]) -> list[str | tuple[str, str]]:
    """Build a skeleton from the reference file.

    Returns a list of:
    - str: comment/blank lines (kept as-is)
    - tuple[str, str]: (key, inline_comment_from_ref) for key lines
    """
    skeleton: list[str | tuple[str, str]] = []
    for line in ref_lines:
        key = _parse_key(line)
        if key is None:
            skeleton.append(line.rstrip("\n"))
        else:
            # Extract any inline comment from reference (for section context)
            skeleton.append((key, ""))
    return skeleton


def _parse_target(target_lines: list[str]) -> dict[str, str]:
    """Parse target .env into key -> full value (including inline comments)."""
    kv: dict[str, str] = {}
    for line in target_lines:
        key = _parse_key(line)
        if key is not None:
            kv[key] = _parse_value(line)
    return kv


def format_env(ref_path: Path, target_path: Path) -> str:
    """Format target .env to match reference structure. Returns formatted content."""
    ref_lines = ref_path.read_text().splitlines(keepends=True)
    target_lines = target_path.read_text().splitlines(keepends=True)

    skeleton = _build_reference_skeleton(ref_lines)
    target_kv = _parse_target(target_lines)
    ref_keys = {k for item in skeleton if isinstance(item, tuple) for k in [item[0]]}

    out_lines: list[str] = []
    emitted_keys: set[str] = set()

    for item in skeleton:
        if isinstance(item, str):
            # Comment or blank line from reference
            out_lines.append(item)
        else:
            key, _ = item
            if key in target_kv:
                out_lines.append(f"{key}={target_kv[key]}")
                emitted_keys.add(key)
            else:
                # Key exists in reference but not in target — skip it
                pass

    # Append extra keys that exist in target but not in reference
    extra_keys = [k for k in target_kv if k not in ref_keys]
    if extra_keys:
        out_lines.append("")
        out_lines.append("# ── Extra keys (not in reference) ──────────────────────────────")
        for key in extra_keys:
            out_lines.append(f"{key}={target_kv[key]}")

    # Clean up: collapse 3+ consecutive blank lines into 2
    result = "\n".join(out_lines).rstrip() + "\n"
    result = re.sub(r"\n{4,}", "\n\n\n", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Format .env files to match a reference structure.")
    parser.add_argument("reference", help="Reference .env file (e.g. .env.eth)")
    parser.add_argument("targets", nargs="+", help="Target .env file(s) to format")
    parser.add_argument("--dry-run", action="store_true", help="Print diff without writing")
    args = parser.parse_args()

    ref_path = Path(args.reference)
    if not ref_path.exists():
        print(f"error: reference file not found: {ref_path}", file=sys.stderr)
        return 1

    targets = []
    for t in args.targets:
        p = Path(t)
        if p == ref_path:
            continue
        if not p.exists():
            print(f"warning: skipping {t} (not found)", file=sys.stderr)
            continue
        targets.append(p)

    if not targets:
        print("No target files to format.", file=sys.stderr)
        return 0

    for target_path in sorted(targets):
        original = target_path.read_text()
        formatted = format_env(ref_path, target_path)

        if original == formatted:
            print(f"  {target_path.name}: already formatted")
            continue

        if args.dry_run:
            print(f"  {target_path.name}: would reformat ({_diff_summary(original, formatted)})")
        else:
            target_path.write_text(formatted)
            print(f"  {target_path.name}: reformatted ({_diff_summary(original, formatted)})")

    return 0


def _diff_summary(before: str, after: str) -> str:
    """Quick summary of what changed."""
    before_keys = set(_parse_target(before.splitlines(keepends=True)).keys())
    after_keys = set(_parse_target(after.splitlines(keepends=True)).keys())
    added = after_keys - before_keys
    removed = before_keys - after_keys
    parts = [f"{len(after_keys)} keys"]
    if added:
        parts.append(f"+{len(added)} new")
    if removed:
        parts.append(f"-{len(removed)} removed")
    return ", ".join(parts)


if __name__ == "__main__":
    raise SystemExit(main())
