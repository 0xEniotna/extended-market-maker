#!/usr/bin/env python3
"""
Generate policy files (bounds.json, whitelist.json, protected_keys.json)
from the MarketMakerSettings pydantic model.

Ensures policy files never drift from config.py — field additions,
renames, or constraint changes are automatically reflected.

Usage:
    python scripts/generate_policy.py              # preview to stdout
    python scripts/generate_policy.py --write      # overwrite policy files
    python scripts/generate_policy.py --check      # exit non-zero if drift
"""
from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal
from pathlib import Path

# Ensure the src package is importable.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from market_maker.config import MarketMakerSettings, SETTINGS_FIELD_GROUPS  # noqa: E402

POLICY_DIR = _REPO_ROOT / "mm_config" / "policy"

# Fields that should never be modified by advisor agents.
PROTECTED_KEYS = frozenset({
    "MM_API_KEY",
    "MM_STARK_PRIVATE_KEY",
    "MM_STARK_PUBLIC_KEY",
    "MM_VAULT_ID",
    "MM_BUILDER_ID",
})

# Env vars in bounds.json that are external (not part of MarketMakerSettings).
EXTERNAL_KEYS = {"EXTENDED_ENV"}


def _json_default(obj):
    if isinstance(obj, Decimal):
        f = float(obj)
        if f == int(f):
            return int(f)
        return f
    return str(obj)


def _numeric_bounds(field_meta: dict) -> dict:
    """Extract numeric min/max bounds from field metadata."""
    bounds: dict = {}
    # Map pydantic constraint attrs → policy min/max.
    if "ge" in field_meta:
        bounds["min"] = float(field_meta["ge"]) if isinstance(field_meta["ge"], (int, float, Decimal)) else field_meta["ge"]
    if "gt" in field_meta:
        bounds["min"] = float(field_meta["gt"]) if isinstance(field_meta["gt"], (int, float, Decimal)) else field_meta["gt"]
    if "le" in field_meta:
        bounds["max"] = float(field_meta["le"]) if isinstance(field_meta["le"], (int, float, Decimal)) else field_meta["le"]
    if "lt" in field_meta:
        bounds["max"] = float(field_meta["lt"]) if isinstance(field_meta["lt"], (int, float, Decimal)) else field_meta["lt"]
    return bounds


def generate_bounds() -> dict:
    """Generate bounds.json from MarketMakerSettings field metadata."""
    metadata = MarketMakerSettings.field_metadata()
    bounds: dict = {}

    for entry in metadata:
        env_var = entry["env_var"]
        type_name = entry["type"]

        spec: dict = {"type": type_name}
        numeric = _numeric_bounds(entry)
        if numeric:
            # Convert Decimal bounds to float for JSON.
            for k, v in numeric.items():
                if isinstance(v, Decimal):
                    spec[k] = float(v)
                else:
                    spec[k] = v
        bounds[env_var] = spec

    # Add external keys that aren't part of the model.
    for key in sorted(EXTERNAL_KEYS):
        if key not in bounds:
            bounds[key] = {"type": "string"}

    return dict(sorted(bounds.items()))


def generate_whitelist() -> list:
    """Generate whitelist.json — fields that advisor agents may modify."""
    metadata = MarketMakerSettings.field_metadata()
    whitelist = []
    for entry in metadata:
        env_var = entry["env_var"]
        # Exclude credentials and protected keys.
        if env_var in PROTECTED_KEYS:
            continue
        if entry["group"] == "credentials":
            continue
        whitelist.append(env_var)
    return sorted(whitelist)


def generate_protected_keys() -> list:
    """Generate protected_keys.json — fields that must never be changed."""
    return sorted(PROTECTED_KEYS)


def main():
    parser = argparse.ArgumentParser(description="Generate policy files from config model")
    parser.add_argument("--write", action="store_true", help="Overwrite policy files")
    parser.add_argument("--check", action="store_true", help="Exit non-zero if files differ")
    args = parser.parse_args()

    bounds = generate_bounds()
    whitelist = generate_whitelist()
    protected = generate_protected_keys()

    files = {
        "bounds.json": bounds,
        "whitelist.json": whitelist,
        "protected_keys.json": protected,
    }

    if args.check:
        drift = False
        for name, data in files.items():
            path = POLICY_DIR / name
            generated = json.dumps(data, indent=2, default=_json_default) + "\n"
            if path.exists():
                existing = path.read_text()
                if existing != generated:
                    print(f"DRIFT: {name} differs from model")
                    drift = True
            else:
                print(f"MISSING: {name}")
                drift = True
        sys.exit(1 if drift else 0)

    for name, data in files.items():
        content = json.dumps(data, indent=2, default=_json_default) + "\n"
        if args.write:
            path = POLICY_DIR / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            print(f"Wrote {path}")
        else:
            print(f"--- {name} ---")
            print(content)


if __name__ == "__main__":
    main()
