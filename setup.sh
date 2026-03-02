#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

echo "=== Extended Market Maker Setup ==="

# 1. Python check
PYTHON="${PYTHON:-python3}"
PY_VERSION=$($PYTHON --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
  echo "ERROR: Python >= 3.10 required (found $PY_VERSION)"
  exit 1
fi
echo "[1/5] Python $PY_VERSION OK"

# 2. Git submodules
if [ ! -f python_sdk/pyproject.toml ]; then
  echo "[2/5] Initializing git submodules..."
  git submodule update --init --recursive
else
  echo "[2/5] Submodules OK"
fi

# 3. Virtual environment
if [ ! -d .venv ]; then
  echo "[3/5] Creating virtual environment..."
  $PYTHON -m venv .venv
else
  echo "[3/5] Virtual environment exists"
fi

# 4. Install dependencies
echo "[4/5] Installing x10 SDK (python_sdk submodule)..."
.venv/bin/pip install -e python_sdk --quiet

echo "[5/5] Installing market maker + dev deps..."
.venv/bin/pip install -e ".[dev]" --quiet

# 5. Verify
echo ""
echo "=== Verification ==="
.venv/bin/python -c "from market_maker.cli import main; print('mmctl: OK')"
.venv/bin/python -c "import x10; print('x10 SDK: OK')"
echo ""
echo "Setup complete. Activate with:"
echo "  source .venv/bin/activate"
echo ""
echo "Then configure:"
echo "  cp .env.example .env"
echo "  # Fill in MM credentials"
echo ""
echo "Usage:"
echo "  mmctl --help"
echo "  mmctl start eth"
echo "  mmctl status --json"
echo "  mmctl pnl ETH-USD --days 7 --json"
