#!/usr/bin/env bash
# Run the custom LUNAR search from the repository root.
#
# Usage:
#   bash custom/run.sh            # runs the 'test' preset
#   bash custom/run.sh default    # runs the 'default' preset
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
python -m custom.main --preset "${1:-test}"
