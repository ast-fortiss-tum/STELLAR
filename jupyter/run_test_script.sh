#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

cd "$repo_root"

PYTHONUNBUFFERED=1 PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}" python -m tarot.run_test