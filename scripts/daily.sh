#!/usr/bin/env bash
# Local daily job. Does not retrain models. Does not fetch MCX/IBJA.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
export GOLD_HOME="$ROOT"
exec gold run --horizon all
