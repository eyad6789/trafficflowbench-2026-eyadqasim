#!/usr/bin/env bash
# One-command launcher for the reproduction package (see README.md).
#   ./run.sh [--data DIR] [--assets DIR] [--out DIR] [other run_all.py flags]   reproduce the selected submission
#   ./run.sh check                                                             check Python and the libraries only
#   ./run.sh help                                                              this text and the run_all.py flags
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"

need() {
  command -v "$PY" >/dev/null 2>&1 || { echo "error: $PY not found. Install Python 3.11+ or set PYTHON=/path/to/python." >&2; exit 1; }
  "$PY" - <<'EOF' || exit 1
import importlib.util, sys
missing = [m for m in ("numpy", "pandas", "scipy", "pyarrow", "lightgbm", "catboost", "torch") if importlib.util.find_spec(m) is None]
if missing:
    sys.exit("error: missing libraries: " + ", ".join(missing) + "\nfix: pip install numpy pandas scipy pyarrow lightgbm catboost torch  (versions in README.md)")
import torch
print("python", sys.version.split()[0], "| torch", torch.__version__, "| GPU", torch.cuda.is_available())
EOF
}

case "${1:-}" in
  help|-h|--help)
    sed -n '2,5p' "$0" | sed 's/^# \{0,1\}//'
    echo
    "$PY" run_all.py --help
    ;;
  check)
    need
    echo "ready: run ./run.sh --data <competition input> --assets <public assets> --out <output folder>"
    ;;
  stop)
    echo "nothing to stop: run_all.py starts no servers or containers"
    ;;
  *)
    need
    exec "$PY" run_all.py "$@"
    ;;
esac
