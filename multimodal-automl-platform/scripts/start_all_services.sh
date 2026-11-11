#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python - <<'PY'
import importlib.util, sys
required = ["fastapi", "uvicorn", "pandas", "sklearn", "skimage", "PIL"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    print("Missing Python packages:", ", ".join(missing))
    sys.exit(1)
PY
exec python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
