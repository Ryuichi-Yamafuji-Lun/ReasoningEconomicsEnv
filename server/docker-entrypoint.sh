#!/bin/sh
# Log effective default env configuration when the container starts (see EnvConfig).
set -e
cd /app/env
export PYTHONPATH="/app/env:${PYTHONPATH:-}"

echo "=== ReasoningEconomicsEnv container start ==="
echo "REE_* env vars (optional overrides for ops):"
env | grep -E '^REE_' || echo "  (none)"

python - <<'PY'
import json
from dataclasses import asdict

from env.config import env_config_for_server

cfg = env_config_for_server()
print("Effective EnvConfig for new sessions (default tokenizer_name):", cfg.tokenizer_name)
print(json.dumps(asdict(cfg), indent=2, default=str))
PY

exec uvicorn server.app:app --host 0.0.0.0 --port 8000
