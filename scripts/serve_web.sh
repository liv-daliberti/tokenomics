#!/usr/bin/env bash
# Launch the Agora web UI. Run a game with a button; read it as a report.
#   scripts/serve_web.sh
#   HOST=0.0.0.0 PORT=8080 scripts/serve_web.sh      # expose on a network
set -euo pipefail
cd "$(dirname "$0")/.."
export HOST="${HOST:-127.0.0.1}" PORT="${PORT:-5000}"
python3 -c 'import flask' 2>/dev/null || { echo "Flask not installed: pip install flask"; exit 1; }
echo "Agora web UI -> http://$HOST:$PORT"
exec python3 -m web.app
