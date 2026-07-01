#!/usr/bin/env bash
# Phase 1 smoke test: run one tiny game with REAL local Qwen agents, then build
# metrics + an HTML report. Assumes vLLM is already serving (scripts/serve_qwen.sh).
#
#   scripts/smoke_qwen.sh
#   BASE_URL=http://gpu-node:8000/v1 MODEL=qwen3-32b scripts/smoke_qwen.sh
set -euo pipefail
cd "$(dirname "$0")/.."

BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
MODEL="${MODEL:-qwen3-32b}"
OUT="${OUT:-runs/qwen_smoke}"

echo "==> checking endpoint $BASE_URL"
if ! curl -sf "${BASE_URL}/models" >/dev/null; then
  echo "ERROR: no OpenAI-compatible server at $BASE_URL"
  echo "       start one first:  scripts/serve_qwen.sh"
  exit 1
fi
curl -s "${BASE_URL}/models" | python3 -c 'import sys,json;print("    served models:",[m["id"] for m in json.load(sys.stdin)["data"]])' || true

echo "==> running Qwen smoke game (2 agents, 2 rounds)"
python3 -m agora.run --config configs/qwen_smoke.yaml --policies llm \
    --model "$MODEL" --base-url "$BASE_URL" --out "$OUT"

echo "==> full metrics"
python3 -m analysis.metrics "$OUT"/*.jsonl

echo "==> building HTML report"
python3 -m analysis.viz "$OUT" -o report/qwen_smoke
echo "==> open report/qwen_smoke/index.html"
