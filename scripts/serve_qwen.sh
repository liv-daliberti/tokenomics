#!/usr/bin/env bash
# Serve a local Qwen3-32B for tool-calling behind an OpenAI-compatible endpoint.
#
# Agora drives this endpoint via agora/backends.py (OpenAIBackend). One vLLM
# instance serves ALL agents concurrently (vLLM batches requests) — you do not
# need one GPU per agent.
#
# Recipe verified against vLLM v0.23.0 + Qwen/Qwen3-32B (2026). Notes:
#   * hermes tool parser is correct for the DENSE Qwen3 (not qwen3_coder/xml).
#   * reasoning-parser qwen3 strips <think> so it never pollutes tool_calls.
#   * The client must use stream=false for tool turns (streaming breaks the
#     hermes parser) and tool_choice="auto" (required is buggy on Qwen3+vLLM).
#     Those are already set in agora/backends.py.
#
# Usage:
#   scripts/serve_qwen.sh            # bf16, tensor-parallel across visible GPUs
#   PRECISION=fp8 scripts/serve_qwen.sh
#   PRECISION=awq scripts/serve_qwen.sh
set -euo pipefail

PRECISION="${PRECISION:-bf16}"       # bf16 | fp8 | awq
TP="${TP:-2}"                        # tensor-parallel size (bf16 needs >=2 on 80GB)
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-32768}"
UTIL="${UTIL:-0.90}"

case "$PRECISION" in
  bf16) MODEL="Qwen/Qwen3-32B" ;;                 # ~66GB weights -> 2x80GB
  fp8)  MODEL="Qwen/Qwen3-32B-FP8"; TP="${TP_OVERRIDE:-1}" ;;  # ~33GB -> 1x80GB
  awq)  MODEL="Qwen/Qwen3-32B-AWQ"; TP="${TP_OVERRIDE:-1}"; MAX_LEN="${MAX_LEN_OVERRIDE:-20480}" ;;  # ~20GB -> 1x24-48GB
  *) echo "unknown PRECISION=$PRECISION (use bf16|fp8|awq)"; exit 1 ;;
esac

echo "Serving $MODEL  (tp=$TP, max_len=$MAX_LEN, port=$PORT)"
exec vllm serve "$MODEL" \
  --served-model-name qwen3-32b \
  --tensor-parallel-size "$TP" \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --reasoning-parser qwen3 \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$UTIL" \
  --host 0.0.0.0 --port "$PORT"
