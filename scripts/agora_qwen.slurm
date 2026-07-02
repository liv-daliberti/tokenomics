#!/bin/bash
#SBATCH -J agora-qwen
#SBATCH --account=mltheory
#SBATCH --partition=mltheory
#SBATCH --nodelist=node302
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH -o /n/fs/similarity/tokenomics/runs/qwen/slurm-%j.out
#SBATCH -e /n/fs/similarity/tokenomics/runs/qwen/slurm-%j.out
set -uo pipefail
cd /n/fs/similarity/tokenomics

# openr1 is a conda env whose python already has vllm 0.9.2; run it directly
# (no activate script exists) as a module, and expose its libs.
VLLM_PY=/n/fs/similarity/Illusion-of-Reasoning/openr1/bin/python
export LD_LIBRARY_PATH=/n/fs/similarity/Illusion-of-Reasoning/openr1/lib:${LD_LIBRARY_PATH:-}
export HF_HOME=/n/fs/similarity/tokenomics/.hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=0
export VLLM_LOGGING_LEVEL=WARNING
export TOKENIZERS_PARALLELISM=false
PORT=8765
mkdir -p runs/qwen "$HF_HOME"

echo "=== node $(hostname) | GPUs ==="; nvidia-smi -L

# node302 has 80GB A100s, so native bf16 (most reliable for tool-calling) fits on
# 2 GPUs. FP8/AWQ are fallbacks if the bf16 download/serve fails.
# Each entry: "<hf_repo> <tensor_parallel_size>"
CANDIDATES=(
  "Qwen/Qwen3-32B 2"
  "Qwen/Qwen3-32B-FP8 1"
  "Qwen/Qwen3-32B-AWQ 1"
)

serve_and_wait() {
  local repo="$1" tp="$2"
  echo "=== trying $repo (tp=$tp) ==="
  "$VLLM_PY" -m vllm.entrypoints.openai.api_server \
    --model "$repo" \
    --served-model-name qwen3-32b \
    --tensor-parallel-size "$tp" \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --reasoning-parser qwen3 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.92 \
    --disable-log-requests \
    --port "$PORT" > runs/qwen/vllm.log 2>&1 &
  VLLM_PID=$!
  # wait up to ~20 min (covers a cold download + load)
  for i in $(seq 1 120); do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then echo "vllm exited early"; return 1; fi
    if curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
      echo "=== $repo is serving on :$PORT ==="; return 0
    fi
    sleep 10
  done
  echo "timed out waiting for $repo"; kill "$VLLM_PID" 2>/dev/null; return 1
}

SERVED=""
for c in "${CANDIDATES[@]}"; do
  set -- $c
  if serve_and_wait "$1" "$2"; then SERVED="$1"; break; fi
  echo "---- last 40 lines of vllm.log for $1 ----"; tail -40 runs/qwen/vllm.log
done

if [ -z "$SERVED" ]; then echo "FATAL: no Qwen variant could be served"; exit 1; fi

echo "=== running Agora match against $SERVED ==="
MODEL=qwen3-32b BASE_URL="http://localhost:$PORT/v1" \
  PRESET=cooperative GAMES=2 ROUNDS=3 MAXTICKS=4 SEED=0 \
  OUT=runs/qwen/cooperative_match \
  "$VLLM_PY" scripts/qwen_match.py

echo "=== shutting down vLLM ==="
kill "$VLLM_PID" 2>/dev/null
echo "=== DONE ($SERVED) ==="
