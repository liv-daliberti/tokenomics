#!/bin/bash
# Interdependence gradient as a SLURM job ARRAY: one INDEPENDENT job per offset,
# all submitted at once so they run in parallel (4 fit at a time on node302's 8
# A100s). Each task serves its own Qwen (bf16, the proven config) and runs one
# match at its bias_sigma. Only bias_sigma varies; everything else is fixed.
#SBATCH -J agora-grad
#SBATCH --account=mltheory
#SBATCH --partition=mltheory
#SBATCH --nodelist=node302
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=100G
#SBATCH --cpus-per-task=6
#SBATCH --time=03:00:00
#SBATCH --array=0-9
#SBATCH -o /n/fs/similarity/tokenomics/runs/qwen/grad-%A_%a.out
#SBATCH -e /n/fs/similarity/tokenomics/runs/qwen/grad-%A_%a.out
set -uo pipefail
cd /n/fs/similarity/tokenomics

BIASES=(0 50 100 150 200 250 300 350 400 500)
BIAS=${BIASES[$SLURM_ARRAY_TASK_ID]}
PORT=$((8800 + SLURM_ARRAY_TASK_ID))

VLLM_PY=/n/fs/similarity/Illusion-of-Reasoning/openr1/bin/python
export LD_LIBRARY_PATH=/n/fs/similarity/Illusion-of-Reasoning/openr1/lib:${LD_LIBRARY_PATH:-}
export HF_HOME=/n/fs/similarity/tokenomics/.hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=0
export VLLM_LOGGING_LEVEL=WARNING
export TOKENIZERS_PARALLELISM=false
mkdir -p runs/qwen "$HF_HOME"

echo "=== task $SLURM_ARRAY_TASK_ID | bias_sigma=$BIAS | port $PORT | node $(hostname) ==="

"$VLLM_PY" -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-32B --served-model-name qwen3-32b \
  --tensor-parallel-size 2 \
  --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 \
  --max-model-len 65536 \
  --rope-scaling '{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":32768}' \
  --gpu-memory-utilization 0.92 --disable-log-requests \
  --port "$PORT" > "runs/qwen/vllm-grad-${BIAS}.log" 2>&1 &
VLLM_PID=$!

for i in $(seq 1 120); do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then echo "vllm exited early"; tail -30 "runs/qwen/vllm-grad-${BIAS}.log"; exit 1; fi
  if curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then echo "=== serving on :$PORT ==="; break; fi
  sleep 10
done

echo "=== match: bias_sigma=$BIAS at $(date) ==="
MODEL=qwen3-32b BASE_URL="http://localhost:$PORT/v1" \
  PRESET=cooperative GAMES="${GAMES:-3}" MAXTICKS="${MAXTICKS:-4}" SEED="${SEED:-0}" \
  BIAS_SIGMA="$BIAS" PRIOR_SIGMA="${PRIOR_SIGMA:-400}" \
  OUT="runs/qwen/grad_b${BIAS}" \
  "$VLLM_PY" scripts/qwen_match.py

kill "$VLLM_PID" 2>/dev/null
echo "=== task $SLURM_ARRAY_TASK_ID (bias=$BIAS) DONE at $(date) ==="
