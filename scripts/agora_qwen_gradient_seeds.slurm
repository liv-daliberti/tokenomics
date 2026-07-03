#!/bin/bash
# Multi-seed interdependence gradient: one array task PER OFFSET, each serving Qwen
# once and running SEEDS matches (default 10 seeds) so we can plot mean ± CI. 10
# tasks, ~4 run at a time on node302's 8 A100s. Only bias_sigma varies; prior,
# noise, budget, survival cost, horizon are fixed (prior σ = 400, wide).
# Each match = 10 games x 5 rounds; agents keep the FULL match context (one growing
# conversation). Low-offset matches (both agents survive all 10 games) accumulate the
# most and peaked near ~98k tokens, so the window is 131072 (yarn x4, Qwen3's max) to
# hold even the chattiest full match. Re-submitting only backfills incomplete seeds.
#SBATCH -J agora-grad-seeds
#SBATCH --account=mltheory
#SBATCH --partition=mltheory
#SBATCH --nodelist=node302
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=100G
#SBATCH --cpus-per-task=6
#SBATCH --time=20:00:00
#SBATCH --array=0-9
#SBATCH -o /n/fs/similarity/tokenomics/runs/qwen/grads-%A_%a.out
#SBATCH -e /n/fs/similarity/tokenomics/runs/qwen/grads-%A_%a.out
set -uo pipefail
cd /n/fs/similarity/tokenomics

# Offsets to sweep, one per array task. Override with OFFSETS="10 20 30 40" plus a
# matching --array=0-N to run a finer or partial sweep (e.g. resolving the switch
# between 0 and 50). Default is the full 10-point dial.
read -ra BIASES <<< "${OFFSETS:-0 50 100 150 200 250 300 350 400 500}"
BIAS=${BIASES[$SLURM_ARRAY_TASK_ID]}
# PORT_BASE lets a second seed-batch run alongside the first without colliding on
# ports while both drain on the same node (GPUs are still gated by SLURM gres).
PORT=$(( ${PORT_BASE:-8900} + SLURM_ARRAY_TASK_ID ))
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
# PREFIX names the output family so a second sweep (e.g. the DE-CONFOUNDED control,
# FRAMING=neutral STRATEGY_HINT=0) writes distinct files instead of mixing into — or
# being skipped by the backfill against — the confounded grad_b* runs. Give a second
# concurrent sweep its own PORT_BASE too (e.g. PREFIX=deconf PORT_BASE=8940).
#   PREFIX=deconf PORT_BASE=8940 FRAMING=neutral STRATEGY_HINT=0 sbatch --array=0-9 ...
PREFIX="${PREFIX:-grad}"

VLLM_PY=/n/fs/similarity/Illusion-of-Reasoning/openr1/bin/python
export LD_LIBRARY_PATH=/n/fs/similarity/Illusion-of-Reasoning/openr1/lib:${LD_LIBRARY_PATH:-}
export HF_HOME=/n/fs/similarity/tokenomics/.hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=0
export VLLM_LOGGING_LEVEL=WARNING
export TOKENIZERS_PARALLELISM=false
# Per-task torch.compile cache: the array tasks start together and would otherwise
# race on one shared cache (~/.cache/vllm), corrupting it when a new config forces a
# recompile. An isolated cache root per task removes the race.
# Cache root is per-task AND per-PREFIX so a de-confounded sweep doesn't race the
# confounded one on the same compile cache (grad keeps its original t<id> path).
CACHE_TAG="t${SLURM_ARRAY_TASK_ID}"
[ "$PREFIX" != grad ] && CACHE_TAG="${PREFIX}_t${SLURM_ARRAY_TASK_ID}"
export VLLM_CACHE_ROOT="/n/fs/similarity/tokenomics/runs/qwen/.vllm_cache/${CACHE_TAG}"
mkdir -p runs/qwen "$HF_HOME" "$VLLM_CACHE_ROOT"

echo "=== task $SLURM_ARRAY_TASK_ID | prefix=$PREFIX bias_sigma=$BIAS | framing=${FRAMING:-preset} hint=${STRATEGY_HINT:-preset} | seeds=$SEEDS | port $PORT | node $(hostname) ==="

"$VLLM_PY" -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-32B --served-model-name qwen3-32b \
  --tensor-parallel-size 2 \
  --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 \
  --max-model-len 131072 \
  --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
  --gpu-memory-utilization 0.92 --disable-log-requests \
  --port "$PORT" > "runs/qwen/vllm-${PREFIX}s-${BIAS}.log" 2>&1 &
VLLM_PID=$!
for i in $(seq 1 120); do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then echo "vllm exited early"; tail -30 "runs/qwen/vllm-${PREFIX}s-${BIAS}.log"; exit 1; fi
  if curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then echo "=== serving on :$PORT ==="; break; fi
  sleep 10
done

for SEED in $SEEDS; do
  OUT="runs/qwen/${PREFIX}_b${BIAS}_s${SEED}"
  # BACKFILL: skip seeds that already finished (a complete transcript ends with
  # match_end). Re-submitting the array only re-runs missing/partial matches — e.g.
  # the low-offset ones that overflowed the smaller window — and leaves good runs be.
  if grep -ql match_end "${OUT}.jsonl" 2>/dev/null; then
    echo "=== $PREFIX bias=$BIAS seed=$SEED already complete — skip ==="
    continue
  fi
  echo "=== $PREFIX bias=$BIAS seed=$SEED at $(date) ==="
  MODEL=qwen3-32b BASE_URL="http://localhost:$PORT/v1" \
    PRESET=cooperative GAMES="${GAMES:-10}" ROUNDS="${ROUNDS:-5}" MAXTICKS="${MAXTICKS:-4}" SEED="$SEED" \
    BIAS_SIGMA="$BIAS" PRIOR_SIGMA="${PRIOR_SIGMA:-400}" \
    FRAMING="${FRAMING:-}" STRATEGY_HINT="${STRATEGY_HINT:-}" \
    OUT="$OUT" \
    "$VLLM_PY" scripts/qwen_match.py
done

kill "$VLLM_PID" 2>/dev/null
echo "=== task $SLURM_ARRAY_TASK_ID (bias=$BIAS) DONE at $(date) ==="
