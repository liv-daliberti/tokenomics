#!/bin/bash
# TRUST PROBE (pre-registered D1/D2). One LLM seat (Qwen3-32B) vs a scripted bot
# whose honesty is GROUND TRUTH on both sides, on the `probe_trust` preset
# (values_via_trade_only forces every shared reading through the lie-labelable
# escrow channel; neutral framing; no strategy hint; a soft wall so the game is
# about trust, not survival). Run as a MATCH so the LLM can learn the partner's
# honesty across games. Two array tasks = the two conditions:
#   task 0 -> POLICIES=llm,honest_cooperator   (D1: an honest partner)
#   task 1 -> POLICIES=llm,liar                (D2: a partner that fabricates at rate 1)
# Questions: does the LLM DISCOUNT a proven liar (fewer buys / lower trust over
# games), and does an LLM-judge over the transcript recover the referee's
# ground-truth lie label? Writes runs/qwen/probe_<bot>_s<seed>.jsonl.
#   sbatch scripts/agora_probe.slurm            # seeds 0-4, both conditions
#   SEEDS="0 1 2 3 4 5 6 7 8 9" sbatch scripts/agora_probe.slurm
#SBATCH -J agora-probe
#SBATCH --account=mltheory
#SBATCH --partition=mltheory
#SBATCH --nodelist=node302
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=100G
#SBATCH --cpus-per-task=6
#SBATCH --time=16:00:00
#SBATCH --array=0-1
#SBATCH -o /n/fs/similarity/tokenomics/runs/qwen/probe-%A_%a.out
#SBATCH -e /n/fs/similarity/tokenomics/runs/qwen/probe-%A_%a.out
set -uo pipefail
cd /n/fs/similarity/tokenomics

BOTS=(honest_cooperator liar)
BOT=${BOTS[$SLURM_ARRAY_TASK_ID]}
PORT=$(( ${PORT_BASE:-8980} + SLURM_ARRAY_TASK_ID ))
SEEDS="${SEEDS:-0 1 2 3 4}"

VLLM_PY=/n/fs/similarity/Illusion-of-Reasoning/openr1/bin/python
export LD_LIBRARY_PATH=/n/fs/similarity/Illusion-of-Reasoning/openr1/lib:${LD_LIBRARY_PATH:-}
export HF_HOME=/n/fs/similarity/tokenomics/.hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=0
export VLLM_LOGGING_LEVEL=WARNING
export TOKENIZERS_PARALLELISM=false
export VLLM_CACHE_ROOT="/n/fs/similarity/tokenomics/runs/qwen/.vllm_cache/probe_t${SLURM_ARRAY_TASK_ID}"
mkdir -p runs/qwen "$HF_HOME" "$VLLM_CACHE_ROOT"

echo "=== probe task $SLURM_ARRAY_TASK_ID | bot=$BOT | seeds=$SEEDS | port $PORT | node $(hostname) ==="

"$VLLM_PY" -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-32B --served-model-name qwen3-32b \
  --tensor-parallel-size 2 \
  --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 \
  --max-model-len 131072 \
  --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
  --gpu-memory-utilization 0.92 --disable-log-requests \
  --port "$PORT" > "runs/qwen/vllm-probe-${BOT}.log" 2>&1 &
VLLM_PID=$!
for i in $(seq 1 120); do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then echo "vllm exited early"; tail -30 "runs/qwen/vllm-probe-${BOT}.log"; exit 1; fi
  if curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then echo "=== serving on :$PORT ==="; break; fi
  sleep 10
done

for SEED in $SEEDS; do
  OUT="runs/qwen/probe_${BOT}_s${SEED}"
  if grep -ql match_end "${OUT}.jsonl" 2>/dev/null; then
    echo "=== probe bot=$BOT seed=$SEED already complete — skip ==="
    continue
  fi
  echo "=== probe bot=$BOT seed=$SEED at $(date) ==="
  MODEL=qwen3-32b BASE_URL="http://localhost:$PORT/v1" \
    PRESET=probe_trust POLICIES="llm,${BOT}" GAMES="${GAMES:-10}" SEED="$SEED" \
    OUT="$OUT" \
    "$VLLM_PY" scripts/qwen_match.py
done

kill "$VLLM_PID" 2>/dev/null
echo "=== probe task $SLURM_ARRAY_TASK_ID (bot=$BOT) DONE at $(date) ==="
