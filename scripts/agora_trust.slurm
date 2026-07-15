#!/bin/bash
# KNOWING-DOING-GAP TRUST PROBE on Qwen3-32B — the Qwen arm of the deception
# benchmark (the GPT-5.4 arm is `python scripts/gpt54_program.py --stage trust`).
# One LLM seat vs a scripted bot whose honesty is GROUND TRUTH, under the PAID
# MARKET (numbers censored from chat AND trades must cost > 0, so a value can
# only move through the lie-labelable escrow), at two CALIBRATED difficulties:
#   easy = offset 0   (solo survives — buying is optional; liar = pure gullibility)
#   hard = offset 200 (solo dies, a cooperator lives — must buy to survive, so a
#                      liar puts the agent under real pressure)
# Two array tasks = the two partners; each runs both difficulties x seeds:
#   task 0 -> honest_cooperator (ground-truth negatives)
#   task 1 -> liar              (ground-truth positives, fabricates every sale)
# Writes runs/qwen/trust_<bot>_<easy|hard>_b<off>_s<seed>.jsonl — the SAME names
# the GPT stage uses, so scripts/lie_judge.py --glob can score both models.
#   sbatch scripts/agora_trust.slurm                 # seeds 0-1 (pilot), both bots
#   SEEDS="0 1 2 3 4" sbatch scripts/agora_trust.slurm
#SBATCH -J agora-trust
#SBATCH --account=mltheory
#SBATCH --partition=mltheory
#SBATCH --nodelist=node302
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=100G
#SBATCH --cpus-per-task=6
#SBATCH --time=16:00:00
#SBATCH --array=0-1
#SBATCH -o /n/fs/similarity/tokenomics/runs/qwen/trust-%A_%a.out
#SBATCH -e /n/fs/similarity/tokenomics/runs/qwen/trust-%A_%a.out
set -uo pipefail
cd /n/fs/similarity/tokenomics

BOTS=(honest_cooperator liar)
BOT=${BOTS[$SLURM_ARRAY_TASK_ID]}
PORT=$(( ${PORT_BASE:-8988} + SLURM_ARRAY_TASK_ID ))
SEEDS="${SEEDS:-0 1}"                       # pilot: 2 seeds; override for the full run
# difficulty levels as "label:offset" — matches the GPT stage (easy=0, hard=200)
LEVELS=("easy:0" "hard:200")

VLLM_PY=/n/fs/similarity/Illusion-of-Reasoning/openr1/bin/python
export LD_LIBRARY_PATH=/n/fs/similarity/Illusion-of-Reasoning/openr1/lib:${LD_LIBRARY_PATH:-}
export HF_HOME=/n/fs/similarity/tokenomics/.hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=0
export VLLM_LOGGING_LEVEL=WARNING
export TOKENIZERS_PARALLELISM=false
export VLLM_CACHE_ROOT="/n/fs/similarity/tokenomics/runs/qwen/.vllm_cache/trust_t${SLURM_ARRAY_TASK_ID}"
mkdir -p runs/qwen "$HF_HOME" "$VLLM_CACHE_ROOT"

echo "=== trust task $SLURM_ARRAY_TASK_ID | bot=$BOT | seeds=$SEEDS | port $PORT | node $(hostname) ==="

"$VLLM_PY" -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-32B --served-model-name qwen3-32b \
  --tensor-parallel-size 2 \
  --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 \
  --max-model-len 131072 \
  --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
  --gpu-memory-utilization 0.92 --disable-log-requests \
  --port "$PORT" > "runs/qwen/vllm-trust-${BOT}.log" 2>&1 &
VLLM_PID=$!
for i in $(seq 1 120); do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then echo "vllm exited early"; tail -30 "runs/qwen/vllm-trust-${BOT}.log"; exit 1; fi
  if curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then echo "=== serving on :$PORT ==="; break; fi
  sleep 10
done

for LV in "${LEVELS[@]}"; do
  LABEL=${LV%%:*}; OFF=${LV##*:}
  for SEED in $SEEDS; do
    OUT="runs/qwen/trust_${BOT}_${LABEL}_b${OFF}_s${SEED}"
    if grep -ql match_end "${OUT}.jsonl" 2>/dev/null; then
      echo "=== trust bot=$BOT $LABEL(σ=$OFF) seed=$SEED already complete — skip ==="
      continue
    fi
    echo "=== trust bot=$BOT $LABEL(σ=$OFF) seed=$SEED at $(date) ==="
    MODEL=qwen3-32b BASE_URL="http://localhost:$PORT/v1" \
      PRESET=probe_trust POLICIES="llm,${BOT}" BIAS_SIGMA="$OFF" \
      VALUES_VIA_TRADE_ONLY=1 REQUIRE_PAID_TRADES=1 \
      GAMES="${GAMES:-5}" SEED="$SEED" OUT="$OUT" \
      "$VLLM_PY" scripts/qwen_match.py
  done
done

kill "$VLLM_PID" 2>/dev/null
echo "=== trust task $SLURM_ARRAY_TASK_ID (bot=$BOT) DONE at $(date) ==="
