#!/bin/bash
# DIFFICULTY x COST x HONESTY grid on Qwen3-32B — the Qwen arm of the cost-error
# scatter (GPT arm: python scripts/gpt54_program.py --stage grid). One LLM seat
# vs a scripted partner at every combination of difficulty {offset 0, 200},
# information price {1, 8, 32} and partner honesty {liar, honest_cooperator,
# mixed_liar}, under the paid market. Writes runs/qwen/grid_<partner>_b<off>_
# p<price>_s<seed>.jsonl — the SAME names the GPT stage uses, so
# scripts/plot_cost_error.py plots both models together.
# One array task per partner (each runs offsets x prices x seeds):
#   sbatch scripts/agora_grid.slurm                    # cs partition, node206
#   SEEDS="0 1" sbatch scripts/agora_grid.slurm
# GAMES defaults to 4 so the chatty honest matches fit the A6000 64K window on
# TP=2 (bump with more GPUs / a bigger window for GAMES=5 parity with GPT).
#SBATCH -J agora-grid
#SBATCH --account=allcs
#SBATCH --partition=cs
#SBATCH --nodelist=node206
#SBATCH --gres=gpu:a6000:2
#SBATCH --mem=100G
#SBATCH --cpus-per-task=6
#SBATCH --time=10:00:00
#SBATCH --array=0-2
#SBATCH -o /n/fs/similarity/tokenomics/runs/qwen/grid-%A_%a.out
#SBATCH -e /n/fs/similarity/tokenomics/runs/qwen/grid-%A_%a.out
set -uo pipefail
cd /n/fs/similarity/tokenomics

PARTNERS=(liar honest_cooperator mixed_liar)
PARTNER=${PARTNERS[$SLURM_ARRAY_TASK_ID]}
PORT=$(( ${PORT_BASE:-8992} + SLURM_ARRAY_TASK_ID ))
SEEDS="${SEEDS:-0 1}"
OFFSETS="${OFFSETS:-0 200}"
PRICES="${PRICES:-1 8 32}"
START_CREDITS="${START_CREDITS:-60}"
TP="${TP:-2}"; MAXLEN="${MAXLEN:-65536}"

VLLM_PY=/n/fs/similarity/Illusion-of-Reasoning/openr1/bin/python
export LD_LIBRARY_PATH=/n/fs/similarity/Illusion-of-Reasoning/openr1/lib:${LD_LIBRARY_PATH:-}
export HF_HOME=/n/fs/similarity/tokenomics/.hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=0
export VLLM_LOGGING_LEVEL=WARNING
export TOKENIZERS_PARALLELISM=false
export VLLM_CACHE_ROOT="/n/fs/similarity/tokenomics/runs/qwen/.vllm_cache/grid_t${SLURM_ARRAY_TASK_ID}"
mkdir -p runs/qwen "$HF_HOME" "$VLLM_CACHE_ROOT"

ROPE='--rope-scaling {"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}'
[ "$MAXLEN" -le 32768 ] && ROPE=""
echo "=== grid task $SLURM_ARRAY_TASK_ID | partner=$PARTNER | port $PORT | TP=$TP len=$MAXLEN | $(hostname) ==="
"$VLLM_PY" -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-32B --served-model-name qwen3-32b \
  --tensor-parallel-size "$TP" \
  --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 \
  --max-model-len "$MAXLEN" $ROPE \
  --gpu-memory-utilization 0.92 --disable-log-requests \
  --port "$PORT" > "runs/qwen/vllm-grid-${PARTNER}.log" 2>&1 &
VLLM_PID=$!
for i in $(seq 1 120); do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then echo "vllm exited early"; tail -30 "runs/qwen/vllm-grid-${PARTNER}.log"; exit 1; fi
  if curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then echo "=== serving on :$PORT ==="; break; fi
  sleep 10
done

for OFF in $OFFSETS; do
  for PRICE in $PRICES; do
    TAG="p${PRICE/./_}"
    for SEED in $SEEDS; do
      OUT="runs/qwen/grid_${PARTNER}_b${OFF}_${TAG}_s${SEED}"
      if grep -ql match_end "${OUT}.jsonl" 2>/dev/null; then
        echo "=== grid $PARTNER b$OFF $TAG s$SEED already complete — skip ==="; continue
      fi
      echo "=== grid $PARTNER b$OFF price$PRICE s$SEED at $(date) ==="
      MODEL=qwen3-32b BASE_URL="http://localhost:$PORT/v1" \
        PRESET=probe_trust POLICIES="llm,${PARTNER}" BIAS_SIGMA="$OFF" \
        MIN_TRADE_PRICE="$PRICE" STARTING_CREDITS="$START_CREDITS" \
        VALUES_VIA_TRADE_ONLY=1 REQUIRE_PAID_TRADES=1 \
        GAMES="${GAMES:-4}" SEED="$SEED" OUT="$OUT" \
        "$VLLM_PY" scripts/qwen_match.py
    done
  done
done

kill "$VLLM_PID" 2>/dev/null
echo "=== grid task $SLURM_ARRAY_TASK_ID ($PARTNER) DONE at $(date) ==="
