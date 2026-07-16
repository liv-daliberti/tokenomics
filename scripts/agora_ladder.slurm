#!/bin/bash
# THE GAP LADDER on Qwen3-32B — the Qwen arm of the knowing-doing dose-response
# (the GPT-5.4 arm is `python scripts/gpt54_program.py --stage ladder`).
# Escalating scaffolds on the trust probe, exactly ONE per rung — what makes
# the model ACT on the fabrication it can detect?
#   mem    -> cross-game markdown notebook (MEMORY=markdown)
#   elicit -> respond_trade REQUIRES a stated p_fabricated (ELICIT_PFAB=1)
#   hist   -> seller track record rendered next to each offer (SHOW_SELLER_HISTORY=1)
#   flag   -> live SELF-judge p(fabricated) injected next to the offer
#             (SHOW_JUDGE_FLAG=1 — judged by the SAME vLLM server, no extra GPU)
# R0 baseline = the existing trust_* runs (see agora_trust.slurm) — not re-run.
# Eight array tasks = rung x bot; each runs both difficulties x seeds:
#   task = rung_index * 2 + bot_index   (bots: 0=honest_cooperator, 1=liar)
# Writes runs/qwen/ladder_<rung>_<bot>_<easy|hard>_b<off>_s<seed>.jsonl — the
# SAME names the GPT stage uses, so scripts/ladder_report.py scores both arms.
#   sbatch scripts/agora_ladder.slurm                 # seeds 0-1, all rungs
#   SEEDS="0 1 2" sbatch scripts/agora_ladder.slurm
#SBATCH -J agora-ladder
#SBATCH --account=mltheory
#SBATCH --partition=mltheory
#SBATCH --nodelist=node302
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=100G
#SBATCH --cpus-per-task=6
#SBATCH --time=16:00:00
#SBATCH --array=0-7%4
#SBATCH -o /n/fs/similarity/tokenomics/runs/qwen/ladder-%A_%a.out
#SBATCH -e /n/fs/similarity/tokenomics/runs/qwen/ladder-%A_%a.out
set -uo pipefail
cd /n/fs/similarity/tokenomics

RUNGS=(mem elicit hist flag)
BOTS=(honest_cooperator liar)
RUNG=${RUNGS[$(( SLURM_ARRAY_TASK_ID / 2 ))]}
BOT=${BOTS[$(( SLURM_ARRAY_TASK_ID % 2 ))]}
PORT=$(( ${PORT_BASE:-8996} + SLURM_ARRAY_TASK_ID ))
SEEDS="${SEEDS:-0 1}"                       # pilot: 2 seeds; override for more
# difficulty levels as "label:offset" — matches the trust stage (easy=0, hard=200)
LEVELS=("easy:0" "hard:200")
# each rung sets exactly one scaffold knob (qwen_match.py env -> GameConfig)
case "$RUNG" in
  mem)    KNOBS=(MEMORY=markdown) ;;
  elicit) KNOBS=(ELICIT_PFAB=1) ;;
  hist)   KNOBS=(SHOW_SELLER_HISTORY=1) ;;
  flag)   KNOBS=(SHOW_JUDGE_FLAG=1) ;;
esac

VLLM_PY=/n/fs/similarity/Illusion-of-Reasoning/openr1/bin/python
export LD_LIBRARY_PATH=/n/fs/similarity/Illusion-of-Reasoning/openr1/lib:${LD_LIBRARY_PATH:-}
export HF_HOME=/n/fs/similarity/tokenomics/.hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=0
export VLLM_LOGGING_LEVEL=WARNING
export TOKENIZERS_PARALLELISM=false
export VLLM_CACHE_ROOT="/n/fs/similarity/tokenomics/runs/qwen/.vllm_cache/ladder_t${SLURM_ARRAY_TASK_ID}"
mkdir -p runs/qwen "$HF_HOME" "$VLLM_CACHE_ROOT"

echo "=== ladder task $SLURM_ARRAY_TASK_ID | rung=$RUNG bot=$BOT | seeds=$SEEDS | port $PORT | node $(hostname) ==="

# TP and MAXLEN default to the A100 recipe (2x80GB, full 131072 window) but are
# overridable so the same script fits smaller GPUs — e.g. the cs partition's
# 2x A6000 (48GB): TP=2 MAXLEN=65536 keeps the 65GB weights + KV cache in 96GB.
# Ladder matches are short (GAMES=5, one LLM seat), so a 64K window is ample.
TP="${TP:-2}"
MAXLEN="${MAXLEN:-131072}"
# yarn 4x lifts Qwen3's native 32768 window; only needed when MAXLEN>32768
ROPE='--rope-scaling {"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}'
[ "$MAXLEN" -le 32768 ] && ROPE=""
echo "=== serving Qwen3-32B: TP=$TP max-model-len=$MAXLEN ==="
"$VLLM_PY" -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-32B --served-model-name qwen3-32b \
  --tensor-parallel-size "$TP" \
  --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 \
  --max-model-len "$MAXLEN" $ROPE \
  --gpu-memory-utilization 0.92 --disable-log-requests \
  --port "$PORT" > "runs/qwen/vllm-ladder-${RUNG}-${BOT}.log" 2>&1 &
VLLM_PID=$!
for i in $(seq 1 120); do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then echo "vllm exited early"; tail -30 "runs/qwen/vllm-ladder-${RUNG}-${BOT}.log"; exit 1; fi
  if curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then echo "=== serving on :$PORT ==="; break; fi
  sleep 10
done

for LV in "${LEVELS[@]}"; do
  LABEL=${LV%%:*}; OFF=${LV##*:}
  for SEED in $SEEDS; do
    OUT="runs/qwen/ladder_${RUNG}_${BOT}_${LABEL}_b${OFF}_s${SEED}"
    if grep -ql match_end "${OUT}.jsonl" 2>/dev/null; then
      echo "=== ladder rung=$RUNG bot=$BOT $LABEL(σ=$OFF) seed=$SEED already complete — skip ==="
      continue
    fi
    echo "=== ladder rung=$RUNG bot=$BOT $LABEL(σ=$OFF) seed=$SEED at $(date) ==="
    env MODEL=qwen3-32b BASE_URL="http://localhost:$PORT/v1" \
      PRESET=probe_trust POLICIES="llm,${BOT}" BIAS_SIGMA="$OFF" \
      VALUES_VIA_TRADE_ONLY=1 REQUIRE_PAID_TRADES=1 \
      GAMES="${GAMES:-5}" SEED="$SEED" OUT="$OUT" "${KNOBS[@]}" \
      "$VLLM_PY" scripts/qwen_match.py
  done
done

kill "$VLLM_PID" 2>/dev/null
echo "=== ladder task $SLURM_ARRAY_TASK_ID (rung=$RUNG bot=$BOT) DONE at $(date) ==="
