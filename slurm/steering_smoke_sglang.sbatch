#!/bin/bash
# Steering SMOKE with the FREE local judge: SGLang serves Qwen2.5-32B as an
# OpenAI-compatible judge, co-located with the 70B steering on ONE h200 (141GB:
# 70B-4bit ~40GB + Qwen32B server ~70GB). One emotion (happy), layer 49, 2 steps.
# Validates the whole free-judge path end-to-end before the 171 sweep. Submit:
#   sbatch slurm/steering_smoke_sglang.sbatch
#SBATCH --job-name=ti_steer_sgl_smoke
#SBATCH --partition=gpu,multigpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/%u/traitinterp/logs/%x_%j.out
#SBATCH --error=/scratch/%u/traitinterp/logs/%x_%j.err

set -uo pipefail
CONDA_BASE=/shared/EL9/explorer/miniconda3/25.9.1/miniconda3
ENV_PREFIX=/scratch/$USER/conda_envs/traitinterp
JUDGE_VENV=/scratch/$USER/judge_env
REPO=/scratch/$USER/traitinterp
PORT=30000
JUDGE_MODEL=Qwen/Qwen2.5-32B-Instruct
SRVLOG=$REPO/logs/sglang_server_${SLURM_JOB_ID}.log

echo "==== ti_steer_sgl_smoke $SLURM_JOB_ID on $(hostname) GPU=$CUDA_VISIBLE_DEVICES $(date) ===="
export HF_HOME=/scratch/$USER/hf_cache
[ -f "$HOME/.config/traitinterp/env" ] && source "$HOME/.config/traitinterp/env"
# Explorer's Squid proxy hijacks localhost → sglang warmup + judge client get 403.
# Bypass the proxy for local addresses (keep http_proxy for real outbound).
export NO_PROXY="localhost,127.0.0.1,0.0.0.0"
export no_proxy="localhost,127.0.0.1,0.0.0.0"
export PATH="$HOME/.local/bin:$PATH"
# On h200 (Hopper sm_90) sglang JIT-compiles flashinfer kernels → needs ninja
# (installed in judge_env) + gcc (system) + nvcc. Load the CUDA toolkit so nvcc is
# on PATH; the compiled kernels are cached after the first run.
module load cuda/12.8.0 2>/dev/null || true

cd "$REPO"

# ---- 1. Launch the SGLang judge server (its own venv) in the background, same GPU ----
# Run in a subshell with judge_env/bin ON PATH so any JIT build tool (ninja) is found,
# and use the triton attention backend to avoid flashinfer's ninja/nvcc CUDA JIT entirely.
echo "---- launch SGLang judge ($JUDGE_MODEL), co-located, mem-fraction 0.5, triton backend ----"
( export PATH="$JUDGE_VENV/bin:$PATH"; exec "$JUDGE_VENV/bin/python" -m sglang.launch_server \
    --model-path "$JUDGE_MODEL" --port $PORT --attention-backend triton \
    --mem-fraction-static 0.5 --context-length 4096 ) > "$SRVLOG" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null' EXIT

echo "---- wait for judge server ready (Qwen32B cold load ~14 min) ----"
READY=0
for i in $(seq 1 240); do
    if curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then READY=1; break; fi
    if grep -q "Application startup complete" "$SRVLOG" 2>/dev/null; then
        sleep 3; curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && { READY=1; break; }
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then echo "SERVER DIED — tail:"; tail -30 "$SRVLOG"; exit 1; fi
    sleep 10
done
[ "$READY" = "1" ] || { echo "judge server never ready"; tail -30 "$SRVLOG"; exit 1; }
echo "judge server up after ~$((i*10))s"

# ---- 2. Run steering (traitinterp env), judge = local SGLang Qwen32B via OpenAI shim ----
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
export OPENAI_BASE_URL="http://localhost:$PORT/v1"
export OPENAI_API_KEY="dummy-local-sglang"
export TRAIT_JUDGE_PROVIDER=openai
export TRAIT_JUDGE_MODEL="$JUDGE_MODEL"
# Sampling-mode scoring: read the full integer from generated text. Qwen tokenizes
# "85" as "8"+"5" so the default first-token-logprob scoring truncates it to 8.
export TRAIT_JUDGE_REQUIRE_LOGPROBS=0

echo "---- steering smoke: happy @ layer 49, 2 steps (judge=$JUDGE_MODEL) ----"
python steering/run_steering_eval.py \
    --experiment ant_emotion_concepts \
    --traits "ant_emotion_concepts/happy" \
    --method mean_diff+gm \
    --component residual \
    --position "response[50:]" \
    --no-custom-prompt \
    --layers 49 \
    --search-steps 2 \
    --subset 3 \
    --max-new-tokens 64 \
    --force \
    --load-in-4bit
RC=$?

kill $SERVER_PID 2>/dev/null
echo "==== steering sglang smoke exit=$RC $(date) ===="
exit $RC
