#!/bin/bash
# FAST delta-verification subset with the free local Qwen judge. 4 LOW-baseline
# emotions (sad/angry/afraid/disgusted — where steering should show a clear positive
# delta, unlike happy whose baseline is already high), 2 layers, small — sized to fit
# sharing's 1h cap so it can grab a free h200 ahead of the full 171 array. Submit:
#   sbatch slurm/steering_subset_sglang.sbatch
#SBATCH --job-name=ti_steer_sgl_sub
#SBATCH --partition=sharing,gpu,multigpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/%u/traitinterp/logs/%x_%j.out
#SBATCH --error=/scratch/%u/traitinterp/logs/%x_%j.err

set -uo pipefail
CONDA_BASE=/shared/EL9/explorer/miniconda3/25.9.1/miniconda3
ENV_PREFIX=/scratch/$USER/conda_envs/traitinterp
JUDGE_VENV=/scratch/$USER/judge_env
REPO=/scratch/$USER/traitinterp
PORT=31000
JUDGE_MODEL=Qwen/Qwen2.5-32B-Instruct
SRVLOG=$REPO/logs/sglang_server_${SLURM_JOB_ID}.log
TRAITS="ant_emotion_concepts/sad,ant_emotion_concepts/angry,ant_emotion_concepts/afraid,ant_emotion_concepts/disgusted"

echo "==== ti_steer_sgl_sub $SLURM_JOB_ID on $(hostname) GPU=$CUDA_VISIBLE_DEVICES $(date) ===="
export HF_HOME=/scratch/$USER/hf_cache
[ -f "$HOME/.config/traitinterp/env" ] && source "$HOME/.config/traitinterp/env"
export NO_PROXY="localhost,127.0.0.1,0.0.0.0"
export no_proxy="localhost,127.0.0.1,0.0.0.0"
export PATH="$HOME/.local/bin:$PATH"
module load cuda/12.8.0 2>/dev/null || true

cd "$REPO"

# ---- 1. Launch the SGLang judge server (own venv, same GPU, triton backend) ----
echo "---- launch SGLang judge ($JUDGE_MODEL) on port $PORT ----"
( export PATH="$JUDGE_VENV/bin:$PATH"; exec "$JUDGE_VENV/bin/python" -m sglang.launch_server \
    --model-path "$JUDGE_MODEL" --port $PORT --attention-backend triton \
    --mem-fraction-static 0.5 --context-length 4096 ) > "$SRVLOG" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null' EXIT

echo "---- wait for judge server ready ----"
READY=0
for i in $(seq 1 200); do
    if curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then READY=1; break; fi
    if grep -q "Application startup complete" "$SRVLOG" 2>/dev/null; then
        sleep 3; curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && { READY=1; break; }
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then echo "SERVER DIED:"; tail -30 "$SRVLOG"; exit 1; fi
    sleep 5
done
[ "$READY" = "1" ] || { echo "judge server never ready"; tail -30 "$SRVLOG"; exit 1; }
echo "judge server up after ~$((i*5))s"

# ---- 2. Steering over the 4 emotions (traitinterp env), judge = local SGLang Qwen32B ----
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
export OPENAI_BASE_URL="http://localhost:$PORT/v1"
export OPENAI_API_KEY="dummy-local-sglang"
export TRAIT_JUDGE_PROVIDER=openai
export TRAIT_JUDGE_MODEL="$JUDGE_MODEL"
export TRAIT_JUDGE_REQUIRE_LOGPROBS=0

echo "---- steering: sad/angry/afraid/disgusted, layers 43,49, 5 steps (judge=$JUDGE_MODEL) ----"
python steering/run_steering_eval.py \
    --experiment ant_emotion_concepts \
    --traits "$TRAITS" \
    --method mean_diff+gm \
    --component residual \
    --position "response[50:]" \
    --no-custom-prompt \
    --layers 43,49 \
    --search-steps 5 \
    --subset 3 \
    --max-new-tokens 64 \
    --save-responses best \
    --force \
    --load-in-4bit
RC=$?

kill $SERVER_PID 2>/dev/null
echo "==== ti_steer_sgl_sub exit=$RC $(date) ===="
exit $RC
