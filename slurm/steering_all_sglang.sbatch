#!/bin/bash
# Full 171-emotion steering sweep with the FREE local judge: each array task
# co-locates the SGLang Qwen2.5-32B judge + the 70B steering on ONE h200 and
# steers a 43-emotion chunk of steering_all_traits.txt (4 tasks). Sampling-mode
# judge (reads full 0-100 numbers). Built from steering_all.sbatch (chunking) +
# steering_smoke_sglang.sbatch (server orchestration). Submit from repo root:
#   sbatch slurm/steering_all_sglang.sbatch
#SBATCH --job-name=ti_steer_sgl_all
#SBATCH --partition=gpu,multigpu
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/%u/traitinterp/logs/%x_%A_%a.out
#SBATCH --error=/scratch/%u/traitinterp/logs/%x_%A_%a.err

set -uo pipefail
CONDA_BASE=/shared/EL9/explorer/miniconda3/25.9.1/miniconda3
ENV_PREFIX=/scratch/$USER/conda_envs/traitinterp
JUDGE_VENV=/scratch/$USER/judge_env
REPO=/scratch/$USER/traitinterp
PORT=$((30000 + SLURM_ARRAY_TASK_ID))   # distinct port per task (in case of same node)
JUDGE_MODEL=Qwen/Qwen2.5-32B-Instruct
SRVLOG=$REPO/logs/sglang_server_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log
CHUNK=43

echo "==== ti_steer_sgl_all A=$SLURM_ARRAY_JOB_ID task=$SLURM_ARRAY_TASK_ID on $(hostname) GPU=$CUDA_VISIBLE_DEVICES $(date) ===="
export HF_HOME=/scratch/$USER/hf_cache
[ -f "$HOME/.config/traitinterp/env" ] && source "$HOME/.config/traitinterp/env"
export NO_PROXY="localhost,127.0.0.1,0.0.0.0"
export no_proxy="localhost,127.0.0.1,0.0.0.0"
export PATH="$HOME/.local/bin:$PATH"
module load cuda/12.8.0 2>/dev/null || true

cd "$REPO"

# This task's 43-emotion slice of the 171-trait list.
START=$(( SLURM_ARRAY_TASK_ID * CHUNK + 1 ))
END=$(( START + CHUNK - 1 ))
TRAITS=$(sed -n "${START},${END}p" steering_all_traits.txt | paste -sd, -)
if [ -z "$TRAITS" ]; then echo "No traits for task $SLURM_ARRAY_TASK_ID — done."; exit 0; fi
echo "task $SLURM_ARRAY_TASK_ID: $(echo "$TRAITS" | tr ',' '\n' | wc -l) traits (lines $START-$END)"

# ---- 1. Launch the SGLang judge server (own venv, same GPU, triton backend) ----
echo "---- launch SGLang judge ($JUDGE_MODEL) on port $PORT ----"
( export PATH="$JUDGE_VENV/bin:$PATH"; exec "$JUDGE_VENV/bin/python" -m sglang.launch_server \
    --model-path "$JUDGE_MODEL" --port $PORT --attention-backend triton \
    --mem-fraction-static 0.5 --context-length 4096 ) > "$SRVLOG" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null' EXIT

echo "---- wait for judge server ready ----"
READY=0
for i in $(seq 1 240); do
    if curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then READY=1; break; fi
    if grep -q "Application startup complete" "$SRVLOG" 2>/dev/null; then
        sleep 3; curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && { READY=1; break; }
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then echo "SERVER DIED:"; tail -30 "$SRVLOG"; exit 1; fi
    sleep 10
done
[ "$READY" = "1" ] || { echo "judge server never ready"; tail -30 "$SRVLOG"; exit 1; }
echo "judge server up after ~$((i*10))s"

# ---- 2. Steering over the chunk (traitinterp env), judge = local SGLang Qwen32B ----
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
export OPENAI_BASE_URL="http://localhost:$PORT/v1"
export OPENAI_API_KEY="dummy-local-sglang"
export TRAIT_JUDGE_PROVIDER=openai
export TRAIT_JUDGE_MODEL="$JUDGE_MODEL"
export TRAIT_JUDGE_REQUIRE_LOGPROBS=0   # sampling mode: read full 0-100 integers

echo "---- steering: 43 emotions, layers 25,31,37,43,49, 5 steps (judge=$JUDGE_MODEL) ----"
python steering/run_steering_eval.py \
    --experiment ant_emotion_concepts \
    --traits "$TRAITS" \
    --method mean_diff+gm \
    --component residual \
    --position "response[50:]" \
    --no-custom-prompt \
    --layers 25,31,37,43,49 \
    --search-steps 5 \
    --subset 5 \
    --max-new-tokens 64 \
    --save-responses best \
    --force \
    --load-in-4bit
RC=$?

kill $SERVER_PID 2>/dev/null
echo "==== ti_steer_sgl_all task $SLURM_ARRAY_TASK_ID exit=$RC $(date) ===="
exit $RC
