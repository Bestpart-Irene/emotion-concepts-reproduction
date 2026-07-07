#!/bin/bash
# Steering RIGOROUS validation — pin "how many emotions are genuinely direction-specific".
# Upgrades over the quick controls: (1) BIGGER judge = Llama-3.3-70B served FP8 on its OWN
# 2nd h200 (fallback Qwen-32B if FP8 server fails); (2) 5 random-permutation SEEDS as a null
# distribution (not 1); (3) SAVE ALL responses (--save-responses all) for later re-judging;
# (4) more prompts per condition (subset 8). 11 emotions x 6 conditions (real + 5 seeds),
# each its OWN method dir AND OWN prompt_set (cmix_real / cmix_s1..s5) so --force truncation
# can't clobber. Verdict rule (computed after): per emotion, real Delta vs the {5 seed Deltas}
# null band -> direction-specific iff real clearly exceeds the random null.
# GPU0 = steering 70B-4bit; GPU1 = judge.  Submit:  sbatch slurm/steering_rigor.sbatch
#SBATCH --job-name=ti_steer_rigor
#SBATCH --partition=gpu,multigpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --gres=gpu:h200:2
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/%u/traitinterp/logs/%x_%j.out
#SBATCH --error=/scratch/%u/traitinterp/logs/%x_%j.err
set -uo pipefail
CONDA_BASE=/shared/EL9/explorer/miniconda3/25.9.1/miniconda3
ENV_PREFIX=/scratch/$USER/conda_envs/traitinterp
JUDGE_VENV=/scratch/$USER/judge_env
REPO=/scratch/$USER/traitinterp
PORT=31700
SRVLOG=$REPO/logs/sglang_server_rigor_${SLURM_JOB_ID}.log
TRAITS="ant_emotion_concepts/stimulated,ant_emotion_concepts/puzzled,ant_emotion_concepts/amused,ant_emotion_concepts/fulfilled,ant_emotion_concepts/alert,ant_emotion_concepts/sentimental,ant_emotion_concepts/exuberant,ant_emotion_concepts/playful,ant_emotion_concepts/euphoric,ant_emotion_concepts/thrilled,ant_emotion_concepts/aroused"
LAYERS="25,31,37,43,49"
SEEDS="1 2 3 4 5"

echo "==== ti_steer_rigor $SLURM_JOB_ID on $(hostname) GPUs=$CUDA_VISIBLE_DEVICES $(date) ===="
source "$CONDA_BASE/etc/profile.d/conda.sh"; conda activate "$ENV_PREFIX"
export HF_HOME=/scratch/$USER/hf_cache
[ -f "$HOME/.config/traitinterp/env" ] && source "$HOME/.config/traitinterp/env"
export NO_PROXY="localhost,127.0.0.1,0.0.0.0"; export no_proxy="localhost,127.0.0.1,0.0.0.0"
export PATH="$HOME/.local/bin:$PATH"
module load cuda/12.8.0 2>/dev/null || true
cd "$REPO"

# ---- 0. Prep: 5 seeded permuted vector dirs (shufperm1..5) + 6 identical prompt files ----
python3 - <<'PYEOF'
import torch, os, json
EMOS="stimulated puzzled amused fulfilled alert sentimental exuberant playful euphoric thrilled aroused".split()
LAYERS=[25,31,37,43,49]
tmpl="experiments/ant_emotion_concepts/extraction/ant_emotion_concepts/{e}/instruct/vectors/response_50_/residual/{m}"
for i,e in enumerate(EMOS):
    src=tmpl.format(e=e,m="mean_diff+gm")
    for s in range(1,6):
        m="shufperm%d"%s; os.makedirs(tmpl.format(e=e,m=m),exist_ok=True)
        g=torch.Generator().manual_seed(1000*s+i)   # deterministic per (emotion,seed)
        for L in LAYERS:
            v=torch.load(f"{src}/layer{L}.pt", weights_only=True)
            perm=torch.randperm(v.numel(), generator=g)
            torch.save(v[perm].clone().contiguous(), tmpl.format(e=e,m=m)+f"/layer{L}.pt")
Q=["Describe what you notice when you look out of a window right now.",
   "Tell me, in a few sentences, about how your day has gone so far.",
   "Walk me through how you might spend a free afternoon.",
   "Describe an ordinary meal and the setting you would eat it in.",
   "What goes through your mind when you think about the coming weekend?",
   "Describe the room or space you are in at the moment.",
   "Tell me about a short walk you might take nearby.",
   "How would you describe today's weather and how the day feels?"]
os.makedirs("steering", exist_ok=True)
for name in ["cmix_real","cmix_s1","cmix_s2","cmix_s3","cmix_s4","cmix_s5"]:
    json.dump({"questions":Q}, open(f"steering/{name}.json","w"), indent=2)
print("PREP OK: shufperm1..5 (5 seeds) + 6 prompt files (8 prompts each)")
PYEOF
[ $? -eq 0 ] || { echo "PREP FAILED"; exit 1; }

# ---- 1. Judge server on GPU 1: try Llama-3.3-70B FP8, fallback Qwen2.5-32B ----
JUDGE_MODEL="meta-llama/Llama-3.3-70B-Instruct"
echo "---- launch judge (Llama-3.3-70B FP8) on GPU1 port $PORT ----"
( export CUDA_VISIBLE_DEVICES=1; export PATH="$JUDGE_VENV/bin:$PATH"; exec "$JUDGE_VENV/bin/python" -m sglang.launch_server \
    --model-path "$JUDGE_MODEL" --served-model-name judge --port $PORT --attention-backend triton \
    --quantization fp8 --mem-fraction-static 0.85 --context-length 4096 ) > "$SRVLOG" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null' EXIT
ready(){ for i in $(seq 1 $1); do curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && return 0; kill -0 $SERVER_PID 2>/dev/null || return 2; sleep 10; done; return 1; }
ready 150; RS=$?
if [ $RS -ne 0 ]; then
    echo "!! Llama-70B-FP8 judge failed (rc=$RS); falling back to Qwen2.5-32B on GPU1"; tail -20 "$SRVLOG"
    kill $SERVER_PID 2>/dev/null; sleep 5
    JUDGE_MODEL="Qwen/Qwen2.5-32B-Instruct"
    ( export CUDA_VISIBLE_DEVICES=1; export PATH="$JUDGE_VENV/bin:$PATH"; exec "$JUDGE_VENV/bin/python" -m sglang.launch_server \
        --model-path "$JUDGE_MODEL" --served-model-name judge --port $PORT --attention-backend triton \
        --mem-fraction-static 0.6 --context-length 4096 ) > "$SRVLOG" 2>&1 &
    SERVER_PID=$!
    ready 150 || { echo "fallback judge never ready"; tail -30 "$SRVLOG"; exit 1; }
fi
echo "judge up: $JUDGE_MODEL"
export OPENAI_API_KEY="dummy-local-sglang"; export TRAIT_JUDGE_PROVIDER=openai
export TRAIT_JUDGE_MODEL="judge"; export TRAIT_JUDGE_REQUIRE_LOGPROBS=0
export OPENAI_BASE_URL="http://localhost:$PORT/v1"
echo "JUDGE_USED=$JUDGE_MODEL"   # <-- grep this to know which judge scored the run

# ---- 2. Steering on GPU 0 ----
export CUDA_VISIBLE_DEVICES=0
COMMON="--experiment ant_emotion_concepts --traits $TRAITS --component residual --position response[50:] --no-custom-prompt --layers $LAYERS --search-steps 5 --subset 8 --max-new-tokens 64 --save-responses all --force --load-in-4bit"
RC=0
echo "==== [REAL] method mean_diff+gm, prompt_set=cmix_real ===="
python steering/run_steering_eval.py $COMMON --method mean_diff+gm --questions-file steering/cmix_real.json; r=$?; echo "[REAL] exit=$r"; [ $r -ne 0 ] && RC=1
for s in $SEEDS; do
    echo "==== [SEED $s] method shufperm$s, prompt_set=cmix_s$s ===="
    python steering/run_steering_eval.py $COMMON --method shufperm$s --questions-file steering/cmix_s$s.json; r=$?; echo "[SEED $s] exit=$r"; [ $r -ne 0 ] && RC=1
done

kill $SERVER_PID 2>/dev/null
echo "==== ti_steer_rigor done RC=$RC JUDGE=$JUDGE_MODEL $(date) ===="
exit $RC
