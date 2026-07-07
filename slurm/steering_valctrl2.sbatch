#!/bin/bash
# Steering VALIDATION job 2 — the clean #1 negative control that job 1 lost.
# Job 1 finding: --force TRUNCATES results.jsonl per run, and all 3 conditions
# shared prompt_set=valctrl, so nmcopy (last) overwrote real+shuf. Also the dry
# technical prompts broke coherence for EVERY method, so the control couldn't
# discriminate. Fixes here:
#   (a) each condition gets its OWN prompt_set (own questions-file stem) -> own
#       results.jsonl, no clobber.
#   (b) OPEN-ENDED, affect-NEUTRAL prompts (emotional coloring is possible, but
#       baseline isn't pre-loaded positive) so REAL can actually steer and SHUF
#       is a fair contrast.
# Two conditions on the 11 strong+moderate emotions (vectors already built in job 1):
#   REAL) method mean_diff+gm  --questions-file cmix_real.json  (prompt_set=cmix_real)
#   SHUF) method shufperm      --questions-file cmix_shuf.json  (prompt_set=cmix_shuf)
# #1 answered by: REAL Delta vs SHUF Delta on identical prompts. If SHUF<<REAL the
# effect is direction-specific (real). (#2 baseline-drop and #5 norm-match already
# answered by job 1.)  Same Qwen-32B judge, co-located on one sharing h200.
# Submit:  sbatch slurm/steering_valctrl2.sbatch
#SBATCH --job-name=ti_steer_valctrl2
#SBATCH --partition=sharing
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
PORT=31600
JUDGE_MODEL=Qwen/Qwen2.5-32B-Instruct
SRVLOG=$REPO/logs/sglang_server_valctrl2_${SLURM_JOB_ID}.log
TRAITS="ant_emotion_concepts/stimulated,ant_emotion_concepts/puzzled,ant_emotion_concepts/amused,ant_emotion_concepts/fulfilled,ant_emotion_concepts/alert,ant_emotion_concepts/sentimental,ant_emotion_concepts/exuberant,ant_emotion_concepts/playful,ant_emotion_concepts/euphoric,ant_emotion_concepts/thrilled,ant_emotion_concepts/aroused"
LAYERS="25,31,37,43,49"

echo "==== ti_steer_valctrl2 $SLURM_JOB_ID on $(hostname) GPU=$CUDA_VISIBLE_DEVICES $(date) ===="
source "$CONDA_BASE/etc/profile.d/conda.sh"; conda activate "$ENV_PREFIX"
export HF_HOME=/scratch/$USER/hf_cache
[ -f "$HOME/.config/traitinterp/env" ] && source "$HOME/.config/traitinterp/env"
export NO_PROXY="localhost,127.0.0.1,0.0.0.0"; export no_proxy="localhost,127.0.0.1,0.0.0.0"
export PATH="$HOME/.local/bin:$PATH"
module load cuda/12.8.0 2>/dev/null || true
cd "$REPO"

# ---- 0. Prep: ensure shufperm vectors exist (idempotent, same seed) + write two identical open-ended neutral prompt files ----
python3 - <<'PYEOF'
import torch, os, json
EMOS="stimulated puzzled amused fulfilled alert sentimental exuberant playful euphoric thrilled aroused".split()
LAYERS=[25,31,37,43,49]
tmpl="experiments/ant_emotion_concepts/extraction/ant_emotion_concepts/{e}/instruct/vectors/response_50_/residual/{m}"
g=torch.Generator().manual_seed(20260706)
for e in EMOS:
    src=tmpl.format(e=e,m="mean_diff+gm"); os.makedirs(tmpl.format(e=e,m="shufperm"),exist_ok=True)
    for L in LAYERS:
        v=torch.load(f"{src}/layer{L}.pt", weights_only=True)
        perm=torch.randperm(v.numel(), generator=g)
        torch.save(v[perm].clone().contiguous(), tmpl.format(e=e,m="shufperm")+f"/layer{L}.pt")
Q=["Describe what you notice when you look out of a window right now.",
   "Tell me, in a few sentences, about how your day has gone so far.",
   "Walk me through how you might spend a free afternoon.",
   "Describe an ordinary meal and the setting you'd eat it in.",
   "What goes through your mind when you think about the coming weekend?"]
os.makedirs("steering", exist_ok=True)
for name in ("cmix_real","cmix_shuf"):
    json.dump({"questions":Q}, open(f"steering/{name}.json","w"), indent=2)
print("PREP OK: shufperm ensured; cmix_real.json + cmix_shuf.json written (identical open-ended neutral prompts)")
PYEOF
[ $? -eq 0 ] || { echo "PREP FAILED"; exit 1; }

# ---- 1. Launch SGLang judge ----
echo "---- launch SGLang judge ($JUDGE_MODEL) on port $PORT ----"
( export PATH="$JUDGE_VENV/bin:$PATH"; exec "$JUDGE_VENV/bin/python" -m sglang.launch_server \
    --model-path "$JUDGE_MODEL" --port $PORT --attention-backend triton \
    --mem-fraction-static 0.5 --context-length 4096 ) > "$SRVLOG" 2>&1 &
SERVER_PID=$!; trap 'kill $SERVER_PID 2>/dev/null' EXIT
READY=0
for i in $(seq 1 240); do
    curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && { READY=1; break; }
    kill -0 $SERVER_PID 2>/dev/null || { echo "SERVER DIED:"; tail -30 "$SRVLOG"; exit 1; }
    sleep 10
done
[ "$READY" = "1" ] || { echo "judge never ready"; tail -30 "$SRVLOG"; exit 1; }
echo "judge up after ~$((i*10))s"
export OPENAI_API_KEY="dummy-local-sglang"; export TRAIT_JUDGE_PROVIDER=openai
export TRAIT_JUDGE_MODEL="$JUDGE_MODEL"; export TRAIT_JUDGE_REQUIRE_LOGPROBS=0
export OPENAI_BASE_URL="http://localhost:$PORT/v1"

COMMON="--experiment ant_emotion_concepts --traits $TRAITS --component residual --position response[50:] --no-custom-prompt --layers $LAYERS --search-steps 5 --subset 5 --max-new-tokens 64 --save-responses best --force --load-in-4bit"
RC=0
echo "==== [REAL] method mean_diff+gm, prompt_set=cmix_real ===="
python steering/run_steering_eval.py $COMMON --method mean_diff+gm --questions-file steering/cmix_real.json; A=$?; echo "[REAL] exit=$A"; [ $A -ne 0 ] && RC=1
echo "==== [SHUF] method shufperm, prompt_set=cmix_shuf (negative control) ===="
python steering/run_steering_eval.py $COMMON --method shufperm --questions-file steering/cmix_shuf.json; B=$?; echo "[SHUF] exit=$B"; [ $B -ne 0 ] && RC=1

kill $SERVER_PID 2>/dev/null
echo "==== ti_steer_valctrl2 done REAL=$A SHUF=$B RC=$RC $(date) ===="
exit $RC
