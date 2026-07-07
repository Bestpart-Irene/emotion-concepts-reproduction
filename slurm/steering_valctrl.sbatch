#!/bin/bash
# Steering VALIDATION run — three controls on the 11 emotions we claim an effect
# for (8 strong + 3 moderate), all under ONE new prompt_set "valctrl" with NEUTRAL
# dry prompts so baseline emotion expression is low (fixes the baseline-saturation
# that inflated the original 43/171). results.jsonl is keyed by (trait,position,
# prompt_set), NOT method, and append_run only appends — so the three method-runs
# coexist in .../valctrl/results.jsonl and are separated by config.method at parse:
#   A) method mean_diff+gm            -> REAL reference (this is also the #2 result:
#                                        real vector on neutral prompts, true headroom)
#   B) method shufperm                -> #1 NEGATIVE CONTROL: coordinate-permuted
#                                        vector (norm preserved exactly, direction
#                                        destroyed). If trait still rises == artifact.
#   C) method nmcopy  + --norm-match  -> #5 norm-matched steering (scale to residual
#                                        norm) — should bite harder at late layers.
# Prep (embedded python) builds shufperm/ + nmcopy/ vector dirs and steering/valctrl.json.
# Judge = local SGLang Qwen2.5-32B, sampling-mode, co-located on one h200 (as before).
# Submit:  sbatch slurm/steering_valctrl.sbatch
#SBATCH --job-name=ti_steer_valctrl
#SBATCH --partition=sharing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=01:00:00
# NOTE: on `sharing` (separate pool, free h200 d4077, 1h cap) to jump the multigpu
# reservation starvation. 11 emotions x 3 conditions fits ~23min incl judge load.
# Conditions run A->B->C so the two critical controls (#2 baseline, #1 negative)
# finish first even if C (norm-match) were ever cut by the 1h wall.
#SBATCH --output=/scratch/%u/traitinterp/logs/%x_%j.out
#SBATCH --error=/scratch/%u/traitinterp/logs/%x_%j.err
set -uo pipefail
CONDA_BASE=/shared/EL9/explorer/miniconda3/25.9.1/miniconda3
ENV_PREFIX=/scratch/$USER/conda_envs/traitinterp
JUDGE_VENV=/scratch/$USER/judge_env
REPO=/scratch/$USER/traitinterp
PORT=31500
JUDGE_MODEL=Qwen/Qwen2.5-32B-Instruct
SRVLOG=$REPO/logs/sglang_server_valctrl_${SLURM_JOB_ID}.log
TRAITS="ant_emotion_concepts/stimulated,ant_emotion_concepts/puzzled,ant_emotion_concepts/amused,ant_emotion_concepts/fulfilled,ant_emotion_concepts/alert,ant_emotion_concepts/sentimental,ant_emotion_concepts/exuberant,ant_emotion_concepts/playful,ant_emotion_concepts/euphoric,ant_emotion_concepts/thrilled,ant_emotion_concepts/aroused"
LAYERS="25,31,37,43,49"

echo "==== ti_steer_valctrl $SLURM_JOB_ID on $(hostname) GPU=$CUDA_VISIBLE_DEVICES $(date) ===="
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
export HF_HOME=/scratch/$USER/hf_cache
[ -f "$HOME/.config/traitinterp/env" ] && source "$HOME/.config/traitinterp/env"
export NO_PROXY="localhost,127.0.0.1,0.0.0.0"; export no_proxy="localhost,127.0.0.1,0.0.0.0"
export PATH="$HOME/.local/bin:$PATH"
module load cuda/12.8.0 2>/dev/null || true
cd "$REPO"

# ---- 0. Prep: shufperm (permuted, norm-preserving) + nmcopy (exact copy) vectors + neutral prompts ----
echo "---- prep: build shufperm/nmcopy vectors + valctrl.json ----"
python3 - <<'PYEOF'
import torch, os, json
EMOS="stimulated puzzled amused fulfilled alert sentimental exuberant playful euphoric thrilled aroused".split()
LAYERS=[25,31,37,43,49]
tmpl="experiments/ant_emotion_concepts/extraction/ant_emotion_concepts/{e}/instruct/vectors/response_50_/residual/{m}"
g=torch.Generator().manual_seed(20260706)
miss=[]
for e in EMOS:
    src=tmpl.format(e=e,m="mean_diff+gm")
    for m in ("shufperm","nmcopy"): os.makedirs(tmpl.format(e=e,m=m),exist_ok=True)
    for L in LAYERS:
        p=f"{src}/layer{L}.pt"
        if not os.path.exists(p): miss.append(p); continue
        v=torch.load(p, weights_only=True)
        torch.save(v.clone().contiguous(), tmpl.format(e=e,m="nmcopy")+f"/layer{L}.pt")
        perm=torch.randperm(v.numel(), generator=g)
        torch.save(v[perm].clone().contiguous(), tmpl.format(e=e,m="shufperm")+f"/layer{L}.pt")
assert not miss, f"missing source vectors: {miss}"
os.makedirs("steering", exist_ok=True)
json.dump({"questions":[
 "What is the boiling point of water at sea level, and briefly why?",
 "Describe the steps to change a flat car tire.",
 "How does a bicycle's gear system change mechanical advantage?",
 "What is the difference between the HTTP and HTTPS protocols?",
 "Explain, step by step, how photosynthesis converts sunlight into chemical energy."
]}, open("steering/valctrl.json","w"), indent=2)
print("PREP OK: 11 emotions x 5 layers -> shufperm + nmcopy; steering/valctrl.json written")
PYEOF
[ $? -eq 0 ] || { echo "PREP FAILED"; exit 1; }

# ---- 1. Launch SGLang judge server (own venv, same GPU, triton backend) ----
echo "---- launch SGLang judge ($JUDGE_MODEL) on port $PORT ----"
( export PATH="$JUDGE_VENV/bin:$PATH"; exec "$JUDGE_VENV/bin/python" -m sglang.launch_server \
    --model-path "$JUDGE_MODEL" --port $PORT --attention-backend triton \
    --mem-fraction-static 0.5 --context-length 4096 ) > "$SRVLOG" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null' EXIT
READY=0
for i in $(seq 1 240); do
    if curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then READY=1; break; fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then echo "SERVER DIED:"; tail -30 "$SRVLOG"; exit 1; fi
    sleep 10
done
[ "$READY" = "1" ] || { echo "judge server never ready"; tail -30 "$SRVLOG"; exit 1; }
echo "judge server up after ~$((i*10))s"
export OPENAI_API_KEY="dummy-local-sglang"
export TRAIT_JUDGE_PROVIDER=openai
export TRAIT_JUDGE_MODEL="$JUDGE_MODEL"
export TRAIT_JUDGE_REQUIRE_LOGPROBS=0
export OPENAI_BASE_URL="http://localhost:$PORT/v1"

COMMON="--experiment ant_emotion_concepts --traits $TRAITS --component residual --position response[50:] --no-custom-prompt --layers $LAYERS --search-steps 5 --subset 5 --max-new-tokens 64 --questions-file steering/valctrl.json --save-responses best --force --load-in-4bit"
RC=0

echo "==== [A] REAL reference (method mean_diff+gm, neutral prompts) = #2 ===="
python steering/run_steering_eval.py $COMMON --method mean_diff+gm ;  A=$?; echo "[A] exit=$A"; [ $A -ne 0 ] && RC=1

echo "==== [B] NEGATIVE CONTROL (method shufperm, permuted vectors) = #1 ===="
python steering/run_steering_eval.py $COMMON --method shufperm ;      B=$?; echo "[B] exit=$B"; [ $B -ne 0 ] && RC=1

echo "==== [C] NORM-MATCHED (method nmcopy --norm-match) = #5 ===="
python steering/run_steering_eval.py $COMMON --method nmcopy --norm-match ; C=$?; echo "[C] exit=$C"; [ $C -ne 0 ] && RC=1

kill $SERVER_PID 2>/dev/null
echo "==== ti_steer_valctrl done A=$A B=$B C=$C RC=$RC $(date) ===="
exit $RC
