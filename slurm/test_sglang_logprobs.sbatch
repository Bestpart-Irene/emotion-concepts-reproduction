#!/bin/bash
# De-risk STEP 2.5: verify SGLang serving Qwen2.5-32B returns OpenAI-style
# top_logprobs (the judge scorer needs them; there is NO sampling fallback).
# Serves the model on one a100, curls /v1/chat/completions with logprobs, prints
# the response, then shuts the server down. Cheap: no 70B, no steering. Submit:
#   sbatch slurm/test_sglang_logprobs.sbatch
#SBATCH --job-name=ti_sglang_lptest
#SBATCH --partition=sharing
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:55:00
#SBATCH --output=/scratch/%u/traitinterp/logs/%x_%j.out
#SBATCH --error=/scratch/%u/traitinterp/logs/%x_%j.err

set -uo pipefail
JUDGE_VENV=/scratch/$USER/judge_env
export HF_HOME=/scratch/$USER/hf_cache
export PATH="$HOME/.local/bin:$PATH"
# Explorer compute nodes export http_proxy/https_proxy (Squid at xfer-00). Without
# this, sglang's internal warmup AND the judge's client both route localhost through
# the proxy → 403 Access Denied. Bypass the proxy for all local addresses.
export NO_PROXY="localhost,127.0.0.1,0.0.0.0"
export no_proxy="localhost,127.0.0.1,0.0.0.0"
PORT=30000
MODEL=Qwen/Qwen2.5-32B-Instruct

echo "==== sglang logprobs test $SLURM_JOB_ID on $(hostname) GPU=$CUDA_VISIBLE_DEVICES $(date) ===="
source "$JUDGE_VENV/bin/activate"

echo "---- launch sglang server ($MODEL) ----"
python -m sglang.launch_server --model-path "$MODEL" --port $PORT \
    --mem-fraction-static 0.85 --context-length 4096 > /scratch/$USER/traitinterp/logs/sglang_server_$SLURM_JOB_ID.log 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null' EXIT

echo "---- wait for server ready (Qwen32B cold load from scratch is ~14 min) ----"
READY=0
for i in $(seq 1 240); do   # 240 x 10s = 40 min ceiling; server usually ready ~14 min
    if curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then READY=1; break; fi
    # also break early if the server log says it's up (belt-and-suspenders)
    if grep -q "Application startup complete" /scratch/$USER/traitinterp/logs/sglang_server_$SLURM_JOB_ID.log 2>/dev/null; then
        sleep 3; curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && { READY=1; break; }
    fi
    sleep 10
done
if [ "$READY" != "1" ]; then
    echo "SERVER NEVER CAME UP — tail server log:"; tail -30 /scratch/$USER/traitinterp/logs/sglang_server_$SLURM_JOB_ID.log; exit 1
fi
echo "server up after ~$((i*10))s"

echo "---- probe: chat/completions with logprobs=true, top_logprobs=20 ----"
curl -s "http://localhost:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"On a 0-100 scale, how happy is this text: 'I love sunny mornings!' Reply with only the number.\"}],\"max_tokens\":1,\"temperature\":0,\"logprobs\":true,\"top_logprobs\":20}" \
    | python -c "import sys,json; d=json.load(sys.stdin); ch=d['choices'][0]; lp=ch.get('logprobs'); print('HAS_LOGPROBS:', bool(lp and lp.get('content'))); print(json.dumps(lp, indent=2)[:1500] if lp else 'NO logprobs field'); print('CONTENT:', ch['message']['content'])"

echo "==== logprobs test done $(date) ===="
