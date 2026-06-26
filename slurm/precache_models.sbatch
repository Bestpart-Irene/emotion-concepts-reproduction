#!/bin/bash
# Pre-download the two gated 70B checkpoints into HF cache on a CPU node, so the
# GPU analysis jobs don't burn GPU time pulling ~280GB. Submit from repo root:
#   sbatch slurm/precache_models.sbatch
#SBATCH --job-name=ti_precache
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/%u/traitinterp/logs/%x_%j.out
#SBATCH --error=/scratch/%u/traitinterp/logs/%x_%j.err

set -uo pipefail
CONDA_BASE=/shared/EL9/explorer/miniconda3/25.9.1/miniconda3
ENV_PREFIX=/scratch/$USER/conda_envs/traitinterp

echo "==== ti_precache $SLURM_JOB_ID on $(hostname) $(date) ===="
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
export HF_HOME=/scratch/$USER/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=0
[ -f "$HOME/.config/traitinterp/env" ] && source "$HOME/.config/traitinterp/env"

python - <<'PY'
from huggingface_hub import snapshot_download
import os
tok = os.environ.get("HF_TOKEN")
for repo in ["meta-llama/Llama-3.3-70B-Instruct", "meta-llama/Llama-3.1-70B"]:
    print(f"\n==== downloading {repo} ====", flush=True)
    p = snapshot_download(
        repo_id=repo, token=tok,
        allow_patterns=["*.safetensors", "*.json", "*.model", "tokenizer*", "*.txt"],
    )
    print("cached at:", p, flush=True)
print("\nALL MODELS CACHED")
PY
echo "==== precache done $(date) ===="
