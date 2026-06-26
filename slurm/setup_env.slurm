#!/bin/bash
# Build the traitinterp conda env on Explorer. Submit from repo root:
#   sbatch slurm/setup_env.slurm
# Mirrors Reinforcement-Casino/setup_env.slurm (conda-on-scratch pattern).
#SBATCH --job-name=ti_setup
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/%u/traitinterp/logs/%x_%j.out
#SBATCH --error=/scratch/%u/traitinterp/logs/%x_%j.err

set -uo pipefail
CONDA_BASE=/shared/EL9/explorer/miniconda3/25.9.1/miniconda3
ENV_PREFIX=/scratch/$USER/conda_envs/traitinterp
REPO=/scratch/$USER/traitinterp

echo "==== ti_setup $SLURM_JOB_ID on $(hostname) $(date) ===="
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Fresh env (remove if rebuilding)
conda deactivate 2>/dev/null || true
rm -rf "$ENV_PREFIX"
conda create --prefix "$ENV_PREFIX" python=3.11 -y
conda activate "$ENV_PREFIX"
echo "Python: $(python --version)"

cd "$REPO"
pip install -e ".[judge,lora]"

echo "==== verify ===="
python - <<'PY'
import torch, transformers
print("torch       :", torch.__version__)
print("transformers:", transformers.__version__)
try:
    import bitsandbytes as bnb; print("bitsandbytes:", bnb.__version__)
except Exception as e:
    print("bitsandbytes: FAIL", e)
try:
    import openai; print("openai      :", openai.__version__)
except Exception as e:
    print("openai      : FAIL", e)
print("CUDA avail  :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU         :", torch.cuda.get_device_name(0))
PY
echo "==== setup done $(date) ===="
