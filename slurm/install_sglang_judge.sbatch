#!/bin/bash
# One-time: create an ISOLATED uv venv with SGLang for the local judge server.
# Uses uv (not conda) per project preference; kept separate from the traitinterp
# env so it pulls its own torch/flashinfer WITHOUT touching the cu126 torch that
# the 70B steering depends on. CPU-only install (pip downloads wheels). Submit:
#   sbatch slurm/install_sglang_judge.sbatch
#SBATCH --job-name=ti_sglang_install
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --output=/scratch/%u/traitinterp/logs/%x_%j.out
#SBATCH --error=/scratch/%u/traitinterp/logs/%x_%j.err

set -uo pipefail
JUDGE_VENV=/scratch/$USER/judge_env          # uv venv for the SGLang judge server
UV_BIN="$HOME/.local/bin/uv"
# Keep the uv cache on the SAME filesystem (/scratch) as the venv so uv can
# HARDLINK packages instead of a slow cross-FS full copy. The default cache
# (~/.cache/uv on /home) is a different FS from /scratch → copy fallback →
# torch takes forever and can leave a half-copied, broken install.
export UV_CACHE_DIR=/scratch/$USER/.uv_cache
export UV_LINK_MODE=hardlink

echo "==== sglang install (uv) $SLURM_JOB_ID on $(hostname) $(date) ===="

# 1. Bootstrap uv if missing (self-contained standalone installer).
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$UV_BIN" ]; then
    echo "---- installing uv ----"
    curl -LsSf https://astral.sh/uv/install.sh | sh || { echo "uv install FAILED"; exit 1; }
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version || { echo "uv not on PATH after install"; exit 1; }

# 2. Fresh isolated venv (py3.11 for sglang support).
uv venv --python 3.11 "$JUDGE_VENV" || { echo "uv venv FAILED"; exit 1; }
source "$JUDGE_VENV/bin/activate"

# 3. SGLang + server extras — uv resolves torch/flashinfer into this venv only.
export UV_HTTP_TIMEOUT=300
uv pip install "sglang[all]>=0.4.0" || { echo "uv pip install sglang FAILED"; exit 1; }

echo "---- verify import ----"
python -c "import sglang; print('sglang', sglang.__version__)" || { echo "IMPORT FAILED"; exit 1; }
python -c "import torch; print('judge-env torch', torch.__version__, 'cuda', torch.version.cuda)"

echo "==== sglang install done $(date) — venv at $JUDGE_VENV ===="
