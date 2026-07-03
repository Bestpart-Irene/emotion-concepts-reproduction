#!/bin/bash
# Step 1 (self-vs-user distinction): two-factor E_user x E_asst readout, forward passes only.
# Cheapest decisive run of the self-vs-user-distinction campaign — no steering, no generation,
# just 60 forward passes on the fresh 70B vectors at L49 (4-bit). Expect a few minutes of GPU,
# like stage5_only (5m20s). Targets `gpu` (8h cap) with a 2h wall for load headroom.
#
# Prereqs (staged by scp, see research/experiments/step1-two-factor-grid/):
#   experiments/ant_emotion_concepts/scripts/run_two_factor_grid.py
#   datasets/inference/ant_emotion_concepts/two_factor_grid_scenarios.json
#   fresh extraction vectors present (full_extraction.sbatch already completed).
#
# Submit from repo root:  sbatch slurm/step1_two_factor_grid.sbatch
#SBATCH --job-name=ti_step1
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G          # 70B 4-bit weights go to GPU (~40G); host-RAM load peak is well under 64G. Trimmed from 128G so we fit nodes whose GPUs are free but RAM is committed (the real scheduling blocker).
#SBATCH --gres=gpu:h200:1
#SBATCH --time=03:00:00    # ~100min slow load + forward passes + margin; shorter than 4h for better backfill priority
#SBATCH --output=/scratch/%u/traitinterp/logs/%x_%j.out
#SBATCH --error=/scratch/%u/traitinterp/logs/%x_%j.err

set -uo pipefail
CONDA_BASE=/shared/EL9/explorer/miniconda3/25.9.1/miniconda3
ENV_PREFIX=/scratch/$USER/conda_envs/traitinterp
REPO=/scratch/$USER/traitinterp

echo "==== ti_step1 $SLURM_JOB_ID on $(hostname) GPU=$CUDA_VISIBLE_DEVICES $(date) ===="
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
export HF_HOME=/scratch/$USER/hf_cache
[ -f "$HOME/.config/traitinterp/env" ] && source "$HOME/.config/traitinterp/env"

cd "$REPO"
EXPDIR="$REPO/experiments/ant_emotion_concepts"
SCEN="$REPO/datasets/inference/ant_emotion_concepts/two_factor_grid_scenarios.json"

if [ ! -d "$EXPDIR/extraction/ant_emotion_concepts" ]; then
    echo "ERROR: fresh vectors not found — run full_extraction.sbatch first." >&2; exit 1
fi
if [ ! -f "$SCEN" ]; then
    echo "ERROR: scenarios not staged at $SCEN" >&2; exit 1
fi

echo "---- Step 1: two-factor grid readout (L49, 4-bit, forward-pass only) ----"
python experiments/ant_emotion_concepts/scripts/run_two_factor_grid.py \
    --experiment ant_emotion_concepts --layers 49 --load-in-4bit \
    --scenarios "$SCEN"

echo "==== step1 done $(date) ===="
echo "Readout JSON: $EXPDIR/results/step1/two_factor_grid.json"
