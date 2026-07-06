#!/bin/bash
# Dependency job: run the Step 1 two-way ANOVA on the cluster right after the readout job
# (afterok), so the verdict is computed + persisted cluster-side regardless of any local session.
# CPU-only, ~seconds. Submit with:
#   sbatch --dependency=afterok:<READOUT_JOBID> slurm/step1_analyze_dep.sbatch
#SBATCH --job-name=ti_step1_analyze
#SBATCH --partition=sharing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/%u/traitinterp/logs/%x_%j.out
#SBATCH --error=/scratch/%u/traitinterp/logs/%x_%j.err

set -uo pipefail
CONDA_BASE=/shared/EL9/explorer/miniconda3/25.9.1/miniconda3
ENV_PREFIX=/scratch/$USER/conda_envs/traitinterp
REPO=/scratch/$USER/traitinterp
STEP1="$REPO/experiments/ant_emotion_concepts/results/step1"

echo "==== ti_step1_analyze $SLURM_JOB_ID on $(hostname) $(date) ===="
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
cd "$REPO"

RES="$STEP1/two_factor_grid.json"
if [ ! -f "$RES" ]; then
    echo "ERROR: readout JSON not found at $RES — did the readout job succeed?" >&2; exit 1
fi

echo "---- Step 1 two-way ANOVA (L49) ----"
python experiments/ant_emotion_concepts/scripts/analyze_grid.py "$RES" --layer 49 \
    | tee "$STEP1/verdict.txt"

echo "==== analyze done $(date) ===="
echo "Verdict saved: $STEP1/verdict.txt"
