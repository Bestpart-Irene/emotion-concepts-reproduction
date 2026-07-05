#!/bin/bash
# Stage 7 BLACKMAIL emotion-steering sweep (gate already PASSED: 1/10 baseline).
# --blackmail skips the gate + the buggy transcript-probing step and runs the sweep
# directly: 6 emotion vectors (desperate/calm/angry/nervous/happy/sad) × strengths ×
# rollouts on the blackmail scenario, rule-graded (grade_blackmail), NO judge/OpenAI.
# Uses mean_diff+gm vectors at layer 53 (shared.py default patched from bare mean_diff).
# Rollouts halved (25) + 6h walltime so it backfills. Submit:
#   sbatch slurm/stage7_blackmail.sbatch
#SBATCH --job-name=ti_stage7_bm
#SBATCH --partition=gpu,multigpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/%u/traitinterp/logs/%x_%j.out
#SBATCH --error=/scratch/%u/traitinterp/logs/%x_%j.err

set -uo pipefail
CONDA_BASE=/shared/EL9/explorer/miniconda3/25.9.1/miniconda3
ENV_PREFIX=/scratch/$USER/conda_envs/traitinterp
REPO=/scratch/$USER/traitinterp

echo "==== ti_stage7_bm $SLURM_JOB_ID on $(hostname) GPU=$CUDA_VISIBLE_DEVICES $(date) ===="
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
export HF_HOME=/scratch/$USER/hf_cache
[ -f "$HOME/.config/traitinterp/env" ] && source "$HOME/.config/traitinterp/env"

cd "$REPO"
echo "---- Stage 7 blackmail sweep (6 emotions × strengths × 25 rollouts, layer 53, mean_diff+gm) ----"
python experiments/ant_emotion_concepts/scripts/stage7_steering.py \
    --experiment ant_emotion_concepts --blackmail \
    --layer 53 --method mean_diff+gm --rollouts 12 --load-in-4bit
RC=$?
echo "==== stage7 blackmail sweep exit=$RC $(date) ===="
exit $RC
