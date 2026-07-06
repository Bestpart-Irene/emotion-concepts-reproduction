#!/bin/bash
# Stage 4 preference Elo + probe-preference correlation (paper Fig 4, Section 1.4.1):
# forced-choice preference over 64 activities (Elo from pairwise comparisons), and
# whether the emotion PROBE predicts those preferences. This is the CORRELATIONAL
# preference result (does emotion representation relate to preference), NOT causal
# emotion-steering of preference (that's a pipeline gap to build separately).
# No LLM judge. Submit:  sbatch slurm/stage4_preference.sbatch
#SBATCH --job-name=ti_stage4_pref
#SBATCH --partition=gpu,multigpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/%u/traitinterp/logs/%x_%j.out
#SBATCH --error=/scratch/%u/traitinterp/logs/%x_%j.err

set -uo pipefail
CONDA_BASE=/shared/EL9/explorer/miniconda3/25.9.1/miniconda3
ENV_PREFIX=/scratch/$USER/conda_envs/traitinterp
REPO=/scratch/$USER/traitinterp

echo "==== ti_stage4_pref $SLURM_JOB_ID on $(hostname) GPU=$CUDA_VISIBLE_DEVICES $(date) ===="
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
export HF_HOME=/scratch/$USER/hf_cache
[ -f "$HOME/.config/traitinterp/env" ] && source "$HOME/.config/traitinterp/env"

cd "$REPO"
echo "---- Stage 4 preference Elo (64 activities, ~4032 pairs) + probe-preference correlation ----"
python experiments/ant_emotion_concepts/scripts/stage4_validation.py \
    --experiment ant_emotion_concepts --layer 49 --only preference --load-in-4bit
RC=$?
echo "==== stage4 preference exit=$RC $(date) ===="
exit $RC
