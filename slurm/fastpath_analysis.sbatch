#!/bin/bash
# Phase 2 fast path: validate the precomputed bundle reproduces the figures.
# Assumes the emotion-concepts-v1 bundle is already unpacked into experiments/.
# Submit from repo root:  sbatch slurm/fastpath_analysis.sbatch
#SBATCH --job-name=ti_fastpath
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/%u/traitinterp/logs/%x_%j.out
#SBATCH --error=/scratch/%u/traitinterp/logs/%x_%j.err

set -uo pipefail
CONDA_BASE=/shared/EL9/explorer/miniconda3/25.9.1/miniconda3
ENV_PREFIX=/scratch/$USER/conda_envs/traitinterp
REPO=/scratch/$USER/traitinterp

echo "==== ti_fastpath $SLURM_JOB_ID on $(hostname) GPU=$CUDA_VISIBLE_DEVICES $(date) ===="
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

# Secrets + caches (kept out of git)
export HF_HOME=/scratch/$USER/hf_cache
[ -f "$HOME/.config/traitinterp/env" ] && source "$HOME/.config/traitinterp/env"

cd "$REPO"

echo "---- Stage 3 (geometry) ----"
bash experiments/ant_emotion_concepts/scripts/run_stage3.sh

echo "---- Stage 4 (validation: logit lens, implicit emotion, numerical intensity) ----"
python experiments/ant_emotion_concepts/scripts/stage4_validation.py \
    --experiment ant_emotion_concepts --layers 49 --load-in-4bit

echo "---- Stage 5 (layer dynamics: dissociation / context propagation / binding) ----"
python experiments/ant_emotion_concepts/scripts/stage5_layer_dynamics.py \
    --experiment ant_emotion_concepts --layers 49 --load-in-4bit

echo "==== fastpath done $(date) ===="
echo "Results: $REPO/experiments/ant_emotion_concepts/results/  (JSON + PNG per stage)"
