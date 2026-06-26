#!/bin/bash
# Phase 4: Stage 8 base-vs-instruct post-training comparison (Figs 36-39).
# Requires BOTH meta-llama/Llama-3.1-70B (base) and meta-llama/Llama-3.3-70B-Instruct
# to be HF-accessible; config.json in the experiment dir already lists both variants.
# Submit from repo root:  sbatch slurm/stage8_dual_model.sbatch
#SBATCH --job-name=ti_stage8
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/%u/traitinterp/logs/%x_%j.out
#SBATCH --error=/scratch/%u/traitinterp/logs/%x_%j.err

set -uo pipefail
CONDA_BASE=/shared/EL9/explorer/miniconda3/25.9.1/miniconda3
ENV_PREFIX=/scratch/$USER/conda_envs/traitinterp
REPO=/scratch/$USER/traitinterp

echo "==== ti_stage8 $SLURM_JOB_ID on $(hostname) GPU=$CUDA_VISIBLE_DEVICES $(date) ===="
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
export HF_HOME=/scratch/$USER/hf_cache
[ -f "$HOME/.config/traitinterp/env" ] && source "$HOME/.config/traitinterp/env"

cd "$REPO"
python experiments/ant_emotion_concepts/scripts/stage8_post_training.py \
    --experiment ant_emotion_concepts --layers 49 --load-in-4bit

echo "==== stage8 done $(date) ===="
