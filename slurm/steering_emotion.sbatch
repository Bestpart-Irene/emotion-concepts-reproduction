#!/bin/bash
# Emotion steering — the LessWrong-post steering section, reproduced on Llama-3.3-70B.
# Batched multi-trait adaptive coefficient search over the paper's ~11-emotion subset,
# validating causally that the extracted emotion vectors *control* emotional expression
# (trait delta vs baseline, at coherence >= 77). Mirror of the coef-search machinery the
# post describes: up_mult 1.3 / down_mult 0.85 / min_coherence 77 / 5 search steps.
# Run AFTER slurm/steering_smoke.sbatch passes. Submit from repo root:
#   sbatch slurm/steering_emotion.sbatch
#SBATCH --job-name=ti_steer_emo
#SBATCH --partition=multigpu
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

echo "==== ti_steer_emo $SLURM_JOB_ID on $(hostname) GPU=$CUDA_VISIBLE_DEVICES $(date) ===="
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

export HF_HOME=/scratch/$USER/hf_cache
[ -f "$HOME/.config/traitinterp/env" ] && source "$HOME/.config/traitinterp/env"

cd "$REPO"

# Paper ~11-emotion subset (all have fresh mean_diff+gm vectors from extraction job 7903504).
TRAITS="ant_emotion_concepts/happy,ant_emotion_concepts/calm,ant_emotion_concepts/loving,ant_emotion_concepts/sad,ant_emotion_concepts/afraid,ant_emotion_concepts/angry,ant_emotion_concepts/desperate,ant_emotion_concepts/nervous,ant_emotion_concepts/disgusted,ant_emotion_concepts/surprised,ant_emotion_concepts/excited"

echo "---- Emotion steering: 11 traits, layers 25,31,37,43,49, 5 search steps ----"
# Layers 25-49 span the 30%-60% depth band (80-layer model) and all have extracted vectors;
# layer 49 is the dissociation headline layer, letting us tie steering back to that result.
python steering/run_steering_eval.py \
    --experiment ant_emotion_concepts \
    --traits "$TRAITS" \
    --method mean_diff+gm \
    --component residual \
    --questions-file datasets/inference/ant_emotion_concepts/basic_steering_prompts.json \
    --no-custom-prompt \
    --layers 25,31,37,43,49 \
    --search-steps 5 \
    --subset 3 \
    --max-new-tokens 64 \
    --save-responses best \
    --load-in-4bit

echo "==== emotion steering done $(date) ===="
echo "Results: $REPO/experiments/ant_emotion_concepts/steering/  (results.jsonl per trait)"
