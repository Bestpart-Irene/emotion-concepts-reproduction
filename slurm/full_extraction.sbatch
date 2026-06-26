#!/bin/bash
# Phase 3: regenerate trait vectors from scratch on Llama 3.3 70B Instruct (~6-8 GPU-hr).
# Submit from repo root:  sbatch slurm/full_extraction.sbatch
#SBATCH --job-name=ti_extract
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=18:00:00
#SBATCH --output=/scratch/%u/traitinterp/logs/%x_%j.out
#SBATCH --error=/scratch/%u/traitinterp/logs/%x_%j.err

set -uo pipefail
CONDA_BASE=/shared/EL9/explorer/miniconda3/25.9.1/miniconda3
ENV_PREFIX=/scratch/$USER/conda_envs/traitinterp
REPO=/scratch/$USER/traitinterp

echo "==== ti_extract $SLURM_JOB_ID on $(hostname) GPU=$CUDA_VISIBLE_DEVICES $(date) ===="
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
export HF_HOME=/scratch/$USER/hf_cache
[ -f "$HOME/.config/traitinterp/env" ] && source "$HOME/.config/traitinterp/env"

cd "$REPO"
EXPDIR="$REPO/experiments/ant_emotion_concepts"

echo "---- preserve author (bundle) vectors before regenerating ----"
if [ -d "$EXPDIR/extraction" ] && [ ! -d "$EXPDIR/extraction_bundle_baseline" ]; then
    mv "$EXPDIR/extraction" "$EXPDIR/extraction_bundle_baseline"
    echo "bundle vectors -> extraction_bundle_baseline"
fi

echo "---- extraction stages 1,3 (171 emotions x 40 stories, 4-bit) ----"
python extraction/run_extraction_pipeline.py \
    --experiment ant_emotion_concepts \
    --category ant_emotion_concepts \
    --only-stage 1,3 --save-activations --load-in-4bit --seed 42

echo "---- cross-trait normalize (grand-mean center + neutral-PC denoise) ----"
python analysis/vectors/cross_trait_normalize.py \
    --experiment ant_emotion_concepts \
    --layers 1,7,13,19,25,31,37,43,49,55,61,67,73,79

echo "==== extraction done $(date) ===="

# --- Clear stale stage outputs (bundle results already backed up to
#     results_bundle_baseline) so stages truly recompute on OUR fresh vectors ---
echo "---- clearing stale stage outputs to force recompute ----"
rm -rf "$EXPDIR/results/stage5" "$EXPDIR/results/stage4_validation" \
       "$EXPDIR/results/geometry" "$EXPDIR/results/stage3_geometry"

# --- Independent re-analysis on our freshly extracted vectors ---
echo "---- Stage 3 (geometry) on fresh vectors ----"
bash experiments/ant_emotion_concepts/scripts/run_stage3.sh
echo "---- Stage 4 (validation) on fresh vectors ----"
python experiments/ant_emotion_concepts/scripts/stage4_validation.py \
    --experiment ant_emotion_concepts --layers 49 --load-in-4bit
echo "---- Stage 5 (layer dynamics / dissociation) on fresh vectors ----"
python experiments/ant_emotion_concepts/scripts/stage5_layer_dynamics.py \
    --experiment ant_emotion_concepts --layers 49 --load-in-4bit

echo "==== full extraction + independent analysis done $(date) ===="
echo "Fresh vectors: $EXPDIR/extraction/   |   Author baseline: $EXPDIR/extraction_bundle_baseline/"
