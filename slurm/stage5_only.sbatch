#!/bin/bash
# Stage 5 only (layer dynamics / dissociation) on the FRESH from-scratch vectors.
# This is the analysis tail of full_extraction.sbatch broken out as its own job, so the
# headline dissociation r can be recomputed without re-running the >8 h extraction.
#
# Why this script exists: the ad-hoc gpu-short run capped at 2h (= partition MaxTime, zero
# headroom) risks TIMEOUT on a 70B 4-bit load. This targets `gpu` (8h cap) with 5h, leaving
# ample headroom. Requires extraction/ant_emotion_concepts/ to already hold fresh vectors
# (full_extraction.sbatch must have completed).
#
# Submit from repo root:  sbatch slurm/stage5_only.sbatch
#SBATCH --job-name=ti_stage5
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/%u/traitinterp/logs/%x_%j.out
#SBATCH --error=/scratch/%u/traitinterp/logs/%x_%j.err

set -uo pipefail
CONDA_BASE=/shared/EL9/explorer/miniconda3/25.9.1/miniconda3
ENV_PREFIX=/scratch/$USER/conda_envs/traitinterp
REPO=/scratch/$USER/traitinterp

echo "==== ti_stage5 $SLURM_JOB_ID on $(hostname) GPU=$CUDA_VISIBLE_DEVICES $(date) ===="
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
export HF_HOME=/scratch/$USER/hf_cache
[ -f "$HOME/.config/traitinterp/env" ] && source "$HOME/.config/traitinterp/env"

cd "$REPO"
EXPDIR="$REPO/experiments/ant_emotion_concepts"

# Guard: this run is only meaningful on freshly-extracted vectors. Bail loudly if the
# fresh extraction dir is missing (do-not-repeat.md: never silently reuse bundle vectors).
if [ ! -d "$EXPDIR/extraction/ant_emotion_concepts" ]; then
    echo "ERROR: $EXPDIR/extraction/ant_emotion_concepts not found — run full_extraction.sbatch first." >&2
    exit 1
fi
echo "fresh vectors: $EXPDIR/extraction/ant_emotion_concepts"
[ -d "$EXPDIR/extraction_bundle_baseline" ] && echo "bundle baseline preserved at: $EXPDIR/extraction_bundle_baseline"

# Force a clean recompute on the fresh vectors.
echo "---- clearing stale Stage 5 outputs to force recompute ----"
rm -rf "$EXPDIR/results/stage5"

echo "---- Stage 5 (layer dynamics: dissociation / context propagation / binding) L49, 4-bit ----"
python experiments/ant_emotion_concepts/scripts/stage5_layer_dynamics.py \
    --experiment ant_emotion_concepts --layers 49 --load-in-4bit

echo "==== stage5 done $(date) ===="
echo "Headline JSON: $EXPDIR/results/stage5/dissociation.json"
