#!/bin/bash
# Stage 7 DECISION GATE: 10 baseline (un-steered) rollouts on the blackmail +
# reward-hacking scenarios, rule-graded — does Llama-3.3-70B ever misbehave at
# baseline? If never, the full emotion-steering-of-behavior sweep is a null result
# and is skipped. No LLM judge / no OpenAI (grading is keyword-based). Just the 70B.
#   sbatch slurm/stage7_gate.sbatch
#SBATCH --job-name=ti_stage7_gate
#SBATCH --partition=sharing,gpu,multigpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/%u/traitinterp/logs/%x_%j.out
#SBATCH --error=/scratch/%u/traitinterp/logs/%x_%j.err

set -uo pipefail
CONDA_BASE=/shared/EL9/explorer/miniconda3/25.9.1/miniconda3
ENV_PREFIX=/scratch/$USER/conda_envs/traitinterp
REPO=/scratch/$USER/traitinterp

echo "==== ti_stage7_gate $SLURM_JOB_ID on $(hostname) GPU=$CUDA_VISIBLE_DEVICES $(date) ===="
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
export HF_HOME=/scratch/$USER/hf_cache
[ -f "$HOME/.config/traitinterp/env" ] && source "$HOME/.config/traitinterp/env"

cd "$REPO"
echo "---- Stage 7.0 decision gate (blackmail + reward-hacking baseline rollouts) ----"
python experiments/ant_emotion_concepts/scripts/stage7_steering.py \
    --experiment ant_emotion_concepts --gate-only --load-in-4bit
RC=$?
echo "==== stage7 gate exit=$RC $(date) ===="
exit $RC
