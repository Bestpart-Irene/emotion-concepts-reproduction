#!/bin/bash
# Stage 7 behavioral causal steering (gate PASSED on Llama: 1/10 blackmail, 2/10 RH).
# Full sweep: emotion vectors x strengths x rollouts on the blackmail + reward-hacking
# scenarios, rule-graded (grade_blackmail / RH check) — NO LLM judge / no OpenAI.
# Tests the paper's core causal claim: does steering an emotion (desperate/fear/...)
# change the rate of misaligned behavior? Heavy (2048-token rollouts). Submit:
#   sbatch slurm/stage7_behavior.sbatch
#SBATCH --job-name=ti_stage7_behav
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/%u/traitinterp/logs/%x_%j.out
#SBATCH --error=/scratch/%u/traitinterp/logs/%x_%j.err

set -uo pipefail
CONDA_BASE=/shared/EL9/explorer/miniconda3/25.9.1/miniconda3
ENV_PREFIX=/scratch/$USER/conda_envs/traitinterp
REPO=/scratch/$USER/traitinterp

echo "==== ti_stage7_behav $SLURM_JOB_ID on $(hostname) GPU=$CUDA_VISIBLE_DEVICES $(date) ===="
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
export HF_HOME=/scratch/$USER/hf_cache
[ -f "$HOME/.config/traitinterp/env" ] && source "$HOME/.config/traitinterp/env"

cd "$REPO"
echo "---- Stage 7 blackmail + reward-hacking emotion-steering sweep ----"
python experiments/ant_emotion_concepts/scripts/stage7_steering.py \
    --experiment ant_emotion_concepts --all --load-in-4bit
RC=$?
echo "==== stage7 behavioral sweep exit=$RC $(date) ===="
exit $RC
