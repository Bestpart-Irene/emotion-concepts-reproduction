#!/bin/bash
# SessionStart hook — auto-surface live Emotion Concepts project state into context
# each time a Claude Code session opens in this repo. Local-only (no SSH, never blocks).
# Complements the durable auto-memory file emotion-concepts-steering-validation.md.
R="$HOME/Emotion Concepts"
cd "$R" 2>/dev/null || exit 0
echo "## Emotion Concepts — live session state ($(date '+%Y-%m-%d %H:%M'))"
echo "- HEAD: $(git log --oneline -1 2>/dev/null || echo '(no git)')"
DIRTY=$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')
echo "- uncommitted: ${DIRTY} file(s)$( [ "$DIRTY" != 0 ] && echo ' — run: git status' )"
echo "- Memory: [[emotion-concepts-steering-validation]]. Steering expression leg VALIDATED — 10/11 candidate"
echo "  emotions direction-specific under a Llama-70B judge (the coarse 32B judge had falsely said ~4)."
echo "- OPEN THREADS: (1) only 11 of 171 emotions re-tested with the good judge — ~160 untested (32 marginal +"
echo "  128 sub-gate); (2) '43/171 valid' still inflated by baseline saturation; (3) norm-match steering breaks"
echo "  coherence — needs a gentler coef schedule; (4) an independent GPT/Claude judge (needs live key) is the"
echo "  gold-standard confirmation still missing."
echo "- Cluster: repo /scratch/\$USER/traitinterp on Explorer; check jobs with: ssh login.explorer.northeastern.edu 'squeue -u \$USER'"
echo "- Ready-made assets (cluster, not in git): shufperm1..5/ + nmcopy/ vector dirs; steering/cmix_*.json,"
echo "  valctrl.json prompts; slurm/steering_{valctrl,valctrl2,rigor}.sbatch. --force truncates results.jsonl;"
echo "  results keyed by (trait,position,prompt_set) NOT method — give each condition its own prompt_set."
