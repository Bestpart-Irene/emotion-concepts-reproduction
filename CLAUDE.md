# Claude Code project instructions — Emotion Concepts reproduction

This repo reproduces the user→assistant emotion-mirroring headline (cross-position Pearson
**r ≈ 0.63**) from ewern's LessWrong post, on Llama-3.3-70B-Instruct, driven on the Northeastern
**Explorer** Slurm cluster. See `README.md` and `RUNBOOK.md`.

It carries a **lightweight subset** of the [burtenshaw/multiautoresearch](https://github.com/burtenshaw/multiautoresearch)
orchestration, adapted to this project's shape. This is a *bounded reproduction*, not an
open-ended optimization loop, so the subset deliberately **drops** the upstream local-master,
promotion, patch-ledger, and worktree-fleet machinery (those assume a continuous
search that lowers `val_bpb` — there is none here). The fixed baseline is the precomputed
`emotion-concepts-v1` bundle (**r = 0.6299**).

## Before running work

1. Read `DISCIPLINE.md` (single-change / read-before-edit / dedup / leave-a-trail).
2. Prefer the project subagents in `.claude/agents/` — start with `coordinator`
   (`claude --agent coordinator`), which delegates to `planner`, `researcher`, `reporter`.
3. Use `research/templates/` as the canonical experiment and campaign templates.
4. Keep `notebook/data/` read-only — it is the upstream bundle the committed notebook reads.

## Managed runner

The managed runner is **Slurm on Explorer over SSH** (not HF Jobs). Per experiment:

- `scripts/slurm_runner.py submit <sbatch>`       — sbatch on the cluster, prints the job id
- `scripts/slurm_runner.py watch <job_id>`        — poll to terminal state
- `scripts/slurm_runner.py fetch <job_id> --results` — pull logs + result JSONs to `.runtime/fetched/`
- `scripts/parse_metric.py <dissociation.json>`   — recompute the headline Pearson r
- `scripts/record_run.py ...`                     — append one row to `research/results.tsv`

Config via env (defaults in `RUNBOOK.md`): `TI_SSH_HOST`, `TI_REMOTE_REPO`, `TI_LOCAL_FETCH`.

## Hard rules

- exactly one variable changed per run; read the stage script / sbatch before changing a CLI.
- check `research/results.tsv` and `slurm_runner.py status` before submitting — never re-run a stage.
- never claim a match without a recomputed metric from `scripts/parse_metric.py`.
- record every completed run in `research/results.tsv` and `research/notes.md`.
- no secrets in git — tokens live only in `~/.config/traitinterp/env` (chmod 600) on the cluster.
- do not reintroduce the upstream master/promotion/DAG/patch/worktree workflow.

For a longer parent-session kickoff: `claude --agent coordinator`, then point it at the open
campaign in `research/campaigns/`.
