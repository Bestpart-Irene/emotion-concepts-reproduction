# Experiment discipline (lightweight multiautoresearch)

Ports a **subset** of [burtenshaw/multiautoresearch](https://github.com/burtenshaw/multiautoresearch),
adapted to this bounded reproduction (see `CLAUDE.md`): the `research/` lab notebook, read-only
`planner`/`researcher`/`reporter` subagents under `.claude/agents/`, and a Slurm-aware managed
runner under `scripts/`. It deliberately **drops** the upstream local-master / promotion /
patch-ledger / worktree-fleet machinery — there is no `val_bpb` to keep lowering and no master to
promote here. One goal: reproduce the LessWrong post's numbers reliably, traceably, and without
wasting GPU.

1. **Read-only planning**: before changing anything, read the official
   `docs/replicate_ant_emotion_concepts.md` and the stage scripts — never guess the CLI.
2. **Single change**: each sbatch changes exactly one variable (model / layer / replication-level /
   quantization). No multi-change patches.
3. **Master baseline**: treat the precomputed `emotion-concepts-v1` bundle results as the
   ground-truth baseline; compare full-extraction results against it.
4. **Dedup**: check `squeue -u $USER` before submitting; never re-run the same stage twice.
5. **Leave a trail**: after each run, append one row to `research/results.tsv` (via
   `scripts/record_run.py`) and one entry to `research/notes.md`. Turn avoidable failures into
   `research/do-not-repeat.md` entries.
6. **No secrets in git**: tokens live only in `~/.config/traitinterp/env` (chmod 600).

## research/results.tsv columns
```
job_id    phase    stage    model    layer    metric    value    ref_value    state    notes
```

## Managed-runner loop (one experiment)
```
scripts/slurm_runner.py submit <sbatch>            # -> job_id
scripts/slurm_runner.py watch  <job_id>            # -> terminal state
scripts/slurm_runner.py fetch  <job_id> --results  # -> .runtime/fetched/<job_id>/
scripts/parse_metric.py .runtime/fetched/<job_id>/dissociation.json \
  | scripts/record_run.py --job-id <job_id> --phase <p> --stage <s> --model <m> --state <STATE> --notes "..."
```
