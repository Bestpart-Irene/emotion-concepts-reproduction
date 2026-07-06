# Program

Reproduce, and now **extend with steering**, ewern's LessWrong replication of Anthropic's
*Emotion Concepts and their Function in a Large Language Model* on **Llama-3.3-70B-Instruct**,
using [traitinterp](https://github.com/ewernn/traitinterp) on the Northeastern **Explorer**
Slurm cluster. (Structure adapted from the `program.md` convention in
[burtenshaw/multiautoresearch](https://github.com/burtenshaw/multiautoresearch).)

## Objectives

1. **Dissociation headline (done).** Confirm the user→assistant emotion-mirroring result:
   cross-position Pearson **r ≈ 0.63** (layer 49, 6-emotion subset, pooled over the 8 Table-3
   scenarios), reproduced from the bundle (r = 0.6299) and an independent from-scratch
   extraction (r = 0.7216, job 7903504). Direction is robust; the point estimate is
   extraction-seed sensitive.
2. **Steering (in progress).** Validate *causally* that the extracted emotion vectors **control**
   emotional expression — the steering section of the post. For each emotion: adaptive
   coefficient search (`up_mult 1.3` / `down_mult 0.85`, `min_coherence 77`, 5 steps) over layers
   25–49 (30–60% depth), scored by the OpenAI TraitJudge. A vector is a **strong** steer when
   `trait_delta ≥ 20` at `coherence ≥ 77`. Report best-per-layer per emotion across all 171
   extracted emotions.

## Fixed / do-not-touch

- `notebook/data/` — the upstream `emotion-concepts-v1` bundle the committed notebook reads.
- The comparability targets: Fig-10 method for the headline r; coherence ≥ 77 for steering validity.
- Do **not** reintroduce the upstream master / promotion / patch-ledger / worktree-fleet
  machinery — this is a bounded reproduction + steering extension, not a continuous val_bpb search.

## Managed runner (Slurm on Explorer over SSH)

Per experiment, smoke-gate then one managed run:

```bash
scripts/slurm_runner.py submit  <sbatch>      # → job id
scripts/slurm_runner.py watch   <job_id>      # → terminal state
scripts/slurm_runner.py fetch   <job_id> --results
scripts/parse_metric.py .runtime/fetched/<job_id>/<result>.json   # recompute the metric
scripts/record_run.py  ...                    # append one row to research/results.tsv
```

Arrays and dependency edits use raw `sbatch --array` / `scontrol update` (the runner has no
array support), but keep the same submit → watch → **verify-real-artifact** → record shape.

### Steering commands

```bash
# smoke (one emotion, one layer, cheap) — gate before the fleet
sbatch --partition=gpu,multigpu --time=02:00:00 slurm/steering_smoke.sbatch

# full sweep — ALL 171 emotions, job array (43/task), held until the smoke verifies
sbatch --dependency=afterok:<smoke_id> slurm/steering_all.sbatch
```

Emotion traits carry only `positive.jsonl`; `steering/run_steering_eval.py`'s `resolve_questions`
hard-requires a per-trait `steering.json`, so `gen_steering_json.py` writes one (generic neutral
questions + `direction: positive`) for every emotion. Vectors are `--method mean_diff+gm`
(not the default `probe`). Use `--no-custom-prompt` (generic judge; no per-trait eval prompt).

## Discipline (see `DISCIPLINE.md`, `CLAUDE.md`)

- exactly one variable per run; read the sbatch before changing a CLI; dedup against `results.tsv`.
- **fail-loud** sbatch (`exit $RC`) and **verify the real artifact**, never trust `sacct` State.
- 70B loads from shared scratch are slow (~5–90 min) — give 70B jobs ≥2–3 h walltime.
- never claim a match without a recomputed metric; record every run in `results.tsv` + `notes.md`.

## Orchestration (Claude Code agents in `.claude/agents/`)

- `coordinator` — parent flow; plans, delegates, owns the managed loop.
- `planner` — small deduplicated queue of single-change experiments.
- `researcher` — scout the paper / post / traitinterp docs for hypotheses (read-only).
- `experiment-worker` — run one single-variable experiment, smoke-gated + verified.
- `reviewer` — read-only integrity check (fake results, comparability, one-variable, smoke evidence).
- `memory-keeper` — maintain `results.tsv`, `notes.md`, `do-not-repeat.md`.
- `reporter` — summarize Slurm job/queue status (read-only, over SSH).
