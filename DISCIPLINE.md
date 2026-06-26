# Experiment discipline (lightweight multiautoresearch)

Borrows the discipline of [burtenshaw/multiautoresearch](https://github.com/burtenshaw/multiautoresearch)
(**without** porting the full orchestration framework). One goal: reproduce the LessWrong post's
numbers reliably, traceably, and without wasting GPU.

1. **Read-only planning**: before changing anything, read the official
   `docs/replicate_ant_emotion_concepts.md` and the stage scripts — never guess the CLI.
2. **Single change**: each sbatch changes exactly one variable (model / layer / replication-level /
   quantization). No multi-change patches.
3. **Master baseline**: treat the precomputed `emotion-concepts-v1` bundle results as the
   ground-truth baseline; compare full-extraction results against it.
4. **Dedup**: check `squeue -u $USER` before submitting; never re-run the same stage twice.
5. **Leave a trail**: after each run, append key metrics to `results.tsv`
   (job_id / stage / model / layer / metric / value / state / notes).
6. **No secrets in git**: tokens live only in `~/.config/traitinterp/env` (chmod 600).

## results.tsv columns
```
job_id    phase    stage    model    layer    metric    value    ref_value    state    notes
```
