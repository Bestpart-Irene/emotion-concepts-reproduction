# Experiment: <experiment-id>

- **campaign**: <campaign-id>
- **hypothesis**: <one sentence — what changes and what you expect>
- **single variable changed**: <model | layer | replication-level | quantization | partition>
- **baseline / ref**: <e.g. bundle r=0.6299, or prior run job_id>
- **sbatch**: `slurm/<name>.sbatch`  (or exact CLI)
- **submitted**: job_id `<id>` · partition `<gpu|multigpu>` · `<date>`
- **log**: `/scratch/$USER/traitinterp/logs/<job-name>_<id>.out`

## Result

- metric `<name>` = `<value>`  (ref `<ref_value>`) → **match / mismatch / failed**
- state: `<COMPLETED | FAILED | TIMEOUT | CANCELLED>`
- recorded in `results.tsv`: yes/no

## Interpretation

<2-3 sentences. What did this run establish? Does it confirm or break the headline?>

## Follow-ups

<next single-change ideas, or "none — branch closed">
