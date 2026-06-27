# Paper-derived ideas

Candidate single-change experiments derived from the paper / LessWrong post / traitinterp docs.
The `researcher` agent appends here; nothing is a "result" until it has a `results.tsv` row.
Each idea = smallest credible change + what it would tell us + main risk.

| # | Idea (single change) | What it tests | Maps to | Risk |
|---|---|---|---|---|
| 1 | Re-extract vectors from scratch on 70B, recompute headline r | Whether r≈0.63 survives independent extraction (not just the bundle) | `slurm/full_extraction.sbatch` | extraction noise / OOM at 4-bit |
| 2 | Layer sweep for the dissociation r (not just L49) | Whether the user→assistant mirroring is L49-specific or broad | `--layers` arg on stage5 | r may be flat → inconclusive |
| 3 | Stage 8 base (Llama-3.1-70B) vs instruct (3.3-70B) | Whether mirroring is induced by instruct post-training | `slurm/stage8_dual_model.sbatch` | needs base-model HF access |
| 4 | 4-bit vs bf16 extraction at fixed layer | Whether quantization inflates/deflates r | `--load-in-4bit` toggle | bf16 70B may not fit one card |
| 5 | N≥3 extractions on different seeds, bracket the r distribution | Pin the from-scratch r (one run gave 0.72 vs bundle 0.63 vs findings.md 0.77 — seed-sensitive) | re-run `full_extraction.sbatch` with `--seed` varied | each extraction is ~6-8 GPU-h |

> Keep the table to live candidates. Move anything ruled out to `do-not-repeat.md` with the reason.
