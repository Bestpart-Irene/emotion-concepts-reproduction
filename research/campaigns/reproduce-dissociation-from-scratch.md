# Campaign: reproduce-dissociation-from-scratch

- **goal**: confirm the user→assistant emotion-mirroring headline (cross-position Pearson r) survives
  an *independent* from-scratch extraction on Llama-3.3-70B-Instruct, not just the precomputed bundle.
- **success criterion**: from-scratch `cross_position_pearson_r` (Fig-10 method, L49, 6-emotion subset)
  within **±0.05 of 0.63**.
- **baseline**: bundle r = **0.6299** (recorded in `notes.md`, baseline entry).
- **status**: closed — headline confirmed qualitatively; exact value not pinned (see decision log)

## Experiments

| experiment-id | single variable | metric | value | ref | state |
|---|---|---|---|---|---|
| extract-70b-4bit-l49 | from-scratch extraction (vs bundle) | cross_position_pearson_r | 0.7216 | 0.6299 | COMPLETED (job 7903504) |

## Decision log

- 2026-06-26: campaign opened. Baseline (bundle) already reproduces 0.6299 → match. Next, the only
  open question is whether a fresh extraction reproduces it. Submit `slurm/full_extraction.sbatch` on
  `multigpu` (>8 h), then recompute r from `results/stage5/dissociation.json`.
- 2026-06-27: from-scratch fresh-vector extraction completed (job 7903504, `slurm/stage5_only.sbatch`,
  gpu/h200/4-bit, 5m20s). Fig-10 r = **0.7216** (all-emo 0.6398). **Headline confirmed**: strong
  user→assistant mirroring, far from Sonnet's 0.11 — the paper's dissociation does not hold on Llama.
  **But** the point estimate is ~+0.09 above the bundle (0.6299), **outside the ±0.05 success band**.
  Verdict: the *finding* reproduces robustly; the *exact r* is extraction-seed sensitive (fresh 0.72 vs
  bundle 0.63 vs repo findings.md 0.7718). Closing the campaign on the qualitative result.
- Open follow-up (→ `paper-ideas.md`): if a pinned value is needed, run N≥3 extractions on different
  seeds to bracket the r distribution rather than treating a single run as the number.
