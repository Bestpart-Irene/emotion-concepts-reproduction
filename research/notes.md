# Research notes — Emotion Concepts reproduction

Durable lab notebook. One short entry per run or decision, newest at the top.
The append-only metric ledger is `results.tsv`; this file is the *why* behind the numbers.

Format per entry:

```
## <date> · <phase/stage> · <one-line title>
- model / layer / single variable changed
- job_id (Slurm) + log path
- metric = value   (ref = expected)   → match / mismatch
- one-sentence interpretation
```

---

## 2026-06-27 · phase 3 / stage5 · from-scratch extraction reproduces the headline (r=0.7216)
- model Llama-3.3-70B-Instruct · layer 49 · single variable: **fresh from-scratch vectors** (vs bundle)
- job_id 7903504 (`slurm/stage5_only.sbatch`, gpu/h200/4-bit) · COMPLETED in 5m20s (model load 279s + ~40s analysis)
  · log `logs/ti_stage5_7903504.out` · loaded 11/11 emotion vectors from `extraction/` (not bundle)
- metric `cross_position_pearson_r` (Fig-10, 6-emo subset) = **0.7216** (bundle 0.6299; post 0.63) → **qualitative match, point estimate ~+0.09 high**
  · all-11-emotion variant = 0.6398 (bundle 0.6017)
- Interpretation: the user→assistant emotion-mirroring headline **reproduces from an independent
  extraction** — r≈0.72, nowhere near Sonnet's dissociated 0.11, so the paper's dissociation does
  *not* hold on Llama. The exact r is extraction-seed sensitive: fresh vectors land ~0.09 above the
  bundle, between the bundle's 0.63 and the repo `findings.md`'s earlier 0.7718. Direction is robust;
  the precise value is not pinned. Outside the campaign's ±0.05 band → see campaign decision log.
- Aside: stage5 alone runs in ~5 min — the 2h-vs-5h walltime concern was moot; the only cost was queue wait.

## 2026-06-26 · baseline · precomputed bundle reproduces the headline
- model Llama-3.3-70B-Instruct · layer 49 · variable: none (read the `emotion-concepts-v1` bundle as-is)
- job_id: n/a (recomputed locally in the notebook, no GPU)
- metric `cross_position_pearson_r` (Fig-10 method, 6-emotion subset, pooled over 8 Table-3 scenarios) = **0.6299** (ref ≈ 0.63) → **match**
- This is the ground-truth baseline every from-scratch run is compared against. Sonnet's r≈0.11 is quoted from the paper, not recomputed.
