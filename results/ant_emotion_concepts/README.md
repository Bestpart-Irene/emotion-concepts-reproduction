# `ant_emotion_concepts` — our from-scratch results

This directory holds **our own** results, produced from scratch on **Llama 3.3 70B Instruct**
on the Northeastern Explorer cluster, laid out in the [traitinterp](https://github.com/ewernn/traitinterp)
experiment form (`results/<stage>/...` + `config.json` + `findings.md`).

> These are **not** the upstream `emotion-concepts-v1` bundle. The bundle (author results) lives,
> gitignored, under `notebook/data/` and is the baseline we compare against. See `metadata.json`
> for full provenance (Slurm job ids, dates, methods).

## What is genuinely fresh (ours)

| File | What | When |
|---|---|---|
| `cross_trait_normalize_summary.json` | grand-mean + neutral-PC denoise over 15 layers, on freshly-extracted vectors | 2026-06-26 |
| `stage5/dissociation.json` | **headline** — User/Assistant emotion dissociation (paper Fig 10), recomputed on fresh vectors (job `7903504`) | 2026-06-27 |
| `config.json` | model variant + defaults used | — |

The 171 raw emotion vectors behind these (8.9 GB under `extraction/` on the cluster) are **not
committed** — too large, and traitinterp commits derived results, not raw activations.

## What is NOT fresh

Stage 3 geometry, Stage 4 validation, and Stage 7/8/9 were **not** recomputed on our fresh vectors
in this run. Any such numbers you see elsewhere come from the author bundle, not from us. See
`metadata.json → not_regenerated_this_run`. Closing that gap is tracked in
`research/paper-ideas.md` and `research/campaigns/`.

## Headline

Cross-position Pearson **r = 0.7216** (Fig-10 method; all-11-emotion variant 0.6398) — the
user→assistant emotion mirroring **reproduces from an independent extraction**, far from Sonnet's
dissociated r ≈ 0.11. Full write-up in [`findings.md`](findings.md) and the repo-root
[`REPORT.md`](../../REPORT.md).
