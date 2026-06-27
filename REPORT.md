# Results Report — Emotion Concepts, from scratch on Llama 3.3 70B Instruct

*2026-06-27 · Northeastern Explorer cluster · traitinterp toolkit*

## TL;DR

We regenerated the emotion vectors **from scratch** on Llama 3.3 70B Instruct and recomputed the
paper's User/Assistant dissociation test. The headline reproduces:

> **Cross-position Pearson r = 0.72** (Fig-10 method) — the assistant position mirrors the user's
> emotion. The paper reports Claude Sonnet 4.5 stays dissociated at r ≈ 0.11. **The paper's emotion
> dissociation does not hold on Llama** — and this survives an independent re-extraction.

The exact value (0.72) runs ~0.09 above the precomputed bundle (0.63); the *direction* is rock-solid,
the *precise number* is extraction-sensitive. We report that honestly rather than rounding to a match.

---

## 1. What we set out to test

Anthropic's *Emotion Concepts and their Function in a Large Language Model* (Sofroniew et al. 2026)
finds that in a chat transcript, a model encodes the **user's** emotion at the user-turn's final `.`
and the **assistant's** emotion at the assistant-turn's `:` — and on **Claude Sonnet 4.5** these two
are essentially **independent** (cross-position correlation r ≈ 0.11). That independence is the
"emotion dissociation."

ewern's [LessWrong post](https://www.lesswrong.com/posts/sJQ62HbA76s3aiuiT) found this dissociation
**breaks on Llama** (r ≈ 0.63 — the assistant mirrors the user). Our goal: confirm that from an
**independent from-scratch extraction**, not just by reading the author's precomputed vectors.

## 2. What we actually ran

| Step | Where | Status |
|---|---|---|
| Extract 171 emotion vectors on Llama 3.3 70B Instruct (4-bit), mean-difference of own activations | Explorer, `multigpu` | ✅ fresh (2026-06-26) |
| Cross-trait normalize (grand-mean center + neutral-PC denoise), 15 layers | Explorer | ✅ fresh |
| **Stage 5 — User/Assistant dissociation (Fig 10), L49** | Explorer `gpu`/h200, job **7903504**, 5m20s | ✅ fresh (2026-06-27) |
| Stage 3 geometry, Stage 4 validation, Stage 7/8/9 | — | ⬜ not regenerated this run |

The single managed run that produced the headline:
`slurm/stage5_only.sbatch` → `stage5_layer_dynamics.py --layers 49 --load-in-4bit`, on the freshly
extracted vectors (11/11 probe emotions loaded from `extraction/`, *not* the bundle). It completed in
**5 min 20 s** (model load 279 s + ~40 s analysis); the rest was queue wait.

## 3. The result

Method: layer 49, the official 6-emotion Fig-10 probe subset (`happy, calm, loving, sad, afraid,
angry`), pooled over the paper's 8 Table-3 scenarios → n = 48 (user-position, assistant-position) pairs.

| Source | r (Fig-10, 6-emo) | r (all 11 emo) | regression slope | reading |
|---|---|---|---|---|
| Claude Sonnet 4.5 (paper) | **0.11** | — | — | dissociated |
| LessWrong post (Llama) | 0.63 | — | — | mirrors user |
| Bundle baseline (author vectors) | 0.6299 | 0.6017 | 0.424 | matches the post |
| **Our from-scratch (job 7903504)** | **0.7216** | **0.6398** | 0.513 | mirrors user, stronger |

**The effect is broad, not an outlier.** Every one of the six probe emotions is individually
positive in both our run and the bundle:

| emotion | our fresh r | bundle r |
|---|---|---|
| happy | 0.48 | 0.50 |
| calm | 0.74 | 0.66 |
| loving | 0.64 | 0.65 |
| sad | 0.67 | 0.66 |
| afraid | 0.48 | 0.53 |
| angry | 0.59 | 0.53 |

(For the figure rendering this scatter — Llama fit vs the Sonnet r = 0.11 reference line — see
`notebook/emotion_concepts_reproduction.ipynb`, Step 3.)

## 4. Interpretation

- **The headline reproduces.** An entirely independent extraction (fresh activations, fresh
  normalization on 70B) gives r ≈ 0.72 — same sign, same magnitude band, ~6.5× the Sonnet reference.
  On Llama 3.3 70B Instruct the assistant's emotion is **strongly predicted by the user's**; the
  dissociation the paper documents on Sonnet **does not hold here**. This is a real cross-model
  contrast, and it is not an artifact of reusing the author's vectors.
- **The exact number is soft.** Three runs of the nominally-identical pipeline span **0.63 → 0.72 →
  0.77** (bundle → ours → repo `findings.md`'s earlier run). Extraction seed + 4-bit quantization
  move the point estimate by ~0.1. Read the result as "strong positive, ≈ 0.6–0.77," not as 0.72
  to four digits. By our pre-registered ±0.05 band around the bundle, 0.72 is **outside** — a clean
  qualitative reproduction with an unpinned magnitude.

## 5. Limitations

- One extraction seed — to bracket the r distribution honestly we'd run N ≥ 3 seeds (paper-idea #5).
- 4-bit (bitsandbytes nf4) extraction; quantization is a known per-emotion noise source.
- Only the Fig-10 dissociation sub-experiment of Stage 5 was saved this run.
- Geometry / validation / post-training stages were **not** regenerated on our vectors — any such
  numbers elsewhere are the author bundle's, not ours.
- Sonnet's r = 0.11 is quoted from the paper, not recomputed.

## 6. What's next

1. **Pin the magnitude** — N ≥ 3 fresh extractions on different seeds; report the r distribution.
2. **Finish the fresh stages** — regenerate Stage 3 geometry + Stage 4 validation on our vectors so
   the whole replication table is ours, not half bundle.
3. **Base vs instruct (Stage 8)** — is the mirroring induced by instruct post-training? (`slurm/stage8_dual_model.sbatch`.)

## Artifacts

- Our results, traitinterp form: [`results/ant_emotion_concepts/`](results/ant_emotion_concepts/)
  (`findings.md`, `metadata.json`, `stage5/dissociation.json`, `cross_trait_normalize_summary.json`)
- Run ledger: [`research/results.tsv`](research/results.tsv) · lab notebook:
  [`research/notes.md`](research/notes.md) · campaign:
  [`research/campaigns/reproduce-dissociation-from-scratch.md`](research/campaigns/reproduce-dissociation-from-scratch.md)
- Figure-embedded walkthrough: [`notebook/emotion_concepts_reproduction.ipynb`](notebook/emotion_concepts_reproduction.ipynb)
