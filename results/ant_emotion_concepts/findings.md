# Emotion Concepts — From-Scratch Findings (Llama 3.3 70B Instruct)

**Status**: Findings document for *our* independent from-scratch run. Mirrors the traitinterp
findings form, scoped to what we actually recomputed on freshly-extracted vectors.

---

## 1. Purpose and scope

Independent replication of the headline result from Sofroniew et al. 2026, *Emotion Concepts and
their Function in a Large Language Model* — specifically the **User/Assistant dissociation (Fig 10)**
— on **Llama 3.3 70B Instruct**, regenerating the emotion vectors from scratch (not reading the
precomputed `emotion-concepts-v1` bundle) and recomputing the cross-position correlation.

**In scope (fresh, ours):** vector extraction (171 emotions) → cross-trait normalization (15 layers)
→ Stage 5 dissociation at L49.
**Out of scope this run:** Stage 3 geometry, Stage 4 validation, Stage 7/8/9 were *not* regenerated
on the fresh vectors — those remain author-bundle numbers and are **not** claimed here.

---

## 2. Headline (Fig 10 dissociation)

The paper's claim: at the **user's "."** position the model encodes the *user's* emotion, and at the
**assistant's ":"** position it encodes the *assistant's* emotion, and on Claude Sonnet 4.5 these stay
**independent** (cross-position Pearson r ≈ 0.11). The LessWrong post found that on Llama the two are
**not** independent (r ≈ 0.63) — the assistant position mirrors the user's emotion.

Our from-scratch run **reproduces and slightly strengthens** that non-dissociation:

| Source | cross-position r (Fig-10, 6-emo) | all-11-emotion r | meaning |
|---|---|---|---|
| Paper, Claude Sonnet 4.5 | **0.11** (from paper, not recomputed) | — | dissociated (independent positions) |
| LessWrong post (Llama) | 0.63 | — | assistant mirrors user |
| Bundle baseline (author vectors, our recompute) | 0.6299 | 0.6017 | matches the post |
| **Our from-scratch (job 7903504)** | **0.7216** | **0.6398** | mirrors user, ~+0.09 above bundle |

- Method: layer 49, 6-emotion probe subset (`happy, calm, loving, sad, afraid, angry`), pooled over
  the 8 Table-3 dissociation scenarios → n = 48 (user_period, assistant_colon) pairs. Regression
  slope (assistant on user) = **0.513** (bundle 0.424).
- The effect is **broad across emotions**, not driven by one. Per-emotion r within the Fig-10 subset:

  | emotion | our fresh r | bundle r |
  |---|---|---|
  | happy | 0.48 | 0.50 |
  | calm | 0.74 | 0.66 |
  | loving | 0.64 | 0.65 |
  | sad | 0.67 | 0.66 |
  | afraid | 0.48 | 0.53 |
  | angry | 0.59 | 0.53 |

---

## 3. Interpretation

- **The qualitative finding is robust.** A wholly independent extraction (fresh mean-difference
  vectors on 70B, fresh normalization) lands at r ≈ 0.72 — same direction, same order of magnitude,
  ~6.5× the Sonnet reference. The user/assistant emotion *dissociation* the paper reports on Sonnet
  **does not hold on Llama 3.3 70B Instruct**. This is the one genuine cross-model contrast, and it
  survives re-derivation.
- **The exact value is extraction-sensitive.** Our fresh r = 0.7216 sits ~0.09 above the bundle's
  0.6299 and below the repo `findings.md`'s earlier 0.7718 — three runs of the "same" pipeline span
  0.63–0.77. A single run should be read as "strong positive, ~0.6–0.77", not as a pinned point
  estimate. This is **outside** our pre-registered ±0.05 success band relative to the bundle, and we
  report it as such rather than rounding it to a match.

---

## 4. Limitations / honest disclosures

- **Single extraction seed.** One fresh extraction. To pin the r distribution, N≥3 seeds are needed
  (tracked as paper-idea #5). We do not claim 0.72 is *the* number.
- **4-bit quantization.** Vectors extracted with bitsandbytes nf4; quantization noise is a known
  contributor to per-emotion variability (see the upstream findings' bnb-int4 noise-floor discussion).
- **Stage 5 = dissociation only.** Of Stage 5's sub-experiments, only the Fig-10 dissociation was
  saved in this run. Context-propagation, negation, and person-binding were not regenerated fresh.
- **Other stages not fresh.** Geometry (PCA / valence-arousal), validation (logit lens, steering),
  and post-training (Stage 8) numbers are **not** from this run; do not attribute them to us.
- **Sonnet r = 0.11 is quoted from the paper**, not recomputed here.

---

## 5. Reproduce

```bash
# on Explorer, with fresh vectors already in experiments/ant_emotion_concepts/extraction/
sbatch slurm/stage5_only.sbatch                 # job -> ti_stage5
# back on a laptop:
scripts/slurm_runner.py fetch <job_id> --results
scripts/parse_metric.py .runtime/fetched/<job_id>/dissociation.json   # -> r = 0.7216
```

Full provenance (job ids, methods, dates): [`metadata.json`](metadata.json).
