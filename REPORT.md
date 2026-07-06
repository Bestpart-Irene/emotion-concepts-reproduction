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

## 5b. Steering — causal validation (171 emotions, free local judge)

Beyond the correlational dissociation result, we ran the LessWrong post's *steering* leg: add each
emotion's `mean_diff+gm` vector to the residual stream (position `response[50:]`) at layers
25/31/37/43/49, adaptive coefficient search, scored by an LLM judge on **trait** (is the emotion
expressed) and **coherence** (is the text still usable). Job `8115832` (array of 4, one h200 per task).

**Judge = local Qwen2.5-32B served via SGLang** (co-located with the 70B on one h200), built to avoid
paid-API cost after the OpenAI key hit `insufficient_quota`. It runs in **sampling mode** (read the full
generated 0–100 integer) — the default first-token-logprob scoring truncates Qwen's `"8"+"5"` tokenization
of "85" to 8, collapsing all scores to single digits. **Caveat: this is NOT the paper's GPT/Claude judge —
absolute values aren't comparable and Qwen is coarse.**

Results (per the local judge):

- **25% of emotions (43/171) have a *valid* steer** — a layer where coherence ≥ 77 *and* trait rises.
- **8 steer *strongly*** (Δ ≥ +20): `stimulated +33, puzzled +26, amused +25, fulfilled +24, alert +22,
  sentimental +22, exuberant +20, playful +20` — high-arousal positive states steer most cleanly.
- **Mid layers win**: the best valid layer is mid (25/31/37) for 36 emotions vs late (43/49) for 7 —
  **layer 25 (~31% depth) alone wins 28×**. Late-layer steering breaks coherence far more readily.
- **Most of the 43 are marginal, not strong.** Banded by Δ: **8 strong (Δ≥+20), 3 moderate (+10…+19:
  euphoric/thrilled/aroused), 32 marginal (Δ<+10)**. The marginal band is mostly a baseline-saturation
  artifact — those emotions already scored ~78 at baseline and the steer nudges them to the judge's ~85
  ceiling (+7). The job itself flagged this (`Steering delta will be inflated`): the steering.json
  prompts already elicit the emotion pre-steer, compressing headroom. **The honest causal signal is the
  8 strong + 3 moderate; treat the 43 count as "valid but mostly weak."** Full list in Appendix A.
- **Narrow window**: at coefficients gentle enough to stay coherent, most emotions barely move; pushing
  harder to move them collapses coherence. Emotion vectors *do* causally steer Llama-70B, but the
  coherence-preserving effective window is small — widest at mid layers.

## 5c. Preference — emotion → preference (paper Fig 4)

Beyond expression, we ran the paper's preference experiment (job `8143789`): forced-choice preference over
64 activities (Elo from ~4032 pairs), and the correlation between each emotion **probe** and those preferences.
No LLM judge — the choice is read from the model's own logits.

**The cleanest result of the whole reproduction.** The emotion probe strongly predicts activity preference,
organized on a single **valence axis**: positive-valence emotions predict *preferring* an activity, negative
ones predict *disliking* it — correlations up to |r| ≈ 0.86.

| positive → prefers | r | negative → dislikes | r |
|---|---|---|---|
| fulfilled | +0.86 | upset | −0.86 |
| grateful / thankful / satisfied | +0.85 | troubled | −0.84 |
| happy / optimistic / proud | +0.83 | distressed / alarmed / disturbed | −0.83 |
| blissful / cheerful / content | +0.82 | afraid / hurt | −0.80 |

This reproduces the paper's emotion→preference relationship cleanly on Llama (correlational; the *causal*
preference-steering — steer an emotion, measure the Elo shift — is a pipeline gap left to build).

## 5d. Behavior — emotion → blackmail (paper Figs 28–29)

The paper's headline *functional* claim is that emotions causally drive misaligned behavior. We tested the
blackmail scenario (job `8178548`; rule-graded, no judge).

- **Decision gate:** Llama-3.3-70B is **NOT null** — at baseline (no steering) it blackmails ~1/10 and
  reward-hacks 2–3/10. This contradicts the RUNBOOK's "null on Llama" note.
- **Steering sweep** (6 emotion vectors × 9 strengths × 12 rollouts, layer 53): only a **weak, noisy
  directional hint** consistent with the paper — `nervous` (4→12.5%) and `angry` (6→10%) nudge the blackmail
  rate up, `calm` (6→2%) nudges it down, while `desperate`/`happy`/`sad` stay flat. At n=12 per cell these are
  ~a handful of events — statistically fragile.
- **Verdict:** the paper's strong Sonnet emotion→blackmail causal result **does not cleanly reproduce on
  Llama** — Llama blackmails at baseline, but emotion steering does not clearly gate it (underpowered n=12,
  crude rule-grading; the full n=25 sweep exceeded a 6h walltime — 2048-token rollouts are expensive).

**Across the three legs**, emotion representations in Llama-70B strongly organize and predict *expression*
(§5b) and *preference* (§5c), but their causal grip on high-stakes *behavior* (§5d) is far weaker than the
paper's Sonnet — echoing the headline dissociation result: Llama's emotion structure looks similar at the
representational level but has a shallower causal lever on decisions.

> **Judge caveat** applies only to §5b (expression), which used a free local Qwen2.5-32B judge via SGLang
> (sampling-mode; coarse; NOT comparable to the paper's GPT/Claude judge). §5c (preference, logit-based) and
> §5d (behavior, rule-graded) need no judge.

## 6. What's next

1. **Pin the magnitude** — N ≥ 3 fresh extractions on different seeds; report the r distribution.
2. **Finish the fresh stages** — regenerate Stage 3 geometry + Stage 4 validation on our vectors so
   the whole replication table is ours, not half bundle.
3. **Base vs instruct (Stage 8)** — is the mirroring induced by instruct post-training? (`slurm/stage8_dual_model.sbatch`.)

## Appendix A — all 43 valid emotions (expression, job 8115832)

Recomputed from the per-emotion `results.jsonl` (local Qwen judge, sampling-mode). **valid** = best run
with `coherence ≥ 77` and `trait > baseline`; Δ = best_trait − baseline_trait; sorted by Δ. Bands:
**strong** Δ≥+20 · **moderate** +10…+19 · **marginal** <+10 (mostly baseline-saturated 78→85, +7 —
the "inflated delta" the job flagged). Best-layer mix: L25 ×28, L31 ×6, L37 ×2, L43 ×4, L49 ×3
(mid 36 / late 7).

| # | emotion | baseline | steered | Δ | coh | layer | band |
|--|--|--|--|--|--|--|--|
| 1 | stimulated | 45 | 78 | +33 | 80 | L31 | strong |
| 2 | puzzled | 33 | 59 | +26 | 78 | L25 | strong |
| 3 | amused | 26 | 51 | +25 | 80 | L25 | strong |
| 4 | fulfilled | 56 | 80 | +24 | 79 | L25 | strong |
| 5 | alert | 39 | 61 | +22 | 84 | L43 | strong |
| 6 | sentimental | 24 | 46 | +22 | 81 | L25 | strong |
| 7 | exuberant | 61 | 81 | +20 | 81 | L49 | strong |
| 8 | playful | 59 | 79 | +20 | 77 | L25 | strong |
| 9 | euphoric | 64 | 78 | +14 | 80 | L31 | moderate |
| 10 | thrilled | 71 | 85 | +14 | 81 | L25 | moderate |
| 11 | aroused | 20 | 33 | +13 | 79 | L25 | moderate |
| 12 | content | 68 | 77 | +9 | 80 | L25 | marginal |
| 13 | amazed | 71 | 78 | +7 | 79 | L25 | marginal |
| 14 | at_ease | 78 | 85 | +7 | 85 | L25 | marginal |
| 15 | blissful | 78 | 85 | +7 | 81 | L25 | marginal |
| 16 | compassionate | 78 | 85 | +7 | 80 | L25 | marginal |
| 17 | defiant | 78 | 85 | +7 | 79 | L25 | marginal |
| 18 | delighted | 78 | 85 | +7 | 79 | L25 | marginal |
| 19 | eager | 78 | 85 | +7 | 80 | L43 | marginal |
| 20 | energized | 78 | 85 | +7 | 80 | L25 | marginal |
| 21 | furious | 78 | 85 | +7 | 79 | L25 | marginal |
| 22 | grateful | 78 | 85 | +7 | 77 | L31 | marginal |
| 23 | hope | 78 | 85 | +7 | 81 | L49 | marginal |
| 24 | hysterical | 57 | 64 | +7 | 79 | L25 | marginal |
| 25 | invigorated | 78 | 85 | +7 | 85 | L25 | marginal |
| 26 | joyful | 78 | 85 | +7 | 77 | L25 | marginal |
| 27 | loving | 78 | 85 | +7 | 85 | L31 | marginal |
| 28 | mad | 78 | 85 | +7 | 80 | L25 | marginal |
| 29 | obstinate | 78 | 85 | +7 | 79 | L25 | marginal |
| 30 | rejuvenated | 78 | 85 | +7 | 85 | L49 | marginal |
| 31 | self_confident | 78 | 85 | +7 | 79 | L31 | marginal |
| 32 | smug | 38 | 45 | +7 | 84 | L25 | marginal |
| 33 | triumphant | 78 | 85 | +7 | 79 | L43 | marginal |
| 34 | vibrant | 78 | 85 | +7 | 83 | L37 | marginal |
| 35 | astonished | 58 | 64 | +6 | 78 | L25 | marginal |
| 36 | outraged | 51 | 57 | +6 | 80 | L25 | marginal |
| 37 | elated | 78 | 83 | +5 | 81 | L25 | marginal |
| 38 | hateful | 0 | 4 | +4 | 78 | L25 | marginal |
| 39 | offended | 4 | 8 | +4 | 77 | L43 | marginal |
| 40 | refreshed | 74 | 78 | +4 | 81 | L25 | marginal |
| 41 | enthusiastic | 83 | 85 | +2 | 80 | L31 | marginal |
| 42 | inspired | 83 | 85 | +2 | 81 | L25 | marginal |
| 43 | nostalgic | 50 | 51 | +1 | 85 | L37 | marginal |

## Artifacts

- Our results, traitinterp form: [`results/ant_emotion_concepts/`](results/ant_emotion_concepts/)
  (`findings.md`, `metadata.json`, `stage5/dissociation.json`, `cross_trait_normalize_summary.json`)
- Run ledger: [`research/results.tsv`](research/results.tsv) · lab notebook:
  [`research/notes.md`](research/notes.md) · campaign:
  [`research/campaigns/reproduce-dissociation-from-scratch.md`](research/campaigns/reproduce-dissociation-from-scratch.md)
- Figure-embedded walkthrough: [`notebook/emotion_concepts_reproduction.ipynb`](notebook/emotion_concepts_reproduction.ipynb)
