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

## 2026-07-05 · behavior+preference · emotion→behavior (blackmail) weak/null on Llama; emotion→preference strong
- **Preference (job 8143789, COMPLETED):** Fig-4 forced-choice preference Elo over 64 activities + probe-preference
  correlation. STRONG clean valence axis: positive emotions predict *preferring* an activity (fulfilled +0.86,
  grateful/thankful/satisfied +0.85, happy/optimistic +0.83), negative anti-predict (upset -0.86, troubled -0.84,
  afraid -0.80). No judge needed (forced-choice/logits). Reproduces the paper's emotion→preference relationship on Llama.
- **Behavior / blackmail (job 8178548, COMPLETED; n=25 8175605 timed out at 6h, redone n=12/8h):** decision gate
  re-confirmed Llama is NON-null (baseline blackmail 1/10, RH 2-3/10 — contradicts the RUNBOOK "null on Llama" note).
  BUT the emotion-steering sweep (6 vectors × 9 strengths × 12 rollouts, layer 53, rule-graded) shows only a
  weak/noisy directional hint: nervous(4→12.5%) & angry(6→10%) nudge blackmail UP, calm(6→2%) DOWN, desperate/happy/sad
  flat. Underpowered (n=12; paper used 50) — the paper's strong Sonnet emotion→blackmail causal result does NOT
  cleanly reproduce on Llama. Interesting cross-model finding: Llama blackmails at baseline but emotion doesn't
  clearly gate it.
- **Bug found+fixed:** stage7 run_probing hardcoded method=mean_diff; our vectors are mean_diff+gm → patched
  shared.py load_single_emotion_vector default (backup .bak). Also: blackmail rollouts are 2048 tokens → very slow
  (n=25 exceeds 6h walltime).

## 2026-07-03 · steering · 171-emotion causal steering sweep, free local Qwen judge (25% valid)
- model Llama-3.3-70B-Instruct · layers 25/31/37/43/49 · method mean_diff+gm · position response[50:]
- job_id 8115832 (array 0-3, `slurm/steering_all_sglang.sbatch`, co-located SGLang Qwen2.5-32B judge + 70B on one h200)
- **JUDGE = local Qwen2.5-32B via SGLang, sampling-mode** (built to dodge the OpenAI-quota 429; NOT the
  post/paper's GPT-4.1-mini/Claude — absolute trait/coherence values are NOT comparable, and Qwen is coarse).
  Sampling (not logprob) is required: Qwen tokenizes "85" as "8"+"5", so first-token-logprob scoring truncates
  to 8 → coherent text scored 6.7/100; reading the full generated integer fixed it (baseline coherence 70).
- **43/171 (25%) emotions have a VALID steer** (a layer with coherence>=77 AND positive trait delta); **8 STRONG**
  (delta>=+20): stimulated +33, puzzled +26, amused +25, fulfilled +24, alert +22, sentimental +22, exuberant +20, playful +20.
- **Best steering layer is MID, not late**: winning valid layer mid(25/31/37)=37 vs late(43/49)=6 — **L25 (~31% depth) wins 29×**.
- Interpretation: emotion vectors *do* causally steer Llama-70B, but the valid window is narrow — at coefficients
  gentle enough to keep coherence>=77 most emotions barely move; pushing harder breaks coherence (esp. late layers).
  Mid layers (esp. 25) preserve coherence far better. High-arousal positive states steer most cleanly. Free-judge
  caveat: the exact rate/values track Qwen's coarseness, not necessarily the paper's judge.

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
