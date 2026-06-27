# Do not repeat

Known dead ends and traps. Check this before proposing or submitting a run — re-running
something already listed here wastes GPU-hours. Add an entry the moment a run fails for an
avoidable reason.

## Environment / cluster

- **cu13 torch silently runs on CPU.** A plain `pip install torch` pulls a CUDA-13 wheel; the
  Explorer driver is CUDA 12.x, so `torch.cuda.is_available()` is False and the job falls back to
  CPU (hours wasted, no error). Always install a cu126 build and verify with a tiny GPU job first.
  (See RUNBOOK §1.)
- **`gpu` partition caps at 8 h.** The full from-scratch extraction is >8 h — submit it to
  `multigpu` (24 h), not `gpu`. Submitting extraction to `gpu` gets it killed mid-run.

## Method / comparability

- **Don't mix the bundle's author vectors with freshly-extracted ones.** `full_extraction.sbatch`
  moves the bundle vectors to `extraction_bundle_baseline/` and clears stale stage outputs first;
  skipping that makes stages silently reuse author vectors, so a "from-scratch" r is not actually
  from scratch.
- **Headline r uses the official Fig-10 method only** (layer 49, the 6-emotion subset
  `happy/calm/loving/sad/afraid/angry`, pooled over the 8 scenarios). The repo's internal
  `findings.md` quotes r=0.7718 from a different/earlier configuration — do not treat that as the
  comparison target. The post's headline is ~0.63.
