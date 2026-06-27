# Emotion Concepts — Reproduction

Reproducing ewern's [LessWrong post](https://www.lesswrong.com/posts/sJQ62HbA76s3aiuiT)
— a replication of Anthropic's *Emotion Concepts and their Function in a Large Language Model*
(Sofroniew et al. 2026) on **Llama 3.3 70B Instruct**, using the
[traitinterp](https://github.com/ewernn/traitinterp) toolkit, driven on the Northeastern
**Explorer** HPC cluster (Slurm + conda).

## Headline result

On Llama 3.3 70B Instruct, the **assistant** token position strongly mirrors the **user's**
emotion — cross-position Pearson **r ≈ 0.63** — whereas the paper reports the two stay independent
on Claude Sonnet 4.5 (**r ≈ 0.11**). I.e. the user/assistant *emotion dissociation* the paper finds
on Sonnet **does not hold on Llama**.

Reproduced from traitinterp's precomputed `emotion-concepts-v1` bundle (`r = 0.6299`, official Fig-10
method: layer 49, 6-emotion subset, pooled over the paper's 8 Table-3 scenarios). An independent
from-scratch run (regenerating the vectors on 70B, then recomputing) runs on the cluster.

## What's here

| Path | What |
|---|---|
| `notebook/emotion_concepts_reproduction.ipynb` | Executed, figure-embedded walkthrough of the reproduced results (no GPU needed) |
| `notebook/build_notebook.py` | Regenerates the notebook from source |
| `slurm/` | Slurm job scripts for the cluster (env setup, model precache, full extraction, Stage 8) |
| `RUNBOOK.md` | End-to-end cluster runbook (paths, steps, expected metrics) |
| `DISCIPLINE.md` | Lightweight single-change / read-only experiment discipline |
| `CLAUDE.md` | Claude Code project instructions (entry point: `claude --agent coordinator`) |
| `.claude/agents/` | Read-only research subagents — `coordinator`, `planner`, `researcher`, `reporter` |
| `research/` | Durable lab notebook — `results.tsv` ledger, `notes.md`, `do-not-repeat.md`, `paper-ideas.md`, `templates/`, `campaigns/` |
| `scripts/` | Slurm managed runner (`slurm_runner.py`), metric recompute (`parse_metric.py`), ledger append (`record_run.py`) |

A lightweight subset of [burtenshaw/multiautoresearch](https://github.com/burtenshaw/multiautoresearch)'s
orchestration, adapted to this bounded reproduction (Claude Code agents + Slurm runner + a `research/`
notebook); the upstream master/promotion/patch/worktree-fleet machinery is intentionally left out (see `CLAUDE.md`).

> **Not committed** (see `.gitignore`): secrets/`env`, cluster logs, and `notebook/data/`
> (the ~19 MB upstream bundle subset the notebook reads). The notebook is committed *with outputs
> embedded*, so you can read all figures on GitHub without the data.

## View the results

Open `notebook/emotion_concepts_reproduction.ipynb` on GitHub (figures render inline).

## Re-run the notebook locally

```bash
# fetch the upstream precomputed bundle that the notebook reads
mkdir -p notebook/data && cd notebook
curl -L https://github.com/ewernn/traitinterp/releases/download/emotion-concepts-v1/ant_emotion_concepts.tar.zst \
  | tar --zstd -xf - -C /tmp/aec
# place results/, paper_figures/, config.json and the two dataset samples under notebook/data/
# (see notebook/build_notebook.py for the exact layout), then:
jupyter notebook emotion_concepts_reproduction.ipynb   # kernel: python3, needs only numpy + matplotlib
```

## Reproduce on a cluster

See [`RUNBOOK.md`](RUNBOOK.md). Note: install a **CUDA 12.x** torch build (the Explorer driver is
CUDA 12.x — the default `pip install torch` pulls a cu13 build and silently falls back to CPU).
Use `multigpu` (24 h limit) for the >8 h full-extraction job; `gpu` is capped at 8 h.

## Provenance

No real user data is involved. Emotion vectors = mean-difference of the model's own activations while
it writes emotion-laden stories; the 8 "user emotion" dissociation scenarios are verbatim from the
paper's Table 3. Sonnet's r≈0.11 is quoted from the paper (not recomputed).
