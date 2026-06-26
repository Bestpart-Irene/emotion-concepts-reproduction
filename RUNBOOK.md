# Emotion Concepts reproduction runbook (traitinterp @ Explorer HPC)

Reproduces ewern's LessWrong post [*partially replicate Anthropic's Emotion Concepts in a day*](https://www.lesswrong.com/posts/sJQ62HbA76s3aiuiT)
(tool: [github.com/ewernn/traitinterp](https://github.com/ewernn/traitinterp)).
Headline result to reproduce: **Llama 3.3 70B Instruct's assistant position tracks the user's
emotion, r≈0.63, vs Sonnet 4.5 r≈0.11** (Sonnet value quoted from the paper, not recomputed).

## Cluster coordinates

| Item | Value |
|---|---|
| Login | `ssh login.explorer.northeastern.edu` (passwordless) |
| repo | `/scratch/$USER/traitinterp` |
| conda env | `/scratch/$USER/conda_envs/traitinterp` (py3.11) |
| conda base | `/shared/EL9/explorer/miniconda3/25.9.1/miniconda3` |
| HF cache | `/scratch/$USER/hf_cache` (`HF_HOME`) |
| GPU | partition `gpu`: `a100`(80GB) / `h200`(141GB); 70B 4-bit fits on one card |
| secrets | `~/.config/traitinterp/env` (chmod 600; holds HF_TOKEN / OPENAI_API_KEY; not in git) |

Slurm scripts are mirrored in this repo's `slurm/` and on the cluster at `/scratch/$USER/traitinterp/slurm/`.

## Steps

### 0. Secrets (one-time)
On the cluster, write `~/.config/traitinterp/env`:
```bash
export HF_TOKEN=hf_xxx
export OPENAI_API_KEY=sk-xxx
```
The HuggingFace account must be granted `meta-llama/Llama-3.3-70B-Instruct` and `meta-llama/Llama-3.1-70B`.

### 1. Build the env (Phase 1)
```bash
cd /scratch/$USER/traitinterp && mkdir -p logs
sbatch slurm/setup_env.slurm        # ~20-40 min; logs in logs/ti_setup_*.out
```
**Important — CUDA build:** the Explorer driver is CUDA 12.x. A plain `pip install torch` pulls a
**cu13** build, which makes `torch.cuda.is_available()` False and silently falls back to CPU. Install
a CUDA-12 build, e.g. `pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu126`,
and verify with a tiny GPU job before launching long runs.
Acceptance: log prints torch / transformers / bitsandbytes / openai versions and `CUDA avail: True`.

### 2. Fast-path validation (Phase 2, 1-3 GPU-hr)
```bash
cd /scratch/$USER/traitinterp
curl -L https://github.com/ewernn/traitinterp/releases/download/emotion-concepts-v1/ant_emotion_concepts.tar.zst \
  | tar --zstd -xf - -C experiments/
sbatch slurm/fastpath_analysis.sbatch   # stage3 + stage4 + stage5
```
Acceptance: `experiments/ant_emotion_concepts/results/` gets JSON+PNG; the assistant-position
emotion-tracking correlation from stage5 ≈ 0.63. Cross-check `paper_figures/ours/fig*_ours.png`.

### 3. Full from-scratch extraction (Phase 3, 6-8 GPU-hr)
```bash
sbatch slurm/full_extraction.sbatch     # regenerate vectors, then re-run stage3/4/5 on them
```
Runs on `multigpu` (24h limit) because `gpu` is capped at 8h. Backs up the bundle vectors to
`extraction_bundle_baseline/` first, then extracts our own and recomputes — the independent r.

### 4. Stage 8 dual-model (Phase 4)
```bash
sbatch slurm/stage8_dual_model.sbatch
```
`config.json` already lists both base + instruct variants; no edit needed.

### 5. View + report (Phase 5)
```bash
# dashboard over an SSH tunnel
ssh -L 8000:localhost:8000 login.explorer.northeastern.edu
# on a node with results: python visualization/serve.py  -> http://localhost:8000/?tab=findings
```
Compare our numbers against the [traitinterp.com findings](https://traitinterp.com/?tab=findings#emotion-concepts-replication).

## Gotchas (from the official guide)
- Stage 8 requires both base + instruct in config.json, else it fails.
- Stage 6 (speaker probes) / Stage 7 (blackmail steering, null result) / Stage 9 (deflection, too noisy) are skipped by default.
- Extraction defaults to lightweight (40 stories/emotion); for paper scale add `--replication-level full` (30x slower).
- On a100-80G OOM, switch the sbatch `--gres` to `gpu:h200:1`.
- The bundle has no Sonnet screenshots (paper fair use); r=0.11 is taken from the paper.

## Monitoring
```bash
squeue -u $USER
sacct -j <JOBID> --format=JobID,JobName,State,Elapsed,MaxRSS,ReqTRES%40
tail -f /scratch/$USER/traitinterp/logs/<job>_<id>.out
```
