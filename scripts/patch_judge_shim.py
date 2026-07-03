#!/usr/bin/env python3
"""Shim for a local (SGLang/vLLM) judge: let TraitJudge's OpenAI backend take the
served model name from the TRAIT_JUDGE_MODEL env var, so a local server's model
(e.g. Qwen/Qwen2.5-32B-Instruct) is used instead of the gpt-4.1-mini default.
base_url is already picked up from OPENAI_BASE_URL by the openai SDK.

Run ON THE CLUSTER against the traitinterp checkout. Idempotent; keeps a .bak.
Repo path defaults to $TI_REMOTE_REPO or /scratch/$USER/traitinterp."""
import os, pathlib, sys, shutil

REPO = os.environ.get("TI_REMOTE_REPO") or f"/scratch/{os.environ.get('USER','')}/traitinterp"
p = pathlib.Path(REPO) / "utils" / "judge.py"
src = p.read_text()
old = "            model=model or DEFAULT_OPENAI_MODEL,\n"
new = "            model=model or os.environ.get(\"TRAIT_JUDGE_MODEL\") or DEFAULT_OPENAI_MODEL,\n"

if new.strip() in src:
    print("already patched — nothing to do")
    sys.exit(0)
if old not in src:
    print("PATTERN NOT FOUND — aborting, no change made")
    sys.exit(1)

shutil.copy(str(p), str(p) + ".bak")
p.write_text(src.replace(old, new, 1))
print(f"patched {p} (backup .bak); TRAIT_JUDGE_MODEL now honored")
