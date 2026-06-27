#!/usr/bin/env python3
"""Slurm managed-runner for the Emotion-Concepts reproduction.

Replaces multiautoresearch's `hf_job.py` (Hugging Face Jobs). Here the managed runner
is the Northeastern **Explorer** Slurm cluster, driven over passwordless SSH from this
laptop. The sbatch scripts live in `slurm/` here and are mirrored on the cluster at
`$REMOTE_REPO/slurm/`.

Subcommands:
  submit   <sbatch>     sbatch a script on the cluster; prints the job id
  status   [job_id]     squeue (one job, or all your jobs)
  watch    <job_id>     poll until the job reaches a terminal state; print final state
  fetch    <job_id>     scp the job's .out/.err logs (and known result JSONs) here

Config via env (defaults from RUNBOOK.md):
  TI_SSH_HOST   default login.explorer.northeastern.edu
  TI_REMOTE_REPO default /scratch/$USER/traitinterp   ($USER expands on the cluster)
  TI_LOCAL_FETCH default ./.runtime/fetched

This is a thin, auditable wrapper — it shells out to ssh/scp and never edits cluster
state beyond `sbatch`. It does not promote, diff, or mutate code.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HOST = os.environ.get("TI_SSH_HOST", "login.explorer.northeastern.edu")
REMOTE_REPO = os.environ.get("TI_REMOTE_REPO", "/scratch/$USER/traitinterp")
LOCAL_FETCH = Path(os.environ.get("TI_LOCAL_FETCH", ".runtime/fetched"))

# Terminal Slurm states (sacct).
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
            "NODE_FAIL", "BOOT_FAIL", "DEADLINE", "PREEMPTED"}

# Result JSONs worth pulling back for parse_metric.py after a run.
RESULT_JSONS = [
    "experiments/ant_emotion_concepts/results/stage5/dissociation.json",
    "experiments/ant_emotion_concepts/results/stage3_geometry/layer_sweep_pc1_valence.json",
]


def ssh(remote_cmd: str, capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["ssh", HOST, remote_cmd], text=True,
                          capture_output=capture)


def cmd_submit(args) -> int:
    sbatch = args.sbatch
    if not sbatch.startswith("slurm/"):
        sbatch = f"slurm/{sbatch}"
    remote = f"cd {REMOTE_REPO} && mkdir -p logs && sbatch {sbatch}"
    print(f"[submit] {HOST}: {remote}", file=sys.stderr)
    res = ssh(remote)
    sys.stderr.write(res.stderr)
    out = res.stdout.strip()
    print(out)
    m = re.search(r"Submitted batch job (\d+)", out)
    if not m:
        print("[submit] could not parse a job id — submission may have failed", file=sys.stderr)
        return 1
    job_id = m.group(1)
    print(f"[submit] job_id={job_id}", file=sys.stderr)
    print(f"[submit] watch:  scripts/slurm_runner.py watch {job_id}", file=sys.stderr)
    return 0


def cmd_status(args) -> int:
    if args.job_id:
        remote = f'squeue -j {args.job_id} -o "%.18i %.12P %.20j %.8T %.10M %.6D %R"'
    else:
        remote = 'squeue -u $USER -o "%.18i %.12P %.20j %.8T %.10M %.6D %R"'
    res = ssh(remote)
    sys.stderr.write(res.stderr)
    print(res.stdout.rstrip())
    return res.returncode


def _sacct_state(job_id: str) -> str | None:
    res = ssh(f"sacct -j {job_id} --format=State --noheader --parsable2 2>/dev/null")
    states = [s.strip() for s in res.stdout.splitlines() if s.strip()]
    if not states:
        return None
    # The batch/sub-steps repeat the state; take the main (first) entry.
    return states[0].split()[0]


def cmd_watch(args) -> int:
    job_id = args.job_id
    interval = args.interval
    print(f"[watch] job {job_id}, polling every {interval}s (Ctrl-C to stop)", file=sys.stderr)
    while True:
        state = _sacct_state(job_id)
        stamp = time.strftime("%H:%M:%S")
        if state is None:
            print(f"[watch {stamp}] job {job_id} not yet in sacct (pending/queued)…", file=sys.stderr)
        else:
            print(f"[watch {stamp}] {state}", file=sys.stderr)
            base = state.split()[0]
            if base in TERMINAL:
                print(base)  # stdout = final state, machine-readable
                return 0 if base == "COMPLETED" else 2
        time.sleep(interval)


def cmd_fetch(args) -> int:
    job_id = args.job_id
    dest = LOCAL_FETCH / job_id
    dest.mkdir(parents=True, exist_ok=True)
    # Logs: filenames are <job-name>_<job_id>.out/.err — glob on the id.
    log_glob = f"{REMOTE_REPO}/logs/*_{job_id}.out {REMOTE_REPO}/logs/*_{job_id}.err"
    print(f"[fetch] logs -> {dest}", file=sys.stderr)
    subprocess.run(f'scp "{HOST}:{log_glob}" "{dest}/"', shell=True)
    if args.results:
        for rel in RESULT_JSONS:
            local = dest / Path(rel).name
            print(f"[fetch] {rel} -> {local}", file=sys.stderr)
            subprocess.run(["scp", f"{HOST}:{REMOTE_REPO}/{rel}", str(local)])
    print(str(dest))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("submit", help="sbatch a script on the cluster")
    p.add_argument("sbatch", help="sbatch filename, e.g. full_extraction.sbatch")
    p.set_defaults(func=cmd_submit)

    p = sub.add_parser("status", help="squeue (one job or all yours)")
    p.add_argument("job_id", nargs="?", help="optional job id")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("watch", help="poll until terminal state")
    p.add_argument("job_id")
    p.add_argument("--interval", type=int, default=120, help="poll seconds (default 120)")
    p.set_defaults(func=cmd_watch)

    p = sub.add_parser("fetch", help="scp logs (and result JSONs) here")
    p.add_argument("job_id")
    p.add_argument("--results", action="store_true", help="also pull known result JSONs")
    p.set_defaults(func=cmd_fetch)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
