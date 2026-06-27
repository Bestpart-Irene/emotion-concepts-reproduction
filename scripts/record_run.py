#!/usr/bin/env python3
"""Append one run to the research ledger (research/results.tsv).

Replaces multiautoresearch's `submit_patch.py`. There is no local "master" to promote
here — this is a bounded reproduction — so recording a run just means appending a row
to the append-only TSV. Columns are fixed by DISCIPLINE.md:

  job_id  phase  stage  model  layer  metric  value  ref_value  state  notes

Typical use (pipe parse_metric's JSON in for value/ref/metric/layer):

  scripts/parse_metric.py dissociation.json \
    | scripts/record_run.py --job-id 123456 --phase 3 --stage stage5 \
        --model Llama-3.3-70B-Instruct --state COMPLETED \
        --notes "from-scratch extraction"

Or pass everything explicitly with flags. Explicit flags override piped JSON.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LEDGER = Path(__file__).resolve().parent.parent / "research" / "results.tsv"
COLUMNS = ["job_id", "phase", "stage", "model", "layer",
           "metric", "value", "ref_value", "state", "notes"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for col in COLUMNS:
        ap.add_argument(f"--{col.replace('_', '-')}", default=None,
                        help=f"{col} column" + (" (auto-filled from piped parse_metric JSON if omitted)"
                                                if col in {"metric", "value", "ref_value", "layer"} else ""))
    ap.add_argument("--dry-run", action="store_true", help="print the row, do not write")
    args = ap.parse_args()

    row = {col: getattr(args, col) for col in COLUMNS}

    # Fill metric/value/ref_value/layer from piped parse_metric.py JSON when not given.
    if not sys.stdin.isatty():
        piped = sys.stdin.read().strip()
        if piped:
            try:
                data = json.loads(piped)
                for col in ("metric", "value", "ref_value", "layer"):
                    if row[col] is None and data.get(col) is not None:
                        row[col] = data[col]
            except json.JSONDecodeError:
                pass  # not JSON on stdin — ignore, rely on flags

    row = {k: ("" if v is None else str(v)) for k, v in row.items()}
    for k, v in row.items():
        if "\t" in v or "\n" in v:
            raise SystemExit(f"value for '{k}' contains a tab/newline; would corrupt the TSV")

    line = "\t".join(row[c] for c in COLUMNS)

    if args.dry_run:
        print(line)
        return 0

    if not LEDGER.exists():
        LEDGER.write_text("\t".join(COLUMNS) + "\n")
    with LEDGER.open("a") as fh:
        fh.write(line + "\n")
    print(f"recorded -> {LEDGER}")
    print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
