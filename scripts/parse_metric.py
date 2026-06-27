#!/usr/bin/env python3
"""Recompute the Emotion-Concepts headline metric(s) from a result JSON.

Replaces multiautoresearch's `parse_metric.py` (which read a single `val_bpb:` line
from a training log). Here the metric is a *Pearson r recomputed from the analysis
JSON* a Slurm job produced, so this reads the JSON directly. Pure stdlib — no numpy.

Usage:
  parse_metric.py path/to/dissociation.json            # headline cross-position r
  parse_metric.py path/to/dissociation.json --subset all
  parse_metric.py path/to/layer_sweep_pc1_valence.json # stage-3 PC1-vs-valence r

Auto-detects the metric from the filename; prints a JSON block with
{metric, value, ref_value, n, ...} suitable for handing to record_run.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Official Fig-10 probe subset and layer (must match notebook/build_notebook.py).
EMO_FIG10 = ["happy", "calm", "loving", "sad", "afraid", "angry"]
FIG10_LAYER = "49"
REF_DISSOCIATION = 0.63   # LessWrong post headline (Llama); paper Sonnet ref is 0.11


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        raise ValueError("need >=2 points for a correlation")
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    denom = (sxx * syy) ** 0.5
    if denom == 0:
        raise ValueError("zero variance — cannot compute Pearson r")
    return sxy / denom


def dissociation_r(data: dict, layer: str, subset: list[str] | None) -> dict:
    """Cross-position correlation: user_period vs assistant_colon at `layer`."""
    up, ac = [], []
    for scenario in data["results"]:
        projections = scenario["projections"]
        emotions = subset if subset is not None else list(projections.keys())
        for e in emotions:
            cell = projections.get(e, {}).get(layer, {})
            if "user_period" in cell and "assistant_colon" in cell:
                up.append(cell["user_period"])
                ac.append(cell["assistant_colon"])
    return {
        "metric": "cross_position_pearson_r",
        "value": round(pearson(up, ac), 4),
        "ref_value": REF_DISSOCIATION,
        "n": len(up),
        "layer": int(layer),
        "subset": "fig10_6emotion" if subset is not None else "all_emotions",
    }


def layer_sweep_r(data: dict) -> dict:
    """Stage-3 best PC1-vs-human-valence correlation across the layer sweep.

    Mirrors the notebook: pick the layer with the largest |PC1 vs valence|.
    `per_layer` maps a layer string -> per-layer stats dict.
    """
    items = data["per_layer"].items()  # (layer_str, stats)
    best_layer, best = max(items, key=lambda kv: abs(kv[1].get("abs_pc1_vs_valence", 0.0)))
    return {
        "metric": "pc1_vs_valence_r",
        "value": round(best["pc1_vs_valence_r"], 4),
        "ref_value": None,
        "n": best.get("n_overlap"),
        "layer": int(best_layer),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json_path", help="result JSON produced by a Slurm analysis job")
    ap.add_argument("--subset", choices=["fig10", "all"], default="fig10",
                    help="dissociation only: Fig-10 6-emotion subset (default) or all emotions")
    ap.add_argument("--layer", default=FIG10_LAYER, help="dissociation only: layer key (default 49)")
    args = ap.parse_args()

    path = Path(args.json_path)
    if not path.is_file():
        raise SystemExit(f"no such file: {path}")
    data = json.loads(path.read_text())
    name = path.name

    if "dissociation" in name:
        subset = EMO_FIG10 if args.subset == "fig10" else None
        out = dissociation_r(data, args.layer, subset)
    elif "layer_sweep_pc1_valence" in name:
        out = layer_sweep_r(data)
    else:
        raise SystemExit(
            f"don't know how to parse '{name}'. Supported: *dissociation*.json, "
            f"*layer_sweep_pc1_valence*.json. Add a handler in parse_metric.py."
        )

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
