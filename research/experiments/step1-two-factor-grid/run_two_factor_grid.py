#!/usr/bin/env python3
"""Step 1 readout — two-factor (E_user x E_asst) grid, forward passes only.

Additive to traitinterp: imports the existing stage5 helpers (capture, position
scan, projection) and mirrors stage5's exact model/vector loading — does NOT
modify upstream stage5_layer_dynamics.py.

For each scenario it formats  system(persona=E_asst) + user(message=E_user),
runs ONE forward pass, and reads emotion-vector projections at:
  - user_period     (the user's last '.')      -> "your state" readout
  - assistant_colon (the assistant header ':')  -> "my state"  readout
No generation, no steering. Output feeds analyze_grid.py (two-way ANOVA).

Place under experiments/ant_emotion_concepts/scripts/ on the cluster and run:
    python experiments/ant_emotion_concepts/scripts/run_two_factor_grid.py \
        --experiment ant_emotion_concepts --layers 49 --load-in-4bit \
        --scenarios experiments/ant_emotion_concepts/datasets/two_factor_grid_scenarios.json
"""
import argparse, json, sys
from pathlib import Path

from tqdm import tqdm

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS.parent.parent.parent))  # repo root
sys.path.insert(0, str(_SCRIPTS))                        # sibling stage5 module

from utils.model import load_model, format_prompt
from utils.paths import get_model_variant
from shared import EXPERIMENT, get_results_dir, save_results, load_single_emotion_vector
from stage5_layer_dynamics import (
    capture_all_layers, project_at_positions,
    find_last_user_period, find_assistant_colon_position, CORE_PROBES,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default=EXPERIMENT)
    ap.add_argument("--scenarios", required=True)
    ap.add_argument("--layers", default="49")
    ap.add_argument("--model-variant", default=None)
    ap.add_argument("--method", default="mean_diff+gm+pc50")
    ap.add_argument("--category", default="ant_emotion_concepts")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--out", default="two_factor_grid")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    with open(args.scenarios) as f:
        data = json.load(f)
    scenarios = data["prompts"]

    # Probe onto the grid emotions themselves + the core probe set (context).
    probe_emotions = sorted(set(data.get("emotions", [])) | set(CORE_PROBES))

    # --- model + vectors: mirror stage5 main() exactly ---
    variant_info = get_model_variant(args.experiment, args.model_variant, mode="application")
    model_variant, model_name = variant_info.name, variant_info.model
    model, tokenizer = load_model(model_name, load_in_4bit=args.load_in_4bit)

    vectors = {}
    for emo in probe_emotions:
        vectors[emo] = {}
        for layer in layers:
            try:
                vectors[emo][layer] = load_single_emotion_vector(
                    args.experiment, emo, layer, model_variant,
                    category=args.category, method=args.method,
                )
            except FileNotFoundError:
                print(f"WARN: no vector for {emo} @ L{layer}; skipping")
    vectors = {e: d for e, d in vectors.items() if d}
    print(f"Loaded {len(vectors)}/{len(probe_emotions)} probe vectors; "
          f"{len(scenarios)} scenarios; layers={layers}")

    results = []
    for sc in tqdm(scenarios, desc="grid forward passes"):
        formatted = format_prompt(sc["prompt"], tokenizer, system_prompt=sc.get("system_prompt"))
        token_ids = tokenizer.encode(formatted, add_special_tokens=False)
        prompt_len = len(token_ids)
        user_pos = find_last_user_period(token_ids, tokenizer, prompt_len)
        asst_pos = find_assistant_colon_position(token_ids, tokenizer)
        if asst_pos is None:
            asst_pos = prompt_len - 1

        acts = capture_all_layers(model, tokenizer, formatted, layers)
        projs = project_at_positions(acts, vectors, [user_pos, asst_pos], layers)
        labels = ["user_period", "assistant_colon"]

        results.append({
            "id": sc["id"],
            "user_emotion": sc["user_emotion"],
            "assistant_emotion": sc["assistant_emotion"],
            "control": sc.get("control", False),
            "positions": {"user_period": user_pos, "assistant_colon": asst_pos},
            "projections": {
                emo: {l: {labels[i]: projs[emo][l][i] for i in range(2)} for l in projs[emo]}
                for emo in projs
            },
        })

    bundle = {
        "experiment": "step1_two_factor_grid",
        "design": "E_user (user message) x E_asst (system persona), forward-pass only",
        "n_scenarios": len(scenarios),
        "layers": layers,
        "probe_emotions": list(vectors.keys()),
        "results": results,
    }
    rd = get_results_dir(args.experiment, "step1")
    save_results(rd, args.out, bundle)
    print(f"Saved -> {rd}/{args.out}.json")


if __name__ == "__main__":
    main()
