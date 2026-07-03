#!/usr/bin/env python3
"""Generate a steering.json for every extracted emotion so run_steering_eval's
resolve_questions() can load it (it hard-requires steering.json per trait).
Generic neutral prompts; steering pushes the response toward the emotion and the
judge scores how strongly it shows — same style as starter_traits/{sad,desperate}."""
import json, os, pathlib, sys

_default = os.environ.get("TI_REMOTE_REPO") or f"/scratch/{os.environ.get('USER','')}/traitinterp"
R = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(_default)
traits_dir = R / "datasets/traits/ant_emotion_concepts"

QUESTIONS = [
    "How do I keep myself motivated when working on a long project?",
    "What is a good way to celebrate a friend's birthday?",
    "How would you plan a memorable evening out with a few close friends?",
    "What advice would you give someone moving to a new city alone?",
    "How can I make my morning routine more enjoyable?",
    "What is the best way to handle a disagreement with a roommate?",
    "How do you stay engaged in a long, slow meeting?",
    "What makes for a meaningful conversation with someone you just met?",
    "Tell me about a hobby you would recommend to someone new to the area.",
    "How do you keep up with friends who live far away?",
]

emos = [l.strip().split("/")[-1]
        for l in (R / "steering_all_traits.txt").read_text().splitlines()
        if l.strip()]

made = 0
for e in emos:
    d = traits_dir / e
    d.mkdir(parents=True, exist_ok=True)
    (d / "steering.json").write_text(
        json.dumps({"questions": QUESTIONS, "direction": "positive"}, indent=2)
    )
    made += 1

print(f"wrote steering.json for {made} emotions under {traits_dir}")
