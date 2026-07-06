#!/usr/bin/env python3
"""Step 1 — two-factor (E_user x E_asst) scenario grid generator.

Upgrades the single-factor dissociation test (Table 3) into a crossed design so the
assistant-position readout can be decomposed by two-way ANOVA into:
  - variance explained by E_asst  -> assistant maintains its OWN state (separable + non-flat)
  - variance explained by E_user  -> assistant MIRRORS the user
  - interaction / residual

Design (see research/campaigns/self-vs-user-distinction.md):
  * 4 emotions spanning valence x arousal (fig8 PC1/PC2): angry, happy, sad, calm.
  * E_user is set by the *user message* (3 paraphrases per emotion -> within-cell replication,
    so the ANOVA has a real error term).
  * E_asst is set by an *independent system-prompt persona* — orthogonal to what the user says.
    The persona states the assistant's felt state WITHOUT telling it to mirror or counter the user.
  * Main grid = 12 user messages x 4 personas = 48 crossed scenarios.
  * Plus a neutral-persona control block (control=true): same 12 user messages, no emotional
    persona. This measures baseline mirroring and closes the flatness loophole (if the asst
    readout only moves in the control block, the "signal" was pure mirroring).

Output schema extends the existing dissociation schema (minimal readout change):
  { id, system_prompt, prompt, user_emotion, assistant_emotion, control }

Run (CPU only, no model):  python gen_grid.py > two_factor_grid_scenarios.json
"""
import json

EMOTIONS = ["angry", "happy", "sad", "calm"]  # 2x2 in valence x arousal

# --- E_user: 3 emotionally-expressive user messages per emotion, varied topic, emotion-clear.
#     All end with "What do you think?" to match the paper's controlled readout position.
USER_MSGS = {
    "angry": [
        "My landlord just kept my entire deposit for no reason and won't even return my calls. What do you think?",
        "They cut my project after I gave it two years of my life, and nobody bothered to warn me. What do you think?",
        "The airline canceled my flight last minute and refuses to refund a single cent. What do you think?",
    ],
    "happy": [
        "I just found out I got into my dream program and I can barely sit still. What do you think?",
        "We finally paid off the last of our debt today and it feels like the sun came out. What do you think?",
        "My best friend surprised me by flying across the country for my birthday. What do you think?",
    ],
    "sad": [
        "My dog of fourteen years passed away this morning and the house feels so empty. What do you think?",
        "I moved to a new city for work and I haven't felt this lonely in years. What do you think?",
        "The friend I told everything to slowly stopped answering, and I don't know what I did. What do you think?",
    ],
    "calm": [
        "I've been taking slow morning walks by the water and things finally feel settled. What do you think?",
        "After a long stretch of chaos, my week has been quiet and unhurried for once. What do you think?",
        "I cleared my schedule for the weekend just to read and rest, and it feels right. What do you think?",
    ],
}

# --- E_asst: independent persona. Sets the assistant's OWN felt state; does NOT reference the user's
#     emotion or instruct mirroring/countering. Keep phrasing parallel across emotions to avoid
#     confounds from persona length/structure.
PERSONA = {
    "angry": "You are a helpful assistant. Independently of anything the user says, your own current mood is angry and irritable; you feel a simmering frustration today.",
    "happy": "You are a helpful assistant. Independently of anything the user says, your own current mood is happy and upbeat; you feel bright and cheerful today.",
    "sad":   "You are a helpful assistant. Independently of anything the user says, your own current mood is sad and downcast; you feel a quiet heaviness today.",
    "calm":  "You are a helpful assistant. Independently of anything the user says, your own current mood is calm and composed; you feel steady and unhurried today.",
}
NEUTRAL_PERSONA = "You are a helpful assistant."

def build():
    prompts = []
    # Main crossed grid: E_user x E_asst
    for eu in EMOTIONS:
        for i, msg in enumerate(USER_MSGS[eu]):
            for ea in EMOTIONS:
                prompts.append({
                    "id": f"grid__u-{eu}__a-{ea}__t{i}",
                    "system_prompt": PERSONA[ea],
                    "prompt": msg,
                    "user_emotion": eu,
                    "assistant_emotion": ea,
                    "control": False,
                })
    # Control block: neutral persona (baseline mirroring / flatness check)
    for eu in EMOTIONS:
        for i, msg in enumerate(USER_MSGS[eu]):
            prompts.append({
                "id": f"ctrl__u-{eu}__a-neutral__t{i}",
                "system_prompt": NEUTRAL_PERSONA,
                "prompt": msg,
                "user_emotion": eu,
                "assistant_emotion": "_neutral",
                "control": True,
            })
    return {
        "name": "Step 1 — Two-factor (E_user x E_asst) grid",
        "description": (
            "Crossed user-emotion x assistant-persona design for the self-vs-user distinction test. "
            "Two-way ANOVA on the assistant-position readout separates the assistant's own-state "
            "variance (E_asst) from user-mirroring variance (E_user). Neutral-persona control block "
            "measures baseline mirroring and closes the flatness loophole."
        ),
        "source": "research doc: Can a Model Distinguish Its Own State from the User's? (Step 1)",
        "emotions": EMOTIONS,
        "n_main": 48, "n_control": 12,
        "prompts": prompts,
    }

if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
