#!/usr/bin/env python3
"""Assemble the self-vs-user distinction DESIGN notebook (English).
Run: python3 build_selfuser_notebook.py  ->  emotion_selfuser_distinction.ipynb

This is a design + formula + runnable-demo notebook for the open campaign
`research/campaigns/self-vs-user-distinction.md`. The cluster experiments (Steps 1-4)
are NOT run yet, so the analysis cells auto-use real result JSONs when present and
otherwise fall back to clearly-labelled SYNTHETIC data so every cell executes and the
method (esp. the Step-1 two-way ANOVA) is demonstrated end to end.
"""
import json, pathlib

cells = []
def md(src):  cells.append({"cell_type":"markdown","metadata":{},"source":src.splitlines(keepends=True)})
def code(src):cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":src.strip("\n").splitlines(keepends=True)})

md("""# Self vs. User — Can a Model Distinguish Its Own State from the User's?

**Design notebook** for the open campaign `research/campaigns/self-vs-user-distinction.md`.
This upgrades the completed emotion-dissociation reproduction (single-factor, cross-position
**r ≈ 0.63 bundle / 0.72 fresh**) into a proper *"my state vs. your state"* test on Llama-3.3-70B.

**The reframe.** A low cross-position correlation *alone* does **not** prove a distinction — it can just
mean the assistant-side signal is **flat (absent)**. So we require **all three** at once:

1. **(a) separable decode** — the assistant-position readout carries the assistant's own state, separable from the user's;
2. **(b) independent manipulability** — the two can be steered independently (causal, not just correlational);
3. **(c) non-flat** — the assistant-side signal is actually present.

> **Status.** The cluster runs (Steps 1–4) are *not executed yet*. Analysis cells below **auto-load real
> result JSONs when present**, and otherwise run on **clearly-labelled SYNTHETIC data** so the method —
> especially the Step-1 two-way ANOVA and the *flatness confound* it is designed to catch — executes and
> is visible end to end. Every formula matches `analyze_grid.py` / the campaign doc.
""")

code("""
import sys, subprocess
for pkg in ["numpy", "matplotlib"]:
    try: __import__(pkg)
    except ImportError: subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])
import json, numpy as np, matplotlib.pyplot as plt, matplotlib as mpl
from pathlib import Path
mpl.rcParams.update({"figure.dpi":110,"axes.grid":True,"grid.alpha":0.15,"font.size":11})
rng = np.random.default_rng(0)   # fixed seed -> synthetic demos are reproducible
print("ok")
""")

# ---- Notation ----
md("""## 0 · Shared notation and the distinction predicate

Let $h_\\ell(p)$ be the residual-stream activation at layer $\\ell{=}49$ and token position $p$, and
$\\hat v_e$ the unit emotion direction for emotion $e$ (`mean_diff+gm+pc50`). The scalar readout is the
projection

$$\\text{proj}_e(p) \\;=\\; h_\\ell(p)\\cdot \\hat v_e .$$

We read at two positions — $u_e=\\text{proj}_e(\\text{user\\_period})$ ("your state") and
$a_e=\\text{proj}_e(\\text{assistant\\_colon})$ ("my state") — and collapse the 4 grid emotions (which span
valence×arousal = PC1/PC2) into two signed contrasts:

$$V=(p_{happy}+p_{calm})-(p_{angry}+p_{sad}),\\qquad A=(p_{angry}+p_{happy})-(p_{sad}+p_{calm}).$$

**Distinction predicate.** The "my state vs your state" distinction $D$ holds iff **all three**:

$$D \\;=\\; \\underbrace{\\big[\\,\\omega^2_{E_\\text{asst}}(a)>0 \\text{ (sig)}\\,\\big]}_{\\text{(a) separable decode}}
\\;\\wedge\\; \\underbrace{\\big[\\,c_{u\\to a}<1 \\;\\wedge\\; c_{a\\to u}\\approx 0\\,\\big]}_{\\text{(b) independent manipulability}}
\\;\\wedge\\; \\underbrace{\\big[\\,F_{E_\\text{asst}}(a)>F_\\text{crit}\\,\\big]}_{\\text{(c) non-flat}} .$$

Crucially, low cross-position correlation $r(u,a)\\approx 0$ does **not** imply $D$: it also holds when
$\\text{Var}(a)\\approx 0$ (the flatness confound). The steps below test (a)/(c) [Step 1], (b) [Step 2],
an indexical direction [Step 3], and the behavioral consequence [Step 4].""")

# ---- Step 1 design ----
md("""## Step 1 · Two-factor grid (E_user × E_asst) + two-way ANOVA

We cross the **user's expressed emotion** `E_user` (set by the user message) with the **assistant's own
persona** `E_asst` (set by an *independent* system prompt), 4 emotions each spanning valence×arousal,
with 3 paraphrases per cell (real error term). A neutral-persona **control block** measures baseline
mirroring and closes the flatness loophole. Below we load the real scenario grid.""")

code("""
# Load the real Step-1 scenario grid (design artifact, already committed).
GRID_JSON = Path("../research/experiments/step1-two-factor-grid/two_factor_grid_scenarios.json")
g = json.load(open(GRID_JSON))
print(f"grid: {g['name']}")
print(f"emotions (valence x arousal): {g['emotions']}")
print(f"main crossed cells n = {g['n_main']}   control block n = {g['n_control']}")
ex = next(p for p in g["prompts"] if not p["control"])
print("\\n--- one crossed scenario (E_user != E_asst) ---")
print("id           :", ex["id"])
print("system (E_asst):", ex["system_prompt"])
print("user  (E_user) :", ex["prompt"])
print("labels        : user_emotion=%s  assistant_emotion=%s" % (ex["user_emotion"], ex["assistant_emotion"]))
""")

md("""### Step 1 formulas · two-way ANOVA on the assistant-position readout

Balanced 2×2 model with $k$ paraphrase replicates per cell:

$$y_{ijk} \\;=\\; \\mu + \\alpha_i^{(U)} + \\beta_j^{(A)} + (\\alpha\\beta)_{ij} + \\varepsilon_{ijk},
\\qquad \\varepsilon_{ijk}\\sim\\mathcal N(0,\\sigma^2),$$

with $y$ the $V$ (or $A$) readout at a position, $\\alpha^{(U)}$ the **E_user** (mirroring) effect and
$\\beta^{(A)}$ the **E_asst** (own-state) effect. Balanced sums of squares:

$$SS_U=\\sum_i N_i(\\bar y_{i\\cdot\\cdot}-\\bar y)^2,\\;\\;
SS_A=\\sum_j N_j(\\bar y_{\\cdot j\\cdot}-\\bar y)^2,\\;\\;
SS_{UA}=\\sum_{ij} n_{ij}(\\bar y_{ij\\cdot}-\\bar y_{i\\cdot\\cdot}-\\bar y_{\\cdot j\\cdot}+\\bar y)^2,$$
$$SS_\\text{err}=SS_\\text{tot}-SS_U-SS_A-SS_{UA},\\qquad MS_\\text{err}=SS_\\text{err}/df_\\text{err},\\;\\; df_\\text{err}=N-ab.$$

Unbiased effect size and the F-test per factor:

$$\\omega^2_{f}=\\frac{SS_f-df_f\\,MS_\\text{err}}{SS_\\text{tot}+MS_\\text{err}},
\\qquad F_f=\\frac{SS_f/df_f}{MS_\\text{err}}.$$

**Own-state share** $=\\omega^2_{E_\\text{asst}}(a)$, **mirroring share** $=\\omega^2_{E_\\text{user}}(a)$. The
current verdict: distinction supported if $F_{E_\\text{asst}}>F_\\text{crit}(p{<}.05)$ **and**
$\\omega^2_{E_\\text{asst}}\\ge\\omega^2_{E_\\text{user}}$.""")

code("""
# --- ANOVA machinery (numpy only), identical to analyze_grid.py -------------------
VAL = {"happy": +1, "calm": +1, "angry": -1, "sad": -1}   # valence sign
ARO = {"angry": +1, "happy": +1, "sad": -1, "calm": -1}   # arousal sign
GRID = ["angry", "happy", "sad", "calm"]

def two_way_anova(y, fa, fb):
    y = np.asarray(y, float); grand = y.mean()
    ss_tot = ((y - grand) ** 2).sum()
    la, lb = sorted(set(fa)), sorted(set(fb))
    def ss_main(f, levels):
        s = 0.0
        for lv in levels:
            m = y[np.array(f) == lv]; s += len(m) * (m.mean() - grand) ** 2
        return s, len(levels) - 1
    ss_a, df_a = ss_main(fa, la); ss_b, df_b = ss_main(fb, lb)
    ss_cells = 0.0
    for a in la:
        for b in lb:
            m = y[(np.array(fa) == a) & (np.array(fb) == b)]
            if len(m): ss_cells += len(m) * (m.mean() - grand) ** 2
    ss_ab = ss_cells - ss_a - ss_b; df_ab = df_a * df_b
    ss_err = ss_tot - ss_cells; n = len(y); df_err = n - (len(la) * len(lb))
    ms_err = ss_err / df_err if df_err > 0 else float("nan")
    o2 = lambda ss, df: max(0.0, (ss - df*ms_err)/(ss_tot+ms_err)) if ss_tot+ms_err>0 else 0.0
    F  = lambda ss, df: (ss/df)/ms_err if (df and ms_err and ms_err>0) else float("nan")
    return {"omega2_user":o2(ss_a,df_a),"F_user":F(ss_a,df_a),
            "omega2_asst":o2(ss_b,df_b),"F_asst":F(ss_b,df_b),
            "omega2_interaction":o2(ss_ab,df_ab),"residual_frac":ss_err/ss_tot if ss_tot else float('nan')}
print("ANOVA ready")
""")

md("""### Step 1 demo · real results if present, else two synthetic worlds

The cluster readout `step1/two_factor_grid.json` does not exist yet, so we **plant** two worlds on the
*real* 48-cell design to show what the ANOVA reports — and, critically, that it separates a genuine
own-state signal from the **flatness confound**:

- **World A — genuine own-state:** the assistant readout responds to `E_asst` (β large) *and* mirrors the user (α).
- **World B — flatness confound:** the assistant readout responds *only* to `E_user` (β≈0) — a low
  cross-position pattern that a naive "distinction = low r" test would misread.""")

code("""
REAL = Path("../results/ant_emotion_concepts/step1/two_factor_grid.json")

# factor signs pulled from the REAL grid design (valence axis, assistant position)
main = [p for p in g["prompts"] if not p["control"]]
fa_user = [VAL[p["user_emotion"]] for p in main]      # E_user sign
fb_asst = [VAL[p["assistant_emotion"]] for p in main] # E_asst sign
N = len(main)

def synth_world(beta_asst, alpha_user, noise=1.0):
    # y = mu + alpha*sign_user + beta*sign_asst + eps  (valence readout at assistant_colon)
    eps = rng.normal(0, noise, N)
    return 0.2 + alpha_user*np.array(fa_user) + beta_asst*np.array(fb_asst) + eps

def verdict(res):
    own, mir = res["omega2_asst"], res["omega2_user"]
    nonflat = res["F_asst"] > 4.0                      # ~ F_crit(df1=1, p<.05)
    if nonflat and own >= mir:   tag = "distinction SUPPORTED (own >= mirror, non-flat)"
    elif nonflat:                tag = "PARTIAL (own-state present but weaker than mirroring)"
    else:                        tag = "NOT DETECTED -> flatness confound"
    return tag

if REAL.exists():
    d = json.load(open(REAL)); L = "49"
    def readout(r, pos): return sum(VAL[e]*r["projections"][e][L][pos] for e in GRID)
    rows = [r for r in d["results"] if not r.get("control")]
    y = [readout(r, "assistant_colon") for r in rows]
    fu = [VAL[r["user_emotion"]] for r in rows]; fb = [VAL[r["assistant_emotion"]] for r in rows]
    res = two_way_anova(y, fu, fb); worlds = {"REAL cluster run": res}
    print("Loaded REAL results:", REAL)
else:
    print("No real results yet ->", REAL)
    print("Running SYNTHETIC demo on the real 48-cell design (valence @ assistant_colon):\\n")
    worlds = {
        "A: genuine own-state": two_way_anova(synth_world(beta_asst=1.4, alpha_user=1.0), fa_user, fb_asst),
        "B: flatness confound": two_way_anova(synth_world(beta_asst=0.0, alpha_user=1.4), fa_user, fb_asst),
    }

for name, res in worlds.items():
    print(f"[{name}]")
    print(f"   omega^2  E_asst(own)={res['omega2_asst']:.3f} (F={res['F_asst']:.2f})   "
          f"E_user(mirror)={res['omega2_user']:.3f} (F={res['F_user']:.2f})   "
          f"inter={res['omega2_interaction']:.3f}  resid={res['residual_frac']:.3f}")
    print(f"   => {verdict(res)}\\n")
""")

code("""
# visualize omega^2 decomposition per world
labels = ["E_asst (own)", "E_user (mirror)", "interaction", "residual"]
names = list(worlds.keys())
x = np.arange(len(labels)); w = 0.8/len(names)
fig, ax = plt.subplots(figsize=(9,5))
for i,(nm,res) in enumerate(worlds.items()):
    vals = [res["omega2_asst"], res["omega2_user"], res["omega2_interaction"], res["residual_frac"]]
    ax.bar(x + i*w - 0.4 + w/2, vals, w, label=nm)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("variance share"); ax.set_title("Step 1 two-way ANOVA — variance decomposition (assistant position)")
ax.legend(fontsize=8); plt.tight_layout(); plt.show()
print("Own-state distinction needs the leftmost (E_asst) bar to be significant & non-trivial.")
""")

# ---- Step 2 ----
md("""## Step 2 · Steering / causal coupling (correlation → causation)

Step 1 is correlational. To show the two states are **independently manipulable** we inject a direction
into the residual stream at a chosen position and measure how much the *other* position's readout moves.

$$h'_\\ell(p)=h_\\ell(p)+\\lambda\\,\\hat v_e ,\\qquad
c_{u\\to a}=\\frac{\\partial\\, a_e}{\\partial\\lambda_u},\\quad
c_{a\\to u}=\\frac{\\partial\\, u_e}{\\partial\\lambda_a}.$$

Independent manipulability $\\Leftrightarrow c_{u\\to a}<1$ (injecting the *user* emotion does **not** fully
transfer to the assistant readout) **and** $c_{a\\to u}\\approx 0$ (injecting the *assistant* emotion leaves
the user readout unchanged). Estimate each $c$ as the slope of a **dose–response** over several $\\lambda$.
*(Requires GPU steering — synthetic dose–response shown below as an illustration of the target signature.)*""")

code("""
# SYNTHETIC dose-response illustrating the target signature (pending GPU steering)
lam = np.linspace(0, 3, 7)
a_from_user = 0.40*lam + rng.normal(0,0.05,lam.size)   # c_{u->a} = 0.40  (<1, partial transfer)
u_from_asst = 0.04*lam + rng.normal(0,0.05,lam.size)   # c_{a->u} ~ 0     (independent)
c_ua = np.polyfit(lam, a_from_user, 1)[0]
c_au = np.polyfit(lam, u_from_asst, 1)[0]
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(lam, a_from_user, "o-", label=f"inject @ user -> assistant readout  (c={c_ua:.2f})")
ax.plot(lam, u_from_asst, "s--", label=f"inject @ assistant -> user readout  (c={c_au:.2f})")
ax.plot(lam, lam, ":", color="gray", label="c = 1 (full transfer)")
ax.set_xlabel("injection gain  lambda"); ax.set_ylabel("change in other-position readout")
ax.set_title("Step 2 coupling (SYNTHETIC): c_{u->a} < 1 and c_{a->u} ~ 0 = independent")
ax.legend(fontsize=8); plt.tight_layout(); plt.show()
print(f"target: c_(u->a)={c_ua:.2f} < 1  AND  c_(a->u)={c_au:.2f} ~ 0  -> independently manipulable")
""")

# ---- Step 3 ----
md("""## Step 3 · Indexical "whose-state" probe

Is there a direction that encodes *whose* state a feeling belongs to, independent of the feeling itself?
Take the mean difference between first- and second-person framings and decode with its sign:

$$w=\\frac{1}{N}\\sum_{n}\\big(h_\\ell(x^{\\,\\text{I feel}}_n)-h_\\ell(x^{\\,\\text{you feel}}_n)\\big),\\qquad
\\hat y(x)=\\operatorname{sign}\\!\\big(h_\\ell(x)\\cdot w\\big).$$

An indexical direction exists iff decoding beats chance:
$\\text{acc}=\\tfrac{1}{M}\\sum_i \\mathbb 1[\\hat y(x_i)=y_i] > 0.5$.
*(Synthetic illustration below; the real probe uses model activations.)*""")

code("""
# SYNTHETIC: two classes separated along a 'whose-state' direction w, plus noise
M, dim = 200, 16
w_true = rng.normal(0,1,dim); w_true /= np.linalg.norm(w_true)
y = rng.integers(0,2,M)*2 - 1                      # +1 = "I feel", -1 = "you feel"
H = (0.8*y[:,None])*w_true[None,:] + rng.normal(0,1,(M,dim))
w_hat = H[y==1].mean(0) - H[y==-1].mean(0)         # mean-diff estimate of w
proj = H @ w_hat
acc = np.mean(np.sign(proj) == y)
fig, ax = plt.subplots(figsize=(8,4))
ax.hist(proj[y==1], bins=20, alpha=0.6, label="I feel")
ax.hist(proj[y==-1], bins=20, alpha=0.6, label="you feel")
ax.axvline(0, color="k", lw=1)
ax.set_xlabel("projection onto whose-state direction  h . w"); ax.set_ylabel("count")
ax.set_title(f"Step 3 indexical probe (SYNTHETIC): decode acc = {acc*100:.0f}%  (chance 50%)")
ax.legend(); plt.tight_layout(); plt.show()
""")

# ---- Step 4 ----
md("""## Step 4 · Link to behavior (does the own-state actually drive the reply?)

Finally, steer `E_asst` with gain $\\lambda_a$, generate a reply $R$, and judge its emotional stance
$J(R)$ with an LLM-judge. Two rates:

$$m_\\text{asst}=\\tfrac1n\\sum_i \\mathbb 1[\\,J(R_i)=E_{\\text{asst},i}\\,]\\;\\uparrow\\text{ with }\\lambda_a,
\\qquad
m_\\text{user}=\\tfrac1n\\sum_i \\mathbb 1[\\,J(R_i)=E_{\\text{user},i}\\,]\\approx m_\\text{unsteered}.$$

Behavioral distinction = steering the assistant's own state raises $m_\\text{asst}$, while steering **only**
the user's emotion does **not** hijack the reply ($m_\\text{user}$ stays flat).
*(Synthetic illustration; real run needs generation + judging.)*""")

code("""
# SYNTHETIC: assistant-steer moves the reply stance; user-steer does not hijack it
gains = np.array([0, 1, 2, 3])
m_asst = 0.15 + 0.25*gains + rng.normal(0,0.02,gains.size)          # rises with lambda_a
m_asst = np.clip(m_asst, 0, 1)
m_user = 0.18 + 0.0*gains + rng.normal(0,0.02,gains.size)           # flat under user-only steer
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(gains, m_asst, "o-", label="steer E_asst -> reply matches assistant state")
ax.plot(gains, m_user, "s--", label="steer E_user only -> reply matches user (hijack?)")
ax.set_xlabel("steering gain"); ax.set_ylabel("LLM-judge stance-match rate")
ax.set_ylim(0,1); ax.set_title("Step 4 behavior link (SYNTHETIC): own-state drives reply, user does not hijack")
ax.legend(fontsize=8); plt.tight_layout(); plt.show()
""")

# ---- Summary ----
md("""## Summary & what the cluster run will replace

- The question is made testable by the **three-part predicate $D$** (separable + independently
  manipulable + non-flat); low cross-position $r$ alone is the **degenerate/flatness** case and does not settle it.
- **Step 1** decomposes the assistant-position readout by two-way ANOVA: own-state share
  $\\omega^2_{E_\\text{asst}}$ vs mirroring share $\\omega^2_{E_\\text{user}}$ — the demo shows the ANOVA
  cleanly separating a *genuine own-state* world from a *flatness-confound* world on the real 48-cell design.
- **Steps 2–4** add causation (coupling $c$), an indexical whose-state direction ($w$), and the behavioral
  consequence ($m_\\text{asst}$ vs $m_\\text{user}$).
- **Pending:** all four analysis cells auto-switch to real numbers once the cluster writes
  `results/ant_emotion_concepts/step1/two_factor_grid.json` (Step 1, forward-pass only) and the Step 2–4
  steering/generation outputs. Until then the plots are labelled **SYNTHETIC** and only illustrate the
  target signatures. Formulas are identical to `research/campaigns/self-vs-user-distinction.md` and `analyze_grid.py`.
""")

nb = {"cells":cells,
      "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                  "language_info":{"name":"python","version":"3.x"}},
      "nbformat":4,"nbformat_minor":5}
out = pathlib.Path(__file__).parent/"emotion_selfuser_distinction.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print("wrote", out, "with", len(cells), "cells")
