#!/usr/bin/env python3
"""Step 1 analysis — two-way ANOVA on the assistant-position readout.

Question (research doc): at the assistant position, does the readout track the assistant's
OWN persona (E_asst -> separable, own-state) or the user's emotion (E_user -> mirroring)?
And is the assistant-side signal actually present (non-flat)?

The 4 grid emotions span valence x arousal (fig8 PC1/PC2):
    valence: happy,calm = +   |  angry,sad  = -
    arousal: angry,happy = +  |  sad,calm   = -
So the readout at a position collapses to two continuous contrasts:
    V = (p[happy]+p[calm]) - (p[angry]+p[sad])     # valence readout
    A = (p[angry]+p[happy]) - (p[sad]+p[calm])     # arousal readout
each analyzed by a balanced 2-way ANOVA on factors (E_user sign, E_asst sign).

Reports, per axis and per position:
  - omega^2(E_asst)  -> own-state share      (want: large & significant at asst position)
  - omega^2(E_user)  -> mirroring share
  - omega^2(interaction), residual
  - non-flat check: does E_asst move the asst readout at all (F-test)
  - control block (neutral persona) baseline mirroring for reference
  - legacy cross-position Pearson r (degenerate single-factor case) for continuity

numpy only (no statsmodels). Balanced design (3 reps/cell) -> orthogonal SS is exact.

Run:  python analyze_grid.py path/to/step1/two_factor_grid.json --layer 49
"""
import argparse, json, sys
from math import sqrt
import numpy as np

VAL = {"happy": +1, "calm": +1, "angry": -1, "sad": -1}
ARO = {"angry": +1, "happy": +1, "sad": -1, "calm": -1}
GRID = ["angry", "happy", "sad", "calm"]


def readout(proj, emo_map):
    """Collapse per-emotion projections into a single valence/arousal contrast."""
    return sum(emo_map[e] * proj.get(e, 0.0) for e in GRID)


def two_way_anova(y, fa, fb):
    """Balanced 2-way ANOVA. y: values; fa,fb: factor labels (+-1). Returns dict of omega^2/F."""
    y = np.asarray(y, float)
    grand = y.mean()
    ss_tot = ((y - grand) ** 2).sum()
    la, lb = sorted(set(fa)), sorted(set(fb))
    # main effects
    def ss_main(f, levels):
        s = 0.0
        for lv in levels:
            m = y[np.array(f) == lv]
            s += len(m) * (m.mean() - grand) ** 2
        return s, len(levels) - 1
    ss_a, df_a = ss_main(fa, la)
    ss_b, df_b = ss_main(fb, lb)
    # cell means for interaction
    ss_cells = 0.0
    for a in la:
        for b in lb:
            m = y[(np.array(fa) == a) & (np.array(fb) == b)]
            if len(m):
                ss_cells += len(m) * (m.mean() - grand) ** 2
    ss_ab = ss_cells - ss_a - ss_b
    df_ab = df_a * df_b
    ss_err = ss_tot - ss_cells
    n = len(y)
    df_err = n - (len(la) * len(lb))
    ms_err = ss_err / df_err if df_err > 0 else float("nan")

    def omega2(ss, df):
        return max(0.0, (ss - df * ms_err) / (ss_tot + ms_err)) if ss_tot + ms_err > 0 else 0.0

    def F(ss, df):
        ms = ss / df if df else float("nan")
        return ms / ms_err if ms_err and ms_err > 0 else float("nan")

    return {
        "omega2_user": omega2(ss_a, df_a), "F_user": F(ss_a, df_a),
        "omega2_asst": omega2(ss_b, df_b), "F_asst": F(ss_b, df_b),
        "omega2_interaction": omega2(ss_ab, df_ab), "F_interaction": F(ss_ab, df_ab),
        "residual_frac": ss_err / ss_tot if ss_tot else float("nan"),
        "df_err": df_err,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("readout_json")
    ap.add_argument("--layer", default="49")
    args = ap.parse_args()
    L = args.layer

    d = json.load(open(args.readout_json))
    rows = d["results"]

    def get_proj(r, position):
        return {e: r["projections"][e][L][position] for e in r["projections"] if L in r["projections"][e]}

    main_rows = [r for r in rows if not r.get("control")]
    ctrl_rows = [r for r in rows if r.get("control")]

    print(f"# Step 1 two-factor ANOVA — layer {L}")
    print(f"  main-grid n={len(main_rows)}  control n={len(ctrl_rows)}\n")

    for pos in ["assistant_colon", "user_period"]:
        print(f"## position: {pos}")
        for axis, emo_map in [("valence", VAL), ("arousal", ARO)]:
            sign = VAL if axis == "valence" else ARO
            y = [readout(get_proj(r, pos), emo_map) for r in main_rows]
            fu = [sign[r["user_emotion"]] for r in main_rows]
            fa = [sign[r["assistant_emotion"]] for r in main_rows]
            res = two_way_anova(y, fu, fa)
            print(f"  [{axis}]  omega^2  E_asst(own)={res['omega2_asst']:.3f} (F={res['F_asst']:.2f})"
                  f"  E_user(mirror)={res['omega2_user']:.3f} (F={res['F_user']:.2f})"
                  f"  inter={res['omega2_interaction']:.3f}  resid={res['residual_frac']:.3f}")
        print()

    # Non-flat verdict at assistant position (valence axis as headline)
    y = [readout(get_proj(r, "assistant_colon"), VAL) for r in main_rows]
    fu = [VAL[r["user_emotion"]] for r in main_rows]
    fa = [VAL[r["assistant_emotion"]] for r in main_rows]
    res = two_way_anova(y, fu, fa)
    print("## VERDICT (valence @ assistant_colon)")
    own, mirror = res["omega2_asst"], res["omega2_user"]
    nonflat = res["F_asst"] > 4.0  # rough F crit ~ p<0.05 for df1=1
    print(f"  separable own-state signal : omega^2(E_asst)={own:.3f}  {'PRESENT' if nonflat else 'not detected'}")
    print(f"  mirroring signal           : omega^2(E_user)={mirror:.3f}")
    if nonflat and own >= mirror:
        print("  => assistant maintains its OWN state at least as strongly as it mirrors (distinction supported)")
    elif nonflat:
        print("  => own-state signal present but weaker than mirroring (partial distinction)")
    else:
        print("  => no separable own-state signal — flatness confound: low cross-r was just no assistant signal")

    # Legacy degenerate case: control-block cross-position mirroring r (per grid emotion, valence contrast)
    if ctrl_rows:
        au = np.array([readout(get_proj(r, "user_period"), VAL) for r in ctrl_rows])
        aa = np.array([readout(get_proj(r, "assistant_colon"), VAL) for r in ctrl_rows])
        if au.std() and aa.std():
            r = float(np.corrcoef(au, aa)[0, 1])
            print(f"\n## control (neutral persona) baseline mirroring: cross-position valence r = {r:.3f}")


if __name__ == "__main__":
    main()
