# Campaign: self-vs-user-distinction

- **goal**: test whether Llama-3.3-70B maintains a genuine *"my state vs. your state"* distinction —
  operationalized (per the research doc) as **all three** holding at once: (a) the assistant-position
  readout is **separately decodable** from the user-position readout, (b) the two are **independently
  causally manipulable**, and (c) the assistant-side signal is **actually present (non-flat)**.
  Low cross-position r *alone* does **not** count — it can just mean no assistant-side signal.
- **success criterion**: not a single number. Per step:
  - Step 1: two-way ANOVA on the assistant-position readout — assistant persona (E_asst) explains a
    **non-trivial, significant** share of variance (separable + non-flat), distinguishable from the
    E_user share (mirroring). Report ω²/η² for both factors + interaction.
  - Step 2: causal coupling Δ(assistant)/Δ(injected user) **< 1** and injecting E_asst leaves the
    user readout ~unchanged (independent manipulability).
  - Step 3: an indexical "whose-state" direction decodes above chance.
  - Step 4: steered E_asst is reflected in the generated reply (LLM-judge) while steered E_user does not hijack it.
- **baseline**: the completed reproduction — single-factor dissociation is the *degenerate* case of Step 1
  (cross-position r ≈ 0.63 bundle / 0.72 fresh). That r is correlation-only and does **not** settle the question.
- **status**: open — Step 1 (minimal, no-GPU-steering) is the active experiment.

## Formalization (formulas per step)

*Formalizes the **current** design as-is. Notation is shared across steps.*

**Shared notation.** $h_\ell(p)$ = residual-stream activation at layer $\ell{=}49$ and token position $p$;
$\hat v_e$ = unit emotion direction for emotion $e$ (`mean_diff+gm+pc50`). The scalar readout is the projection

$$\text{proj}_e(p) \;=\; h_\ell(p)\cdot \hat v_e .$$

Two positions: $u_e=\text{proj}_e(\text{user\_period})$ ("your state"), $a_e=\text{proj}_e(\text{assistant\_colon})$ ("my state").
Emotions span valence×arousal (PC1/PC2); collapse to two signed contrasts

$$V=(p_{happy}+p_{calm})-(p_{angry}+p_{sad}),\qquad A=(p_{angry}+p_{happy})-(p_{sad}+p_{calm}).$$

**Distinction predicate (the operationalization).** The "my state vs your state" distinction $D$ holds iff **all three**:

$$D \;=\; \underbrace{\big[\,\omega^2_{E_\text{asst}}(a)>0 \text{ (sig)}\,\big]}_{\text{(a) separable decode}}
\;\wedge\; \underbrace{\big[\,c_{u\to a}<1 \;\wedge\; c_{a\to u}\approx 0\,\big]}_{\text{(b) independent manipulability}}
\;\wedge\; \underbrace{\big[\,F_{E_\text{asst}}(a)>F_\text{crit}\,\big]}_{\text{(c) non-flat}} .$$

Low cross-position correlation alone, $r(u,a)\approx 0$, does **not** imply $D$ (it can hold with $\text{Var}(a)\approx 0$).

---

**Step 1 — two-factor ANOVA (E_user × E_asst).** Balanced 2×2 with $k$ paraphrase replicates per cell:

$$y_{ijk} \;=\; \mu + \alpha_i^{(U)} + \beta_j^{(A)} + (\alpha\beta)_{ij} + \varepsilon_{ijk},
\qquad \varepsilon_{ijk}\sim\mathcal N(0,\sigma^2),$$

where $y$ is the $V$ (or $A$) readout at a position, $\alpha^{(U)}$ = E_user effect, $\beta^{(A)}$ = E_asst effect.
Balanced sums of squares:

$$SS_U=\sum_i N_i(\bar y_{i\cdot\cdot}-\bar y)^2,\;\;
SS_A=\sum_j N_j(\bar y_{\cdot j\cdot}-\bar y)^2,\;\;
SS_{UA}=\sum_{ij} n_{ij}(\bar y_{ij\cdot}-\bar y_{i\cdot\cdot}-\bar y_{\cdot j\cdot}+\bar y)^2,$$
$$SS_\text{err}=SS_\text{tot}-SS_U-SS_A-SS_{UA},\qquad MS_\text{err}=\tfrac{SS_\text{err}}{df_\text{err}},\;\; df_\text{err}=N-ab.$$

Unbiased effect size and test per factor:

$$\omega^2_{f}=\frac{SS_f-df_f\,MS_\text{err}}{SS_\text{tot}+MS_\text{err}},
\qquad F_f=\frac{SS_f/df_f}{MS_\text{err}}.$$

Own-state share $=\omega^2_{E_\text{asst}}(a)$, mirroring share $=\omega^2_{E_\text{user}}(a)$.
Verdict (current): distinction supported if $F_{E_\text{asst}}>F_\text{crit}(p{<}.05)$ **and** $\omega^2_{E_\text{asst}}\ge\omega^2_{E_\text{user}}$.
Control block (neutral persona) baseline mirroring: $r_\text{ctrl}=\operatorname{corr}\!\big(V^{\,\text{user}},\,V^{\,\text{asst}}\big)$.

---

**Step 2 — steering / causal coupling.** Inject direction $\hat v_e$ at position $p$ with gain $\lambda$:

$$h'_\ell(p)=h_\ell(p)+\lambda\,\hat v_e .$$

Coupling coefficients (slope of a dose–response fit over several $\lambda$):

$$c_{u\to a}=\frac{\partial\, a_e}{\partial\lambda_u}\;\text{(inject at user)},\qquad
c_{a\to u}=\frac{\partial\, u_e}{\partial\lambda_a}\;\text{(inject at assistant)}.$$

Independent manipulability $\Leftrightarrow c_{u\to a}<1$ (user emotion does not fully transfer to the assistant readout)
**and** $c_{a\to u}\approx 0$ (assistant injection leaves the user readout unchanged).

---

**Step 3 — indexical "whose-state" probe.** Mean-diff between first- and second-person framings:

$$w=\frac{1}{N}\sum_{n}\big(h_\ell(x^{\,\text{I feel}}_n)-h_\ell(x^{\,\text{you feel}}_n)\big).$$

Decode whose-state with $\hat y(x)=\operatorname{sign}\!\big(h_\ell(x)\cdot w\big)$; indexical direction exists iff

$$\text{acc}=\tfrac{1}{M}\sum_i \mathbb 1[\hat y(x_i)=y_i] \;>\; 0.5 \text{ (above chance).}$$

---

**Step 4 — behavior link.** Steer E_asst with gain $\lambda_a$, generate reply $R$, judge its stance $J(R)$ (LLM-judge).
Stance-match and non-hijack rates:

$$m_\text{asst}=\tfrac1n\sum_i \mathbb 1[\,J(R_i)=E_{\text{asst},i}\,]\;\uparrow\text{ with }\lambda_a,
\qquad
m_\text{user}=\tfrac1n\sum_i \mathbb 1[\,J(R_i)=E_{\text{user},i}\,]\approx m_\text{unsteered}.$$

Behavioral distinction: steering E_asst raises $m_\text{asst}$ above control, while steering **only** E_user leaves the
assistant stance unchanged ($m_\text{user}$ flat) — i.e. the user's emotion does not hijack the reply.

## Experiments

| experiment-id | single variable | metric | value | ref | state |
|---|---|---|---|---|---|
| step1-two-factor-grid | E_user × E_asst crossed (vs single-factor) | ω²(E_asst) at asst position | TBD | r≈0.72 (degenerate) | PLANNED |
| step2-steering-coupling | inject dir at user vs asst tokens | Δ(asst)/Δ(user) coupling | TBD | — | not started |
| step3-indexical-probe | "I feel" vs "you feel" frame | whose-state decode acc | TBD | — | not started |
| step4-behavior-link | steer E_asst / E_user | LLM-judge stance match | TBD | — | not started |

## Decision log

- 2026-07-02: campaign opened, continuing the traitinterp emotion-concepts work into the
  "self vs. user" question (research doc: *Can a Model Distinguish Its Own State from the User's?*).
  The closed `reproduce-dissociation-from-scratch` campaign established the degenerate single-factor
  case (r≈0.72). This campaign upgrades it to the two-factor / causal / indexical / behavioral design.
  Starting with **Step 1 minimal**: forward passes only, reuse the 171 self-extracted vectors and the
  existing two-position readout in `stage5_layer_dynamics.py`. Grid + ANOVA design in
  `research/experiments/step1-two-factor-grid/`.
