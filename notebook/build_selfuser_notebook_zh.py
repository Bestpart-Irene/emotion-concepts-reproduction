#!/usr/bin/env python3
"""生成中文版 self-vs-user 距离设计 notebook。
运行: python3 build_selfuser_notebook_zh.py  ->  emotion_selfuser_distinction_zh.ipynb
说明: markdown 与 print 用中文;matplotlib 图内文字保留英文(避免缺 CJK 字体渲染成方块)。
这是 research/campaigns/self-vs-user-distinction.md 的设计+公式+可跑示范 notebook。
"""
import json, pathlib

cells = []
def md(src):  cells.append({"cell_type":"markdown","metadata":{},"source":src.splitlines(keepends=True)})
def code(src):cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":src.strip("\n").splitlines(keepends=True)})

md("""# Self vs. User — 模型能区分"自己的状态"和"用户的状态"吗?

open campaign `research/campaigns/self-vs-user-distinction.md` 的**设计笔记本**。它把已完成的情绪解离复现
(单因子、跨位置 **r ≈ 0.63 数据包 / 0.72 从头**)升级为 Llama-3.3-70B 上正经的*"我的状态 vs 你的状态"*检验。

**关键重构。** 跨位置相关低**本身**并不能证明"区分"——它可能只是助手侧信号**平坦(缺失)**。所以我们要求
**三条件同时成立**:

1. **(a) 可分离解码** —— 助手位置的读数带有助手自身状态,且与用户状态可分离;
2. **(b) 可独立操纵** —— 两者能被独立 steering(因果,不只是相关);
3. **(c) 非平坦** —— 助手侧信号确实存在。

> **状态。** 集群实验(Step 1–4)*尚未执行*。下面的分析 cell **有真实结果 JSON 就现算**,否则退回到
> **明确标注为合成(SYNTHETIC)的数据**跑通,好让方法——尤其是 Step-1 两因子 ANOVA 及它要抓的
> *flatness 混淆*——端到端可见。所有公式与 `analyze_grid.py` / campaign 文档一致。
""")

code("""
import sys, subprocess
for pkg in ["numpy", "matplotlib"]:
    try: __import__(pkg)
    except ImportError: subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])
import json, numpy as np, matplotlib.pyplot as plt, matplotlib as mpl
from pathlib import Path
mpl.rcParams.update({"figure.dpi":110,"axes.grid":True,"grid.alpha":0.15,"font.size":11})
rng = np.random.default_rng(0)   # 固定种子 -> 合成 demo 可复现
print("就绪")
""")

# ---- Notation ----
md("""## 0 · 共用记号与"区分"判据

记 $h_\\ell(p)$ 为第 $\\ell{=}49$ 层、token 位置 $p$ 的残差流激活,$\\hat v_e$ 为情绪 $e$ 的单位方向
(`mean_diff+gm+pc50`)。标量读数即投影

$$\\text{proj}_e(p) \\;=\\; h_\\ell(p)\\cdot \\hat v_e .$$

在两个位置读:$u_e=\\text{proj}_e(\\text{user\\_period})$("你的状态")与
$a_e=\\text{proj}_e(\\text{assistant\\_colon})$("我的状态");把 4 个网格情绪(张成 valence×arousal = PC1/PC2)
塌缩成两个带符号 contrast:

$$V=(p_{happy}+p_{calm})-(p_{angry}+p_{sad}),\\qquad A=(p_{angry}+p_{happy})-(p_{sad}+p_{calm}).$$

**区分判据。** "我的状态 vs 你的状态"的区分 $D$ 成立当且仅当**三条件同时**:

$$D \\;=\\; \\underbrace{\\big[\\,\\omega^2_{E_\\text{asst}}(a)>0 \\text{ (显著)}\\,\\big]}_{\\text{(a) 可分离解码}}
\\;\\wedge\\; \\underbrace{\\big[\\,c_{u\\to a}<1 \\;\\wedge\\; c_{a\\to u}\\approx 0\\,\\big]}_{\\text{(b) 可独立操纵}}
\\;\\wedge\\; \\underbrace{\\big[\\,F_{E_\\text{asst}}(a)>F_\\text{crit}\\,\\big]}_{\\text{(c) 非平坦}} .$$

关键:跨位置相关低 $r(u,a)\\approx 0$ **推不出** $D$——当 $\\text{Var}(a)\\approx 0$(flatness 混淆)时它也成立。
下面各步分别检验 (a)/(c)[Step 1]、(b)[Step 2]、indexical 方向[Step 3]、行为后果[Step 4]。""")

# ---- Step 1 design ----
md("""## Step 1 · 两因子网格 (E_user × E_asst) + 两因子 ANOVA

把**用户表达的情绪** `E_user`(由用户消息设定)与**助手自身人设** `E_asst`(由**独立**的 system prompt 设定)
交叉,各 4 情绪张成 valence×arousal,每格 3 个改写(真实误差项)。中性人设的**control block**量 baseline
mirroring,堵住 flatness 漏洞。下面加载真实的网格。""")

code("""
# 加载真实 Step-1 场景网格(已提交的设计产物)
GRID_JSON = Path("../research/experiments/step1-two-factor-grid/two_factor_grid_scenarios.json")
g = json.load(open(GRID_JSON))
print(f"网格: {g['name']}")
print(f"情绪 (valence x arousal): {g['emotions']}")
print(f"主交叉格 n = {g['n_main']}   control block n = {g['n_control']}")
ex = next(p for p in g["prompts"] if not p["control"])
print("\\n--- 一个交叉场景 (E_user != E_asst) ---")
print("id            :", ex["id"])
print("system (E_asst):", ex["system_prompt"])
print("user  (E_user) :", ex["prompt"])
print("标签           : user_emotion=%s  assistant_emotion=%s" % (ex["user_emotion"], ex["assistant_emotion"]))
""")

md("""### Step 1 公式 · 助手位置读数的两因子 ANOVA

平衡 2×2 模型,每格 $k$ 个改写副本:

$$y_{ijk} \\;=\\; \\mu + \\alpha_i^{(U)} + \\beta_j^{(A)} + (\\alpha\\beta)_{ij} + \\varepsilon_{ijk},
\\qquad \\varepsilon_{ijk}\\sim\\mathcal N(0,\\sigma^2),$$

$y$ 是某位置的 $V$(或 $A$)读数,$\\alpha^{(U)}$ 为 **E_user**(镜像)效应,$\\beta^{(A)}$ 为 **E_asst**(自身状态)效应。
平衡平方和:

$$SS_U=\\sum_i N_i(\\bar y_{i\\cdot\\cdot}-\\bar y)^2,\\;\\;
SS_A=\\sum_j N_j(\\bar y_{\\cdot j\\cdot}-\\bar y)^2,\\;\\;
SS_{UA}=\\sum_{ij} n_{ij}(\\bar y_{ij\\cdot}-\\bar y_{i\\cdot\\cdot}-\\bar y_{\\cdot j\\cdot}+\\bar y)^2,$$
$$SS_\\text{err}=SS_\\text{tot}-SS_U-SS_A-SS_{UA},\\qquad MS_\\text{err}=SS_\\text{err}/df_\\text{err},\\;\\; df_\\text{err}=N-ab.$$

每个因子的无偏效应量与 F 检验:

$$\\omega^2_{f}=\\frac{SS_f-df_f\\,MS_\\text{err}}{SS_\\text{tot}+MS_\\text{err}},
\\qquad F_f=\\frac{SS_f/df_f}{MS_\\text{err}}.$$

**自身状态占比** $=\\omega^2_{E_\\text{asst}}(a)$,**镜像占比** $=\\omega^2_{E_\\text{user}}(a)$。当前 verdict:若
$F_{E_\\text{asst}}>F_\\text{crit}(p{<}.05)$ **且** $\\omega^2_{E_\\text{asst}}\\ge\\omega^2_{E_\\text{user}}$ 则判区分成立。""")

code("""
# --- ANOVA 机制(仅 numpy),与 analyze_grid.py 一致 -------------------
VAL = {"happy": +1, "calm": +1, "angry": -1, "sad": -1}   # valence 符号
ARO = {"angry": +1, "happy": +1, "sad": -1, "calm": -1}   # arousal 符号
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
print("ANOVA 就绪")
""")

md("""### Step 1 演示 · 有真实结果则现算,否则两个合成世界

集群读数 `step1/two_factor_grid.json` 还不存在,所以我们在*真实*的 48 格设计上**植入**两个世界,展示 ANOVA
会报什么——关键是它能把真正的自身状态信号和 **flatness 混淆**分开:

- **世界 A — 真自身状态:** 助手读数既响应 `E_asst`(β 大)又镜像用户(α)。
- **世界 B — flatness 混淆:** 助手读数**只**响应 `E_user`(β≈0)——一个跨位置相关很低的模式,朴素的
  "区分=低 r"检验会误判成区分。""")

code("""
REAL = Path("../results/ant_emotion_concepts/step1/two_factor_grid.json")

# 因子符号取自真实网格设计(valence 轴,助手位置)
main = [p for p in g["prompts"] if not p["control"]]
fa_user = [VAL[p["user_emotion"]] for p in main]      # E_user 符号
fb_asst = [VAL[p["assistant_emotion"]] for p in main] # E_asst 符号
N = len(main)

def synth_world(beta_asst, alpha_user, noise=1.0):
    # y = mu + alpha*sign_user + beta*sign_asst + eps  (助手位置 valence 读数)
    eps = rng.normal(0, noise, N)
    return 0.2 + alpha_user*np.array(fa_user) + beta_asst*np.array(fb_asst) + eps

def verdict(res):
    own, mir = res["omega2_asst"], res["omega2_user"]
    nonflat = res["F_asst"] > 4.0                      # ~ F_crit(df1=1, p<.05)
    if nonflat and own >= mir:   tag = "区分成立 (own >= mirror, 非平坦)"
    elif nonflat:                tag = "部分 (自身信号存在但弱于镜像)"
    else:                        tag = "未检出 -> flatness 混淆"
    return tag

if REAL.exists():
    d = json.load(open(REAL)); L = "49"
    def readout(r, pos): return sum(VAL[e]*r["projections"][e][L][pos] for e in GRID)
    rows = [r for r in d["results"] if not r.get("control")]
    y = [readout(r, "assistant_colon") for r in rows]
    fu = [VAL[r["user_emotion"]] for r in rows]; fb = [VAL[r["assistant_emotion"]] for r in rows]
    res = two_way_anova(y, fu, fb); worlds = {"真实集群结果": res}
    print("已加载真实结果:", REAL)
else:
    print("尚无真实结果 ->", REAL)
    print("在真实 48 格设计上跑合成 demo (valence @ assistant_colon):\\n")
    worlds = {
        "A: 真自身状态": two_way_anova(synth_world(beta_asst=1.4, alpha_user=1.0), fa_user, fb_asst),
        "B: flatness 混淆": two_way_anova(synth_world(beta_asst=0.0, alpha_user=1.4), fa_user, fb_asst),
    }

for name, res in worlds.items():
    print(f"[{name}]")
    print(f"   omega^2  E_asst(own)={res['omega2_asst']:.3f} (F={res['F_asst']:.2f})   "
          f"E_user(mirror)={res['omega2_user']:.3f} (F={res['F_user']:.2f})   "
          f"inter={res['omega2_interaction']:.3f}  resid={res['residual_frac']:.3f}")
    print(f"   => {verdict(res)}\\n")
""")

code("""
# 可视化每个世界的 omega^2 分解
labels = ["E_asst (own)", "E_user (mirror)", "interaction", "residual"]
names = list(worlds.keys())
x = np.arange(len(labels)); w = 0.8/len(names)
fig, ax = plt.subplots(figsize=(9,5))
for i,(nm,res) in enumerate(worlds.items()):
    vals = [res["omega2_asst"], res["omega2_user"], res["omega2_interaction"], res["residual_frac"]]
    ax.bar(x + i*w - 0.4 + w/2, vals, w, label=nm)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("variance share"); ax.set_title("Step 1 two-way ANOVA - variance decomposition (assistant position)")
ax.legend(fontsize=8); plt.tight_layout(); plt.show()
print("自身状态区分需要最左 (E_asst) 那根柱子显著且非平凡。")
""")

# ---- Step 2 ----
md("""## Step 2 · Steering / 因果耦合(相关 → 因果)

Step 1 是相关的。要证明两状态**可独立操纵**,在选定位置往残差流注入一个方向,测*另一个*位置读数动多少。

$$h'_\\ell(p)=h_\\ell(p)+\\lambda\\,\\hat v_e ,\\qquad
c_{u\\to a}=\\frac{\\partial\\, a_e}{\\partial\\lambda_u},\\quad
c_{a\\to u}=\\frac{\\partial\\, u_e}{\\partial\\lambda_a}.$$

可独立操纵 $\\Leftrightarrow c_{u\\to a}<1$(注入*用户*情绪**不会**完全传到助手读数)**且** $c_{a\\to u}\\approx 0$
(注入*助手*情绪不动用户读数)。每个 $c$ 用多个 $\\lambda$ 的**剂量-响应**斜率估计。
*(需要 GPU steering——下面用合成剂量-响应示意目标信号形态。)*""")

code("""
# 合成剂量-响应,示意目标信号形态(GPU steering 待跑)
lam = np.linspace(0, 3, 7)
a_from_user = 0.40*lam + rng.normal(0,0.05,lam.size)   # c_{u->a} = 0.40  (<1, 部分传递)
u_from_asst = 0.04*lam + rng.normal(0,0.05,lam.size)   # c_{a->u} ~ 0     (独立)
c_ua = np.polyfit(lam, a_from_user, 1)[0]
c_au = np.polyfit(lam, u_from_asst, 1)[0]
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(lam, a_from_user, "o-", label=f"inject @ user -> assistant readout  (c={c_ua:.2f})")
ax.plot(lam, u_from_asst, "s--", label=f"inject @ assistant -> user readout  (c={c_au:.2f})")
ax.plot(lam, lam, ":", color="gray", label="c = 1 (full transfer)")
ax.set_xlabel("injection gain  lambda"); ax.set_ylabel("change in other-position readout")
ax.set_title("Step 2 coupling (SYNTHETIC): c_{u->a} < 1 and c_{a->u} ~ 0 = independent")
ax.legend(fontsize=8); plt.tight_layout(); plt.show()
print(f"目标: c_(u->a)={c_ua:.2f} < 1  且  c_(a->u)={c_au:.2f} ~ 0  -> 可独立操纵")
""")

# ---- Step 3 ----
md("""## Step 3 · Indexical "whose-state" 探针

是否存在一个方向,编码某个感受"属于谁"、而与感受本身无关?取第一/第二人称框架的均值差,用其符号解码:

$$w=\\frac{1}{N}\\sum_{n}\\big(h_\\ell(x^{\\,\\text{I feel}}_n)-h_\\ell(x^{\\,\\text{you feel}}_n)\\big),\\qquad
\\hat y(x)=\\operatorname{sign}\\!\\big(h_\\ell(x)\\cdot w\\big).$$

indexical 方向存在当且仅当解码高于随机:
$\\text{acc}=\\tfrac{1}{M}\\sum_i \\mathbb 1[\\hat y(x_i)=y_i] > 0.5$。
*(下面是合成示意;真实探针用模型激活。)*""")

code("""
# 合成:两类沿 'whose-state' 方向 w 分开,叠加噪声
M, dim = 200, 16
w_true = rng.normal(0,1,dim); w_true /= np.linalg.norm(w_true)
y = rng.integers(0,2,M)*2 - 1                      # +1 = "I feel", -1 = "you feel"
H = (0.8*y[:,None])*w_true[None,:] + rng.normal(0,1,(M,dim))
w_hat = H[y==1].mean(0) - H[y==-1].mean(0)         # w 的均值差估计
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
md("""## Step 4 · 与行为挂钩(自身状态真的驱动回复吗?)

最后,用增益 $\\lambda_a$ steer `E_asst`,生成回复 $R$,用 LLM-judge 判其情绪 stance $J(R)$。两个比率:

$$m_\\text{asst}=\\tfrac1n\\sum_i \\mathbb 1[\\,J(R_i)=E_{\\text{asst},i}\\,]\\;\\uparrow\\text{ 随 }\\lambda_a,
\\qquad
m_\\text{user}=\\tfrac1n\\sum_i \\mathbb 1[\\,J(R_i)=E_{\\text{user},i}\\,]\\approx m_\\text{unsteered}.$$

行为层面的区分 = steer 助手自身状态会抬高 $m_\\text{asst}$,而**只** steer 用户情绪**不**劫持回复
($m_\\text{user}$ 保持平)。*(合成示意;真实 run 需生成+评判。)*""")

code("""
# 合成:steer 助手会移动回复 stance;steer 用户不劫持
gains = np.array([0, 1, 2, 3])
m_asst = 0.15 + 0.25*gains + rng.normal(0,0.02,gains.size)          # 随 lambda_a 上升
m_asst = np.clip(m_asst, 0, 1)
m_user = 0.18 + 0.0*gains + rng.normal(0,0.02,gains.size)           # 只 steer 用户时保持平
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(gains, m_asst, "o-", label="steer E_asst -> reply matches assistant state")
ax.plot(gains, m_user, "s--", label="steer E_user only -> reply matches user (hijack?)")
ax.set_xlabel("steering gain"); ax.set_ylabel("LLM-judge stance-match rate")
ax.set_ylim(0,1); ax.set_title("Step 4 behavior link (SYNTHETIC): own-state drives reply, user does not hijack")
ax.legend(fontsize=8); plt.tight_layout(); plt.show()
""")

# ---- Summary ----
md("""## 小结 & 集群 run 将替换什么

- 用**三段式判据 $D$**(可分离 + 可独立操纵 + 非平坦)把问题变得可检验;单看跨位置 $r$ 低是
  **退化/flatness** 情形,不能定论。
- **Step 1** 用两因子 ANOVA 分解助手位置读数:自身状态占比 $\\omega^2_{E_\\text{asst}}$ vs 镜像占比
  $\\omega^2_{E_\\text{user}}$——demo 展示 ANOVA 在真实 48 格设计上干净地把*真自身状态*世界和
  *flatness 混淆*世界分开。
- **Step 2–4** 补上因果(耦合 $c$)、indexical whose-state 方向($w$)、行为后果($m_\\text{asst}$ vs $m_\\text{user}$)。
- **待办:** 一旦集群写出 `results/ant_emotion_concepts/step1/two_factor_grid.json`(Step 1,仅 forward-pass)
  及 Step 2–4 的 steering/生成输出,四个分析 cell 会自动切到真实数字。在那之前图都标 **SYNTHETIC**、只示意目标
  信号形态。公式与 `research/campaigns/self-vs-user-distinction.md` 和 `analyze_grid.py` 完全一致。
""")

nb = {"cells":cells,
      "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                  "language_info":{"name":"python","version":"3.x"}},
      "nbformat":4,"nbformat_minor":5}
out = pathlib.Path(__file__).parent/"emotion_selfuser_distinction_zh.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print("wrote", out, "with", len(cells), "cells")
