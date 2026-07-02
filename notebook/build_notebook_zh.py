#!/usr/bin/env python3
"""生成中文版 Emotion Concepts 复现 notebook。
运行: python3 build_notebook_zh.py  ->  emotion_concepts_reproduction_zh.ipynb
说明: markdown 与 print 用中文;matplotlib 图内文字保留英文(避免缺 CJK 字体渲染成方块)。
"""
import json, pathlib

cells = []
def md(src):  cells.append({"cell_type":"markdown","metadata":{},"source":src.splitlines(keepends=True)})
def code(src):cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":src.strip("\n").splitlines(keepends=True)})

md("""# Emotion Concepts 复现 — 中文可视化笔记本

复现 ewern 的 [LessWrong 帖子](https://www.lesswrong.com/posts/sJQ62HbA76s3aiuiT)
(工具 [traitinterp](https://github.com/ewernn/traitinterp)),即 Anthropic *Emotion Concepts and their
Function in a Large Language Model* (Sofroniew et al. 2026) 在 **Llama 3.3 70B Instruct** 上的复现。

**要复现的头条结论**:在 Llama 上,*assistant* 位置会强烈**镜像** *用户* 的情绪(跨位置相关
**r ≈ 0.63**);而论文报告 Claude Sonnet 4.5 上两者基本**独立**(**r ≈ 0.11**)。即论文在 Sonnet 上观察到的
user/assistant "情绪解离" **在 Llama 上不成立**——这是 **Llama vs Sonnet 的模型差异**,不是方法失效。

第 1–4 步全部基于 traitinterp 官方 `emotion-concepts-v1` **预计算数据包**(向量 + 各 stage 结果),
本地即可运行、**不需要 GPU 或加载模型**,所有数字现场从结果 JSON 重算、可核对。**§3e 再加入我们自己的
独立结果**:在 Llama 3.3 70B 上**从头重抽全部 171 个情绪向量**(Slurm 作业 `7903504`),用同样方法重算头条,
与数据包并排(从头 **r ≈ 0.72** vs 数据包 0.63)。

> 数据目录:`./data/`(从集群 `/scratch/$USER/traitinterp/experiments/ant_emotion_concepts/` 拉取)
""")

code("""
# 环境:仅需 numpy + matplotlib(无需 torch / GPU)
import sys, subprocess
for pkg in ["numpy", "matplotlib"]:
    try: __import__(pkg)
    except ImportError: subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])

import json, numpy as np, matplotlib.pyplot as plt, matplotlib as mpl
from pathlib import Path
mpl.rcParams.update({"figure.dpi":110,"axes.grid":True,"grid.alpha":0.15,"font.size":11})
DATA = Path("data")
def load(p): return json.load(open(DATA/p))
print("数据文件数:", sum(1 for _ in DATA.rglob("*") if _.is_file()))
print("就绪")
""")

# ---- Step 1 ----
md("""## 第 1 步 · 数据从哪来(没有真实用户数据)

复现里有两类数据,**都不是从真人采集的**:

1. **抽情绪向量的对比数据** `datasets/traits/<情绪>/positive.jsonl`:让模型写一个"传达某情绪但**不准点名**该情绪"
   的故事;模型自己生成,再取「情绪激活 − 中性激活」的**均值差(mean_diff)**作为该情绪的方向向量。
2. **dissociation 测试场景** `dissociation_scenarios.json`:直接**逐字抄自**论文 Table 3 的 8 条手写 prompt。

**公式 — 情绪方向 `mean_diff+gm+pc50`(层 $\\ell=49$)。** 记 $h_\\ell(x)$ 为输入 $x$ 在第 $\\ell$ 层的残差流激活,
$x^{+}$ 为情绪故事,$x^{0}$ 为中性文本:

$$v_e^{\\text{raw}} \\;=\\; \\frac{1}{N_e}\\sum_{i=1}^{N_e} h_\\ell(x_i^{+}) \\;-\\; \\frac{1}{N_0}\\sum_{j=1}^{N_0} h_\\ell(x_j^{0})
\\qquad\\text{(均值差 mean-difference)}$$

$$v_e^{\\text{gm}} \\;=\\; v_e^{\\text{raw}} - \\frac{1}{171}\\sum_{e'} v_{e'}^{\\text{raw}}
\\qquad\\text{(总均值居中 }gm\\text{)}$$

$$v_e \\;=\\; v_e^{\\text{gm}} - U_k U_k^{\\top} v_e^{\\text{gm}}, \\qquad \\hat v_e = \\frac{v_e}{\\lVert v_e\\rVert}
\\qquad\\text{(去掉共享的前 }k\\text{ 个主成分,L49 上 }k{=}21\\text{;再单位化)}$$

其中 $U_k$ 是全部 171 个已居中向量共享的前 $k$ 个主成分(即 `pc50` 去噪步骤)。后续所有数字都用单位向量 $\\hat v_e$。

下面看一条抽向量用的 prompt:""")

code("""
angry = [json.loads(l) for l in open(DATA/"traits/ant_emotion_concepts/angry/positive.jsonl")]
print(f"'angry' 故事生成 prompt 数: {len(angry)}\\n")
print(angry[0]["prompt"][:680], "...")
""")

# ---- Step 2 ----
md("""## 第 2 步 · 情绪向量的几何结构 (Stage 3)

171 个情绪各得到一个方向向量。先看它们彼此的关系。""")

md("""### 2a · 情绪×情绪 余弦相似度热图
相近情绪(如 happy/pleased/delighted)互相高相似,对立情绪呈负相关。

**公式。** 每个格子是两个情绪方向的余弦相似度:
$$\\cos(v_a, v_b) = \\frac{v_a \\cdot v_b}{\\lVert v_a\\rVert\\,\\lVert v_b\\rVert}\\in[-1,1].$$""")
code("""
ch = load("results/stage3_geometry/cosine_heatmap.json")
M = np.array(ch["ordered_matrix"]); names = ch["ordered_names"]
fig, ax = plt.subplots(figsize=(9,8))
im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
step = 6
ax.set_xticks(range(0,len(names),step)); ax.set_xticklabels(names[::step], rotation=90, fontsize=6)
ax.set_yticks(range(0,len(names),step)); ax.set_yticklabels(names[::step], fontsize=6)
ax.set_title(f"Emotion x Emotion cosine similarity (171 traits)\\nmean={ch['stats']['mean']:.2f}")
fig.colorbar(im, fraction=0.046, pad=0.04, label="cosine"); ax.grid(False); plt.tight_layout(); plt.show()
""")

md("""### 2b · PCA:第一主成分 = 效价(valence)
对 171 个向量做 PCA。PC1 与人工标注的 valence(Russell–Mehrabian PAD 范式)强相关——
说明模型自发地把情绪沿"正负"轴组织起来。

**公式。** 把向量堆成 $V\\in\\mathbb{R}^{171\\times d}$,居中 $\\tilde V = V - \\bar v$,对协方差
$C = \\tfrac{1}{171}\\tilde V^{\\top}\\tilde V = W\\Lambda W^{\\top}$ 做特征分解。第 $k$ 个主成分得分为
$P_{\\cdot k} = \\tilde V\\,w_k$,其解释方差占比为 $\\lambda_k / \\sum_j \\lambda_j$。报告的对齐度是 PC1 与人工
valence 之间的 **Pearson 相关**:
$$r = \\frac{\\sum_i (x_i-\\bar x)(y_i-\\bar y)}{\\sqrt{\\sum_i (x_i-\\bar x)^2}\\,\\sqrt{\\sum_i (y_i-\\bar y)^2}}.$$""")
code("""
pca = load("results/stage3_geometry/pca_analysis.json")
P = np.array(pca["projections"]); ve = pca["variance_explained"]
fig, ax = plt.subplots(figsize=(9,7))
ax.scatter(P[:,0], P[:,1], s=18, alpha=0.5, color="#1f77b4")
for name,val in pca["pc1_sorted"][:5] + pca["pc1_sorted"][-5:]:
    i = pca["trait_names"].index(name)
    ax.annotate(name, (P[i,0],P[i,1]), fontsize=8)
ax.set_xlabel(f"PC1 ({ve[0]*100:.0f}% var)  ~ valence")
ax.set_ylabel(f"PC2 ({ve[1]*100:.0f}% var)  ~ arousal")
ax.set_title("PCA of 171 emotion vectors"); plt.tight_layout(); plt.show()
ls = load("results/stage3_geometry/layer_sweep_pc1_valence.json")
best = max(ls["per_layer"].values(), key=lambda d: d["abs_pc1_vs_valence"])
print(f"PC1 与人工 valence 相关: r = {best['pc1_vs_valence_r']:.3f}  (重叠情绪数 n={best['n_overlap']})")
print(f"PC2 与人工 arousal 相关: r = {best['pc2_vs_arousal_r']:.3f}")
""")

md("""### 2c · UMAP + k-means:情绪自然成簇
降到 2D 后按 k-means 上色,可见正向/负向/高唤醒等簇。

**公式。** *k-means* 最小化簇内平方距离
$$\\min_{\\{S_1,\\dots,S_k\\}} \\sum_{c=1}^{k}\\sum_{x\\in S_c} \\lVert x-\\mu_c\\rVert^2,
\\qquad \\mu_c = \\frac{1}{|S_c|}\\sum_{x\\in S_c} x.$$
*UMAP* 无闭式解:它在高维/低维各建一个模糊近邻图,最小化二者的交叉熵
$\\sum_{ij}\\big[p_{ij}\\log\\tfrac{p_{ij}}{q_{ij}} + (1{-}p_{ij})\\log\\tfrac{1-p_{ij}}{1-q_{ij}}\\big]$,
所以坐标只有**相对**意义(看距离,不看坐标轴)。""")
code("""
um = load("results/stage3_geometry/clusters_umap.json")
XY = np.array(um["umap_coordinates"]); lab = np.array(um["cluster_assignments"])
clusters = um.get("clusters", {})
fig, ax = plt.subplots(figsize=(9,7))
for c in sorted(set(lab)):
    m = lab==c
    cname = clusters.get(str(c),{}).get("name") or clusters.get(str(c),{}).get("label") or f"cluster {c}"
    ax.scatter(XY[m,0], XY[m,1], s=22, alpha=0.7, label=str(cname)[:30])
ax.set_title(f"UMAP of emotion vectors, k={um.get('k')} clusters")
ax.legend(fontsize=7, loc="best"); ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2"); plt.tight_layout(); plt.show()
""")

md("""### 2d · 哪一层效价信号最强(层扫描)
沿层扫描 PC1↔valence 的 |r|,在模型中后段达到峰值——这也是后续选 L49 的依据。

**公式。** 逐层重做 PCA,选 PC1 与 valence 相关最强的那一层:
$$\\ell^{\\star} = \\arg\\max_{\\ell} \\big| r\\big(\\text{PC1}^{(\\ell)},\\, \\text{valence}\\big)\\big|.$$""")
code("""
ls = load("results/stage3_geometry/layer_sweep_pc1_valence.json")
layers = ls["layers"]
rv = [ls["per_layer"][str(l)]["abs_pc1_vs_valence"] for l in layers]
ra = [ls["per_layer"][str(l)]["abs_pc2_vs_arousal"] for l in layers]
fig, ax = plt.subplots(figsize=(9,5))
ax.plot(layers, rv, "o-", label="|PC1 <-> valence|")
ax.plot(layers, ra, "s--", label="|PC2 <-> arousal|", alpha=0.7)
ax.axvline(49, color="gray", ls=":", label="L49 (used downstream)")
ax.set_xlabel("layer"); ax.set_ylabel("|Pearson r| vs human PAD norms")
ax.set_title("Valence/arousal alignment across layers"); ax.legend(); plt.tight_layout(); plt.show()
""")

# ---- Step 3 ----
md("""## 第 3 步 · 头条结论:User / Assistant 情绪解离 (Stage 5 · Fig 10)

论文设计了 8 个场景:**用户表达的情绪**故意与**助手应有的回应情绪**不一致(如用户 angry,助手应 calm)。
分别在**用户那句话末尾的句号(user_period)**和**助手回合开头的冒号(assistant_colon)**两个 token 位置,
把激活投影到情绪探针上。

- 两位置相关**弱** → 助手"知道"自己该有独立情绪(论文在 Sonnet 上:r≈0.11)。
- 两位置相关**强** → 助手位置只是镜像用户情绪(我们在 Llama 上看到的)。

**公式 — 探针投影。** 在 token 位置 $p$(`user_period` 或 `assistant_colon`)上,情绪 $e$ 的标量读数就是该位置激活在
单位方向上的点积:
$$\\text{proj}_e(p) = h_\\ell(p)\\cdot \\hat v_e \\qquad (\\ell = 49).$$""")

code("""
scen = load("inference/ant_emotion_concepts/dissociation_scenarios.json")["prompts"]
print(f"{len(scen)} 个场景(逐字抄自论文 Table 3):\\n")
for s in scen:
    print(f"  [{s['user_emotion']:>10} -> {s['expected_assistant_emotion']:<7}]  {s['prompt'][:70]}")
""")

md("""### 3a · 计算跨位置相关(官方 Fig10 方法:L49,6 情绪探针子集)

**公式。** 每个 (场景 $s$, 情绪 $e$) 出一个点,取 $x_{s,e}=\\text{proj}_e(\\text{user\\_period})$、
$y_{s,e}=\\text{proj}_e(\\text{assistant\\_colon})$,再对合并后的点集(共 $8\\text{ 场景}\\times 6\\text{ 情绪}=48$ 点)算 Pearson $r$:
$$r = \\frac{\\sum_{s,e}(x_{s,e}-\\bar x)(y_{s,e}-\\bar y)}{\\sqrt{\\sum_{s,e}(x_{s,e}-\\bar x)^2}\\,\\sqrt{\\sum_{s,e}(y_{s,e}-\\bar y)^2}}.$$""")
code("""
diss = load("results/stage5/dissociation.json")
L = "49"
EMO_FIG10 = ["happy","calm","loving","sad","afraid","angry"]   # 官方脚本 PLOT_EMOTIONS_FIG10
up, ac, labels = [], [], []
for s in diss["results"]:
    for e in EMO_FIG10:
        c = s["projections"].get(e,{}).get(L,{})
        if "user_period" in c and "assistant_colon" in c:
            up.append(c["user_period"]); ac.append(c["assistant_colon"]); labels.append(e)
up, ac = np.array(up), np.array(ac)
r = np.corrcoef(up, ac)[0,1]
print(f"数据点 n = {len(up)}  (8 场景 x 6 情绪)")
print(f"pooled Pearson r (user_period vs assistant_colon) = {r:.4f}")
print(f"-> LessWrong 帖子: 0.63 | 论文 Sonnet 对照: 0.11")
up2,ac2=[],[]
for s in diss["results"]:
    for e,layers in s["projections"].items():
        c=layers.get(L,{})
        if "user_period" in c: up2.append(c["user_period"]); ac2.append(c["assistant_colon"])
n_all=len(set(l for s in diss['results'] for l in s['projections']))
print(f"(变体:全 {n_all} 情绪, n={len(up2)} -> r={np.corrcoef(up2,ac2)[0,1]:.4f})")
""")

md("### 3b · 散点图:用户位置 vs 助手位置(头条图)")
code("""
COLORS = {"happy":"#2ca02c","calm":"#1f77b4","loving":"#e377c2","sad":"#9467bd","afraid":"#d62728","angry":"#ff7f0e"}
fig, ax = plt.subplots(figsize=(7.5,7))
for e in EMO_FIG10:
    m = [l==e for l in labels]
    ax.scatter(up[m], ac[m], s=120, alpha=0.85, color=COLORS[e], edgecolors="white", linewidths=0.6, label=e, zorder=3)
sl, ic = np.polyfit(up, ac, 1)
xs = np.linspace(up.min(), up.max(), 50)
ax.plot(xs, sl*xs+ic, "-", color="#222", lw=2.5, label=f"Llama fit  r={r:.2f}", zorder=4)
ps = 0.11 * (ac.std()/up.std()); pi = ac.mean() - ps*up.mean()
ax.plot(xs, ps*xs+pi, "--", color="#999", lw=2.5, label="Sonnet ref  r=0.11", zorder=4)
ax.axhline(0, color="k", lw=0.5, alpha=0.3); ax.axvline(0, color="k", lw=0.5, alpha=0.3)
ax.set_xlabel("probe proj @ user period"); ax.set_ylabel("probe proj @ assistant colon")
ax.set_title("Fig 10 - assistant position mirrors user emotion (Llama 3.3 70B, L49)")
ax.legend(fontsize=8); plt.tight_layout(); plt.show()
""")

md("### 3c · 热图:每个场景在 U(用户)/A(助手)位置的探针激活\n每个情绪一对行(U/A)。U 与 A 颜色高度一致 = 助手镜像用户。")
code("""
scen_ids = [s["id"] for s in diss["results"]]
rows = []; row_labels = []
for e in EMO_FIG10:
    for pos in ["user_period","assistant_colon"]:
        rows.append([diss["results"][k]["projections"].get(e,{}).get(L,{}).get(pos,0.0) for k in range(len(diss["results"]))])
        row_labels.append(f"{e[:4]}-{'U' if pos=='user_period' else 'A'}")
Mh = np.array(rows); vmax=np.abs(Mh).max()
fig, ax = plt.subplots(figsize=(10,7))
im = ax.imshow(Mh, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels, fontsize=8)
ax.set_xticks(range(len(scen_ids))); ax.set_xticklabels(scen_ids, rotation=45, ha="right", fontsize=8)
for i in range(2,len(row_labels),2): ax.axhline(i-0.5, color="k", lw=0.5, alpha=0.3)
ax.set_title("Probe activation per scenario at User(U) vs Assistant(A) positions")
fig.colorbar(im, fraction=0.046, pad=0.04); ax.grid(False); plt.tight_layout(); plt.show()
""")

md("""### 3d · 复现核对表

| 来源 | 跨位置 r | 含义 |
|---|---|---|
| 论文 Sonnet 4.5 对照 | 0.11 | 解离(取自论文,不重算) |
| LessWrong 帖子 (Llama) | **0.63** | 助手镜像用户情绪 |
| 数据包·作者向量(3a 重算) | ~0.63 | 与帖子一致 |
| **我们从头自抽(3e)** | **~0.72** | 独立 run,结论一致,~+0.09 |

> 注:repo 内部 `findings.md` 另给出 r=0.7718(不同/更早的 run)。同一流水线三次 run 落在
> **0.63 → 0.72 → 0.77**:**方向稳健**,**精确值对抽取 seed 敏感**。我们把 ~0.72 当作定性一致,而非钉死的数。

**"在 Llama 上不成立"到底指什么(避免常见误读):** traitinterp **方法**在 Llama 上迁移得很干净——几何、
PC1=valence(r≈0.97)、隐含情绪分类、数值强度**全部复现**,我们从头自抽的头条(0.64–0.72)与帖子(0.63)
只差一点点。**唯一的跨模型差异**就是解离本身:**Sonnet** 上 user/assistant 情绪保持**独立**(r≈0.11),而
**Llama** 上助手位置强烈**镜像**用户(r≈0.63–0.72)。所以"在 Llama 上不成立"是 ewern 的**真实发现**——
一个 **Llama vs Sonnet 的行为差异**——**不是**方法失效、也不是我们复刻没对齐。""")

# ---- Step 3e ----
md("""### 3e · 我们独立从头重抽 vs 作者数据包

前面全部用作者的**预计算数据包**向量。为验证头条不是那一次抽取的偶然产物,我们在集群上于 Llama 3.3 70B
Instruct 上**从头重抽全部 171 个情绪向量**(Slurm 作业 `7903504`,4-bit,与数据包相同的 L49 /
`mean_diff+gm+pc50` 方法),再重算**同一个** Fig-10 dissociation。现场从
`../results/ant_emotion_concepts/stage5/dissociation.json` 读取。""")
code("""
# 我们从头自抽的结果 vs 数据包,同一 Fig-10 方法
FRESH = Path("../results/ant_emotion_concepts/stage5/dissociation.json")
def xpos(d, emos):
    u, a, lab = [], [], []
    for s in d["results"]:
        proj = s["projections"]
        for e in (emos if emos else list(proj.keys())):
            c = proj.get(e,{}).get(L,{})
            if "user_period" in c and "assistant_colon" in c:
                u.append(c["user_period"]); a.append(c["assistant_colon"]); lab.append(e)
    u, a = np.array(u), np.array(a)
    return np.corrcoef(u,a)[0,1], u, a, lab

bundle = diss                       # 3a 里从数据包加载
fresh  = json.load(open(FRESH))     # 我们从头自抽的 run
rb6,_,_,_   = xpos(bundle, EMO_FIG10);  rf6,uf,af,flab = xpos(fresh, EMO_FIG10)
rb_all,*_   = xpos(bundle, None);       rf_all,*_      = xpos(fresh, None)
print(f"Fig-10 (6情绪, n={len(uf)}):   数据包 r = {rb6:.4f}    从头自抽 r = {rf6:.4f}")
print(f"全情绪变体:               数据包 r = {rb_all:.4f}    从头自抽 r = {rf_all:.4f}")
print(f"参考: Sonnet(论文) 0.11 | LW 帖子 0.63 | repo findings.md 更早 run 0.7718")
print(f"-> 结论一致(强镜像、远离 0.11);精确值对抽取敏感")
""")
code("""
# 左:我们从头自抽的散点+拟合;右:各来源的头条 r 柱状图
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.5))
flab = np.array(flab)
for e in EMO_FIG10:
    m = flab == e
    axL.scatter(uf[m], af[m], s=90, alpha=0.85, color=COLORS[e], edgecolors="white", linewidths=0.5, label=e, zorder=3)
sl, ic = np.polyfit(uf, af, 1); xs = np.linspace(uf.min(), uf.max(), 50)
axL.plot(xs, sl*xs+ic, "-", color="#222", lw=2.5, label=f"from-scratch fit  r={rf6:.2f}", zorder=4)
ps = 0.11*(af.std()/uf.std()); axL.plot(xs, ps*xs+(af.mean()-ps*uf.mean()), "--", color="#999", lw=2, label="Sonnet ref  r=0.11", zorder=4)
axL.axhline(0, color="k", lw=.5, alpha=.3); axL.axvline(0, color="k", lw=.5, alpha=.3)
axL.set_xlabel("probe proj @ user period"); axL.set_ylabel("probe proj @ assistant colon")
axL.set_title("Our from-scratch vectors (Llama 3.3 70B, L49)"); axL.legend(fontsize=8)
srcs = ["Sonnet\\n(paper)", "LW post", "bundle\\n(author)", "from-scratch\\n(ours)"]
rs   = [0.11, 0.63, rb6, rf6]; cols = ["#bbbbbb", "#7f7f7f", "#1f77b4", "#d62728"]
axR.bar(srcs, rs, color=cols)
for i, v in enumerate(rs): axR.text(i, v+0.012, f"{v:.2f}", ha="center", fontsize=11)
axR.axhline(0.63, color="gray", ls=":", alpha=.6, label="post 0.63")
axR.set_ylabel("cross-position Pearson r (Fig-10)"); axR.set_ylim(0, 0.85)
axR.set_title("Headline r by source - dissociation only on Sonnet"); axR.legend(fontsize=8)
plt.tight_layout(); plt.show()
""")

# ---- Step 4 ----
md("""## 第 4 步 · 旁证验证 (Stage 4)

探针不仅几何漂亮,还能真正"读懂"隐含情绪、并对强度单调响应。""")

md("""### 4a · 隐含情绪分类 (Fig 2):12 场景 × 12 探针 混淆热图
对角线最亮 = 探针能从不点名情绪的场景里识别出正确情绪。

**公式。** 场景 $i$ 的激活与探针 $j$ 的相似度为 $S_{ij}=\\cos\\!\\big(h_\\ell(\\text{scenario}_i),\\,\\hat v_j\\big)$;
top-1 准确率统计"最亮探针恰为真情绪"的场景比例:
$$\\text{acc} = \\frac{1}{M}\\sum_{i=1}^{M} \\mathbb{1}\\!\\left[\\arg\\max_j S_{ij} = \\text{true}(i)\\right].$$""")
code("""
ie = load("results/stage4_validation/implicit_emotion.json")
Mf = np.array(ie["similarity_matrix_focused"]); pe = ie["prompt_emotions"]; fp = ie["focused_probes"]
acc = np.mean([np.argmax(Mf[i])==fp.index(pe[i]) if pe[i] in fp else np.argmax(Mf[i])==i for i in range(len(pe))])
fig, ax = plt.subplots(figsize=(8,7))
im = ax.imshow(Mf, cmap="viridis", aspect="auto")
ax.set_xticks(range(len(fp))); ax.set_xticklabels(fp, rotation=90, fontsize=8)
ax.set_yticks(range(len(pe))); ax.set_yticklabels(pe, fontsize=8)
ax.set_xlabel("probe"); ax.set_ylabel("scenario's true emotion")
ax.set_title(f"Implicit emotion (Fig 2) - top-1 accuracy ~ {acc*100:.0f}%")
fig.colorbar(im, fraction=0.046, pad=0.04); ax.grid(False); plt.tight_layout(); plt.show()
""")

md("""### 4b · 数值强度单调性 (Fig 3)
例:Tylenol 剂量 200→8000mg,'afraid' 探针单调上升——模型对危险程度有连续表征。

**公式。** 对剂量 $d$,套模板生成 prompt 并读同一个投影 $\\text{proj}_e(d) = h_\\ell(\\text{prompt}(d))\\cdot\\hat v_e$;
单调性即 $d_1<d_2 \\Rightarrow \\text{proj}_{\\text{afraid}}(d_1)\\le\\text{proj}_{\\text{afraid}}(d_2)$
(等价于剂量与投影之间 Spearman $\\rho\\approx 1$)。""")
code("""
ni = load("results/stage4_validation/numerical_intensity.json")
t = ni["tylenol_dose"]; vals = t["values"]
fig, ax = plt.subplots(figsize=(8,5))
for probe in ["afraid","desperate","calm","sad"]:
    if probe in t["probes"]:
        ax.plot(vals, t["probes"][probe], "o-", label=probe)
ax.set_xlabel(f"{t['variable']} (mg Tylenol)"); ax.set_ylabel("probe projection")
ax.set_title("Numerical intensity (Fig 3): probe response vs dose"); ax.legend(); plt.tight_layout(); plt.show()
""")

# ---- Step 5 ----
md("""## 第 5 步 · 与作者渲染图交叉核对

数据包附带作者在 Llama 3.3 70B 上渲染的图。我们上面用同一份数据现算的图,应与之吻合。""")
code("""
import matplotlib.image as mpimg
show = [("fig10_ours.png","Fig 10 dissociation (headline)"),
        ("fig2_ours.png","Fig 2 implicit emotion"),
        ("fig3_ours.png","Fig 3 numerical intensity")]
fig, axes = plt.subplots(len(show),1, figsize=(11,16))
for ax,(f,title) in zip(axes, show):
    p = DATA/"paper_figures/ours"/f
    if p.exists(): ax.imshow(mpimg.imread(p)); ax.set_title("author: "+title, fontsize=10)
    ax.axis("off")
plt.tight_layout(); plt.show()
""")

md("""## 小结

- **头条已复现**:Llama 3.3 70B 上 user/assistant 跨位置情绪相关 **r ≈ 0.63**(论文 Sonnet 0.11)——
  论文的"情绪解离"在 Llama 上不成立(即 Llama 会镜像用户情绪)。
- 情绪向量几何(PC1=valence、成簇、中后层峰值)、隐含情绪分类、数值强度单调性**均复现**。
- 来源透明:情绪向量 = 模型生成情绪故事激活的均值差;dissociation 场景 = 论文 Table 3 原文。
- **独立确认(§3e)**:在 Llama 3.3 70B 上从头重抽全部 171 向量(Slurm 作业 `7903504`,4-bit)重算头条
  → 从头 **r ≈ 0.72**(数据包 0.63)。完全独立的一次抽取仍得同样结论;精确值对抽取 seed 敏感
  (三次 run 0.63 / 0.72 / 0.77),故按定性一致对待、不钉死数值。
- 第 1–4 步基于预计算数据包、**无需 GPU**;仅 §3e 背后的从头向量用到了集群。
- **通用性结论**:方法在 Llama 上通用、复刻差别很小;"差异"专指 Llama 相对 Sonnet 的行为差异,不是方法/复刻问题。
""")

nb = {"cells":cells,
      "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                  "language_info":{"name":"python","version":"3.x"}},
      "nbformat":4,"nbformat_minor":5}
out = pathlib.Path(__file__).parent/"emotion_concepts_reproduction_zh.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print("wrote", out, "with", len(cells), "cells")
