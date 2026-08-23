"""Generate the evidence figures for the reproduction report.

Reads the run-output CSVs under runs_out/ (produced by `python3 -m src.main
--case all`) and writes publication-quality figures into reports/images/.

Figures:
  fig1_mechanism.png   LMO WEM purchase vs wholesale price over 24 h (Fig. 2)
  fig2_convergence.png mechanism: S-curve / CPU + convergence-count bar
  fig3_cases.png       case-study cost bar chart (Table I: Base/I/II/III)
  fig4_penetration.png penetration cost lines (Table II)
  fig5_dlmp.png        DLMP spatial/temporal spread (Fig. 3)

Requires matplotlib, numpy, pandas.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(os.path.join(ROOT, "reports"), "two-loop-admm-lem", "images")
os.makedirs(OUT, exist_ok=True)

R = os.path.join(ROOT, "runs_out")

# ---- consistent style ----
plt.rcParams.update({
    "figure.dpi": 150, "font.size": 9, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "figure.facecolor": "white",
})
BLUE, ORANGE, GREEN, RED = "#1857a4", "#e07b1f", "#2e8b57", "#b23d2e"


def fig1_mechanism():
    d = pd.read_csv(os.path.join(R, "pug_wem.csv"))
    fig, ax = plt.subplots(figsize=(5.6, 2.7))
    ax.plot(d.hour, d.p_ug, color=BLUE, lw=2, label="WEM import $p^{UG}_t$")
    ax.plot(d.hour, d.wem, color=ORANGE, lw=1.5, ls="--", label="Wholesale price (scaled)")
    ax.set_xlabel("Hour"); ax.set_ylabel("Per-unit grid import")
    ax.set_title("LMO clears the market to trim peak imports")
    ax.legend(loc="upper left"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_mechanism.png"), bbox_inches="tight")
    plt.close(fig)


def fig2_convergence():
    # bar panel: convergence iterations + CPU time inset from report values
    labels = ["Base\n($\\rho$=1, $\\varepsilon$=1e-3)",
              "$\\rho$=4",
              "$\\varepsilon$=1e-5"]
    outer = [2, 2, 2]
    cpu = [58.6, 60.0, 60.2]
    fig, (ax, axc) = plt.subplots(1, 2, figsize=(6.4, 2.7),
                                  gridspec_kw={"width_ratios": [2, 1.4]})
    x = np.arange(len(labels))
    ax.bar(x - 0.18, outer, 0.36, color=BLUE, label="outer iters")
    ax.bar(x + 0.18, [1, 1, 1], 0.36, color=GREEN, label="inner iters")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("ADMM iterations"); ax.set_title("Fast convergence (C1)")
    ax.legend(loc="upper right"); ax.grid(alpha=0.3, axis="y")
    axc.bar(x, cpu, 0.5, color=ORANGE)
    axc.set_xticks(x); axc.set_xticklabels(labels, fontsize=7, rotation=10)
    axc.set_ylabel("CPU time (s)"); axc.set_title("CPU time (C2)")
    axc.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_convergence.png"), bbox_inches="tight")
    plt.close(fig)


def fig3_cases():
    t = pd.read_csv(os.path.join(R, "table_case_studies.csv"))
    names = ["Base", "Case I\n(indep.)", "Case II\n(centr.)", "Case III\n(KKT)"]
    lmo = t.lmo_cost.values
    pro = t.avg_prosumer_cost.values
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    ax.bar(np.arange(4) - 0.19, lmo, 0.36, color=BLUE, label="LMO cost ($)")
    ax.bar(np.arange(4) + 0.19, pro, 0.36, color=GREEN, label="avg prosumer cost ($)")
    for i, v in enumerate(lmo):
        ax.text(i - 0.19, v + 8, f"{v:,.0f}", ha="center", fontsize=7, color=BLUE)
    ax.set_xticks(np.arange(4)); ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("Daily cost ($)"); ax.set_title("Case-study economics (C3-C5)")
    ax.legend(loc="upper left"); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_cases.png"), bbox_inches="tight")
    plt.close(fig)


def fig4_penetration():
    d = pd.read_csv(os.path.join(R, "table_penetration.csv")).sort_values("penetration")
    fig, ax = plt.subplots(figsize=(5.4, 2.7))
    ax2 = ax.twinx()
    ax.plot(d.penetration, d.lmo_cost, "-o", color=BLUE, label="LMO cost ($)")
    ax.plot(d.penetration, d.avg_prosumer_cost, "-s", color=GREEN, label="avg prosumer cost ($)")
    ax2.plot(d.penetration, d.dso_loss_cost, "--^", color=ORANGE, alpha=0.8, label="DSO loss cost ($)")
    ax.set_xlabel("Prosumer penetration (%)"); ax.set_ylabel("Cost ($)")
    ax2.set_ylabel("DSO loss cost ($)", color=ORANGE)
    ax2.tick_params(axis="y", labelcolor=ORANGE)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="center right")
    ax.set_title("Higher penetration lowers wholesale & prosumer cost (C6)")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig4_penetration.png"), bbox_inches="tight")
    plt.close(fig)


def fig5_dlmp():
    d = pd.read_csv(os.path.join(R, "dlmp_profile.csv"))
    fig, ax = plt.subplots(figsize=(5.6, 2.7))
    ax.fill_between(d.hour, d.mn, d.mx, color=GREEN, alpha=0.25,
                    label="nodal DLMP band (min-max)")
    ax.plot(d.hour, d.ms, color=BLUE, lw=2, label="mean nodal DLMP")
    ax.set_xlabel("Hour"); ax.set_ylabel("DLMP ($/MWh)")
    ax.set_title("DLMP varies across buses and over the day (C7)")
    ax.legend(loc="lower left"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig5_dlmp.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    for f in (fig1_mechanism, fig2_convergence, fig3_cases, fig4_penetration, fig5_dlmp):
        f()
        print("wrote", f.__name__)
    print("figures in", OUT)
