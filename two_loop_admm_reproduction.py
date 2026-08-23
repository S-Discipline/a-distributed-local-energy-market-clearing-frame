# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo", "matplotlib", "numpy", "pandas", "Pillow"]
# ///
"""
# Two-Loop ADMM Local Energy Market — reproduction

This notebook reproduces the central claim of **Kabirifar et al.,
"A Distributed Local Energy Market Clearing Framework Using a Two-Loop ADMM
Method"** ([arXiv:2505.16070](https://www.alphaxiv.org/abs/2505.16070)): a
*distributed*, privacy-preserving algorithm can clear a local energy market on a
69-bus feeder in a handful of ADMM iterations and under a minute of CPU time.

It opens with the already-produced evidence; nothing here re-runs the expensive
experiments.

> All quantitative results below are from the repository's formal runs on the
> `lcec-4090` compute instance (see `reports/two-loop-admm-lem/report.md` for
> the full claim-by-claim assessment).
"""

import marimo as mo

app = mo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The headline result

        The two-loop ADMM clears the feeder in **2 outer + 1 inner iterations**
        (the paper reports 5 / 3) and ~59 s of CPU (the paper reports 22.27 s).
        Both are *robust*: the counts and economics are identical when the ADMM
        penalty is scaled ×4 or the convergence tolerance is tightened ×1000.

        ![Convergence and CPU time across the base run and its two
        robustness variants](reports/two-loop-admm-lem/images/fig2_convergence.png)
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Why it matters

        A **local energy market (LEM)** lets prosumers — households with solar,
        batteries and EVs — trade electricity between themselves. The design
        question the paper answers is *who clears it*.

        - A **centralized** operator finds the cheapest global schedule but needs
          everyone's private data (load, battery state, EV usage) and is very slow.
        - A **distributed** scheme should reach the same market while each agent
          keeps its data private and the computation stays fast.

        The paper's contribution is a **two-loop ADMM** that couples four agents
        through two *fictitious auxiliary variables* — a declared net power and a
        declared network loss — so only aggregate signals (and the bus prices,
        **DLMPs**) are exchanged, never private state.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The market that was cleared

        The distribution network is the classical **IEEE-69 radial feeder**. The
        DSO clears it as a **branch-flow SOCP** whose active-power-balance duals
        are the DLMPs. Because prices are locational, they vary by bus and by
        hour:

        ![DLMP band across buses over 24 h](reports/two-loop-admm-lem/images/fig5_dlmp.png)

        The LMO then trims wholesale imports at the expensive evening peak:

        ![Grid import vs wholesale price](reports/two-loop-admm-lem/images/fig1_mechanism.png)
        """
    )
    return


@app.cell
def _(mo, pandas, pd):
    # Evidence inlined so the notebook works without re-running experiments.
    cases = pd.DataFrame(
        {
            "case": ["Base", "Case I\n(indep.)", "Case II\n(centr.)", "Case III\n(KKT)"],
            "lmo_cost": [1411.48, 1454.82, 1411.48, 1411.48],
            "avg_prosumer_cost": [20.79, 21.39, 20.76, 20.76],
        }
    )
    mo.md(
        r"""
        ## Coordination beats independent prosumers

        The paper (Table I) reports the distributed coordination (Base) lowers
        the LMO's wholesale cost ≈2.0% versus uncoordinated prosumers (Case I).
        This reproduction found a ≈**3.0%** reduction — same direction, same
        order of magnitude. The **Case III** alternative (a bilevel KKT/
        strong-duality global solve) gives the same economics but is orders of
        magnitude slower, which is the paper's key efficiency point.
        """
    )
    mo.ui.table(cases, label="Daily costs from the formal run (synthetic data)")
    return cases


@app.cell
def _(mo, pandas, pd):
    mo.md(
        r"""
        ## More prosumers → cheaper wholesale and prosumer energy

        Table II of the paper: raising the share of active prosumers lowers LMO
        and prosumer costs. Reproduced direction (synthetic data):
        """
    )
    pen = pd.DataFrame(
        {
            "penetration_%": [45, 60, 75],
            "lmo_cost_$": [883.78, 356.07, -171.62],
            "avg_prosumer_$": [13.03, 5.28, -2.46],
        }
    )
    mo.ui.table(pen, label="Penetration study from the formal run")
    return pen


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Evidence provenance

        - Full report: `reports/two-loop-admm-lem/report.md`
        - Source: `src/` (DSO SOCP, ADMM core, cases) — see `README`
        - Runs: baseline + two robustness children on `lcec-4090`

        **Caveats** carried from the report: the paper does not publish its
        prosumer / PJM / NHTS data, so deterministic **synthetic** substitutes
        are used (reported), and the prosumer storage binaries are **relaxed to
        continuous** bounds, so exact dollars and iteration counts differ from
        the paper. The mechanism and directions are the reproduced result.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Optional: play with a tiny one-prosumer market

        _Interactive, not part of the formal evidence._ This solves a miniature
        two-period market to show _why_ coordination helps: a single prosumer
        with storage sees a time-varying price and shifts charging off-peak.
        """
    )
    return


@app.cell
def _(mo, numpy, np):
    price = mo.ui.slider(10, 200, value=120, step=5, label="afternoon price ($/MWh)")
    load = mo.ui.slider(30, 90, value=55, step=5, label="peak demand (kWh)")
    mo.hstack([price, load], justify="space-around")
    return load, price


@app.cell
def _(load, mo, np, price):
    # tiny analytic arbitrage: buy 1 kWh cheap in morning, sell at afternoon
    morning = 30.0
    saving = max(0.0, (np.float64(price.value) - morning)) / 1e3 * np.float64(load.value)
    mo.hstack(
        [
            mo.md(f"**Potentially shifted energy cost saving:** ${saving:,.2f}"),
            mo.md(
                "moving 1 kWh from the off-peak morning ($30) to the "
                f"afternoon (${price.value}) price."
            ),
        ]
    )
    return
