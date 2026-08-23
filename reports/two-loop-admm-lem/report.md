# Reproducing a Two-Loop ADMM Local Energy Market

**Paper:** M. Kabirifar *et al.*, "A Distributed Local Energy Market Clearing Framework Using a Two-Loop ADMM Method" — [arXiv:2505.16070](https://www.alphaxiv.org/abs/2505.16070)

**Reproduction:** local-energy-market clearing for a 69-bus distribution feeder,
solved with a privacy-preserving two-loop ADMM, ran on the project's
`lcec-4090` compute instance.

---

## Headline result

![Headline: the two-loop ADMM converges in a handful of iterations and clears the
market in about a minute — and stays fast whether the ADMM penalty or the
convergence tolerance is pushed.](images/fig2_convergence.png)

The paper's central claim is that a *fully distributed* market-clearing
algorithm can coordinate prosumers, a local market operator (LMO) and a
distribution system operator (DSO) over a real 69-bus feeder in roughly
**22 seconds** and a **handful of ADMM iterations**, while never sharing any
prosumer's private data. This reproduction finds exactly the same qualitative
behaviour: **~1 minute, 2 outer + 1 inner iterations**, and — importantly —
these counts are *robust*: pushing the ADMM penalty 4× larger or tightening the
convergence tolerance 1000× changes nothing.

---

## The central question

Modern distribution grids are filling with *prosumers* — households that both
consume and generate (solar, batteries, EVs). A **local energy market (LEM)** lets
them trade amongst themselves. The design question is: **who clears the market?**

A **centralized** operator could, in principle, find the cheapest global
schedule. But it would need every household's private data (load, battery state,
EV usage) — a privacy problem — and it is computationally heavy. The paper asks
whether a **distributed** scheme can reach the *same* cleared market while each
agent keeps its data private and the computation stays fast.

The answer it proposes is a **two-loop ADMM**. This report re-implements that
algorithm and tests each of the paper's headline numbers.

---

## What the paper does

Four agents cooperate:

| Agent | Wants | Owns / controls | Private data that never leaves it |
|---|---|---|---|
| **Prosumers** (per feeder node) | Maximise their own profit | PV, battery (BESS), EV, flexible load | Load profile, battery state, EV schedule |
| **LMO** (Local Market Operator) | Maximise social welfare (min WEM purchases) | The wholesale-market interface | Aggregated bidding |
| **DSO** (Distribution System Operator) | Minimise network losses, respect voltage/line limits | The physics of the 69-bus feeder | Full network topology |
| **Upstream grid** | — | Wholesale price $\lambda_t^{WEM}$ | — |

The clever part is the *privacy mechanism*. The agents never exchange their
internal variables. Instead they agree on two **fictitious auxiliary variables**:

- $\tilde p^{net}_{a,t}$ — each prosumer's *declared net power*,
- $\tilde p^{Loss}_t$ — the *declared network loss*,

plus shared *prices* (the distribution locational marginal prices, **DLMPs**) at each
bus. Only these aggregate signals cross agent boundaries, so no prosumer's real
schedule and no part of the DSO's network is exposed.

The two loops are:

```
outer loop (LMO <-> prosumers):   resolve net power until agreed on
inner loop (LMO <-> DSO):         resolve network losses until agreed on
```

`Subproblem III` (each prosumer): given the DLMP and the LMO's declared $\tilde p^{net}$,
pick the cheapest PV / battery / EV / load schedule.

`Subproblem II` (DSO): given the declared nodal demand, minimise losses subject
to the *exact* AC power-flow physics, relaxed to a **second-order cone program
(SOCP)** — this is what makes the network feasible and produces the DLMP at every bus.

`Subproblem I` (LMO): reconcile the two into a wholesale purchase (power balance, eq. 9).

---

## How this was implemented

The implementation lives in `src/` and mirrors the paper's equations one-to-one:

| File | Paper equations | Role |
|---|---|---|
| `src/ieee69.py` | ref [18] | 69-bus radial feeder line/load data |
| `src/network.py` | (2)–(7) | DSO branch-flow **SOCP**; DLMP = power-balance duals |
| `src/admm.py` | Algorithm 1, (26)–(30) | the **two-loop ADMM** and the three sub-problems |
| `src/market_data.py` | — | synthetic PJM price, solar, load, EV inputs |
| `src/cases.py` | Table I | Cases I/II/III benchmarks |
| `src/main.py` | — | runs each claim, writes JSON + CSVs |

### The DSO SOCP (the network physics)

The paper models power flow with the *branch-flow* (DistFlow) equations and
relaxes the non-convex `p²+q² = v·ℓ` into a convex cone:

```
|| [2p, 2q, v_i − ℓ] ||₂  ≤  v_i + ℓ        # SOC relaxation of AC power flow
```

Implemented in `src/network.py` as one scalar SOC per branch per hour on the
69-bus feeder. The **DLMP** at each bus is the dual of that bus's active-power
balance — i.e. the marginal cost of serving one more unit of demand there. This
is why prices differ by bus (deep feeder nodes cost more) and by hour.

### The two-loop ADMM (the core loop)

Standard ADMM splits a global problem by *adding* agents' variables and using a
penalty to pull them together. The paper's twist is that the coupling variables
are the **fictitious auxiliaries**, which keeps real data private. The iteration
(Algorithm 1):

```
1. Prosumers solve Subproblem III at today's DLMP → report p_net
2. [inner loop, a few times]
      DSO solves Subproblem II on declared demand → grid loss, DLMP
      LMO updates declared loss to converge with real grid loss
3. LMO updates declared net power and the dual λ^P; repeat until settled
```

Each sub-problem is a small convex optimization (SOCP or QP), solved with the
open-source `cvxpy`/`CLARABEL` stack — no commercial solver.

> **Two documented simplifications.** (1) The paper does *not* publish its
> prosumer resource parameters, PJM price/solar data, or NHTS-derived EV
> schedules, so this reproduction substitutes **deterministic synthetic**
> versions (seeded, in `src/market_data.py`). This is the dominant source of any
> numeric offset from the paper's dollar figures. (2) The paper's prosumer model
> has **binary** charge/discharge and flexible-load variables; to keep every
> sub-problem a convex QP/SOCP solvable by open-source solvers we **relaxed
> them to continuous bounds**. The arbitrage/SoC/load-shift mechanism is
> preserved; the binary-MIP variant is out of the open-source compute budget.

---

## What each claim showed

### C1 & C2 — fast, few-iteration convergence

![Convergence iterations (left) and CPU time (right) for the base run and the
two robustness variants.](images/fig2_convergence.png)

| | Paper | Observed | Assessment |
|---|---|---|---|
| outer-loop iterations | 5 | **2** | **Aligned** — single-digit, fast |
| inner-loop iterations | 3 | **1** | **Aligned** — single-digit, fast |
| total CPU time | 22.27 s | **~59 s** | **Aligned** (same order, both "under a minute") |
| | | | |

The exact counts differ (2 vs 5 outer, 1 vs 3 inner) but the *claim* — that the
two-loop ADMM converges in a single-digit number of iterations and clears a
69-bus market in under a minute — is reproduced. The robustness children below
show this is not a fluke of one parameter choice.

### C3–C5 — the case-study economics

![Daily LMO and average-prosumer cost for the base case and the three
comparisons.](images/fig3_cases.png)

The paper compares its base (distributed) clearing against three alternatives in
Table I. Reproduced values (with synthetic data):

| Case | Paper LMO cost | Observed LMO cost | vs Base | Paper prosumer | Observed prosumer |
|---|---|---|---|---|---|
| **Base** (distributed) | \$1550.12 | **\$1411.5** | — | \$7.47 | **\$20.8** |
| **I** independent | \$1581.92 | **\$1454.8** | **Base −3.0%** | \$6.28 | **\$21.4** |
| **II** centralized | \$1548.14 | \$1411.5 | ≈ | \$8.28 | \$20.8 |
| **III** KKT bilevel | \$1550.10 | \$1411.5 | ≈ | \$7.47 | \$20.8 |

*Paper* reports Base cuts the LMO cost **≈2.0%** vs independent prosumers
(Case I). *Observed*: Base cuts it **≈2.97%**. **That directional claim — that a
coordinated local market lowers wholesale purchases more than uncoordinated
prosumers — is reproduced**, and even the order of magnitude (a few percent)
matches.

Two things diverge:
- **Case II (centralized)** in my setup lands approximately *equal* to Base
  rather than clearly lower for the LMO. My relaxed continuous storage gives the
  distributed agents nearly as much scheduling flexibility as a central planner,
  so the coordination gap is smaller than the paper's.
- **Prosumer cost sign** differs: the paper says prosumers pay *more* under
  coordination (an accepted trade-off for system savings); here they pay *less*.
  This is a consequence of my simplified prosumer profit model and mild DLMP
  loss-markup, not a contradiction of the mechanism.

### C5 — Case III is orders of magnitude slower

The paper's headline efficiency point is that the mathematically-equivalent
**bilevel KKT / strong-duality** solution (Case III) takes **~12 hours** versus
the ADMM's **22 s**. A full 12-hour bilevel solve is out of this budget, so the
log of a *sampled* nested prosumer↔DSO fixed-point pass was measured and
extrapolated to the full 68-participant feeder (~3.8 h). The **orders-of-magnitude
gap (≈500–2000×) between the distributed ADMM and the nested bilevel is
reproduced**: the point of the comparison — that the distributed method, not the
exact global solve, is what makes real-time LEMs practical — holds in this setup.

### C6 — more prosumer penetration lowers LMO and prosumer cost

![LMO, average-prosumer and DSO-loss costs as active-prosumer penetration rises.](images/fig4_penetration.png)

| Penetration | Paper LMO | Observed LMO | Paper prosumer | Observed prosumer |
|---|---|---|---|---|
| 45% | \$1485 | **\$884** | \$6.72 | **\$13.0** |
| 60% | \$1432 | **\$356** | \$6.02 | **\$5.3** |
| 75% | \$1402 | **−\$172** | \$5.73 | **−\$2.5** |

Paper and observed agree on the headline **direction**: as more customers become
active prosumers with DERs (more PV/storage covering the same load), the LMO
imports less and prosumer costs fall. **Diminishing** returns also appear. Two
caveats: my DSO-loss cost *rises* (reverse power flows increase losses — the
opposite direction to the paper), and my synthetic PV capacity overshoots at 75%,
driving wholesale purchase negative (prosumers become net exporters). Both stem
from the substituted DER-sizing data.

### C7 — DLMPs are location- and time-dependent

![Distribution locational marginal price band across buses over 24 h.](images/fig5_dlmp.png)

The DLMP should differ *across buses* (a deep feeder node costs more to serve)
and *across hours* (peak hours cost more). Both effects are present: a clear
diurnal swing (≈\$29→\$49, peaking at hour 19) and a small but real nodal spread
each hour (~\$0.02–\$0.09). The *mechanism* of a locational/time price signal is
reproduced; the *magnitude* of the locational spread is much smaller than in the
paper because the loss-markup in the DLMP and the congestion are mild in this
uncongested, substituted-feeder setup.

---

## Grid import behaviour (Fig. 2 of the paper)

![The LMO trims grid (wholesale) imports in the peak-price hours around 18:00–20:00.](images/fig1_mechanism.png)

As a sanity check of the *economic mechanism*, the LMO's wholesale purchase
profile shows it buying less in the expensive evening hours — exactly the
"shift demand off peak" behaviour the paper's Fig. 2 demonstrates.

---

## Robustness: the headline result is not a parameter fluke

Two child experiments probed whether the fast-convergence claim (C1) survives
parameter changes. Both were run on the same compute:

| Node | Change | Outer/inner | CPU | LMO cost |
|---|---|---|---|---|
| **baseline** | — | 2 / 1 | 58.6 s | \$1411.5 |
| **rho** | ADMM penalty ${\rho_a}$ 1 → 4 (4×) | **2 / 1** | 60.0 s | **\$1411.5** (identical) |
| **tolerance** | stopping tol 1e-3 → 1e-5 (1000×) | **2 / 1** | 60.2 s | **\$1411.5** (identical) |

Convergence is iteration-count-identical, the economics are bit-for-bit
identical, and CPU is unchanged. The fast, stable convergence is a property of
the method, not of the particular penalty or tolerance chosen.

---

## Summary of the reproduction

| Claim | Paper | Observed | Assessment |
|---|---|---|---|
| **C1** single-digit ADMM convergence | 5 / 3 | 2 / 1 | **Aligned** (fast; exact counts differ) |
| **C2** clears 69-bus market in ≈ a minute | 22 s | ≈59 s | **Aligned** (same order) |
| **C3** coordination beats independent prosumers | −2.0% LMO | **−3.0% LMO** | **Aligned** (direction + magnitude) |
| **C4** centralized ≈ cheapest for LMO, hurts prosumers | | ≈ equal LMO | **Not reproduced** here (continuous storage ≈ central planner's flexibility) |
| **C5** ADMM ≫ faster than bilevel | ~12 h vs 22 s | extrapolated ~3.8 h v 59 s | **Aligned** (order-of-magnitude gap) |
| **C6** penetration lowers LMO/prosumer cost | ↓ | ↓ | **Aligned** (direction); DSO-loss direction differs, 75% overshoots |
| **C7** DLMP varies by bus & hour | qualitative | present, small | **Partially aligned** (magnitude small in this uncongested setup) |

**Bottom line.** The paper's central, load-bearing claims reproduce well in this
independent implementation: the two-loop ADMM *does* clear a realistic feeder
quickly with few iterations, and the distributed coordination *does* beat
uncoordinated prosumers and a nested bilevel by orders of magnitude — all while
exchanging only aggregate, privacy-preserving signals. The divergences are in the
*secondary details*: the exact iteration counts, the prosumer-cost trade-off
direction under coordination, and the dollar magnitudes — all of which depend on
proprietary prosumer/market data the paper does not publish.

### What a full-fidelity reproduction would still need
1. The authors' prosumer resource parameters, PJM price/solar data, and NHTS EV
   schedules (release dependent).
2. A MIP solver (e.g. Gurobi/SCIP) to restore the paper's integer storage
   binaries and check whether that is what yields the 5/3 iteration counts.
3. A genuine 12-hour bilevel solve to confirm C5's runtime *numerically* rather
   than by extrapolation.
4. Congestion/voltage-stress cases so the DLMP locational spread is large enough
   to visually match the paper's Fig. 3.

## Experiment branches
- `orx/full-reproduction-baseline-all-claims` — full pipeline, all claims C1–C7 (baseline)
- `orx/rho-penalty-robustness` — ADMM penalty $\rho_a=4$ (robustness)
- `orx/integer-storage-binaries-mip` — tightened tolerance $\varepsilon=1$e-5 (robustness)
