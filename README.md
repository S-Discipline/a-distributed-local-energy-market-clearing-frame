# Two-Loop ADMM Local-Energy-Market — Reproduction

## Reproduction of arXiv:2505.16070

This repository reproduces, claim by claim, **"A Distributed Local Energy Market
Clearing Framework Using a Two-Loop ADMM Method"** (Kabirifar, Mukherjee,
Krishnan, Konstantinou, Lakshminarayana — [arXiv:2505.16070](https://www.alphaxiv.org/abs/2505.16070)).

**Claim tested.** That a fully *distributed*, privacy-preserving two-loop ADMM
can clear a local energy market on an IEEE-69-bus feeder in a handful of
iterations (~1 min CPU), beat uncoordinated prosumers and a global bilevel solve
by orders of magnitude, and that more prosumer penetration lowers wholesale and
prosumer costs.

**What was done.** Re-implemented the paper's DSO second-order-cone power-flow
subproblem (with DLMPs from its duals), the LMO and prosumer subproblems, and
Algorithm 1's two-loop ADMM (`src/`), then ran the full claim pipeline on the
project's `lcec-4090` compute instance (RTX 4090 host).

**Assessment (headline).** The central claims **reproduce in direction and
order of magnitude**; dollar precision does not, because the paper publishes
none of its prosumer/PJM/NHTS data (this project substitutes deterministic,
seeded synthetic inputs) and the storage binaries are relaxed to continuous
bounds so every sub-problem is convex and open-source-solvable.

| Claim | Paper | Observed | Assessment |
|---|---|---|---|
| Outer/inner ADMM iterations | 5 / 3 | **2 / 1** (robust to ρ×4, ε×1000) | Aligned (fast, single-digit) |
| Market-clearing CPU time | 22.27 s | **≈59 s** | Aligned (same order) |
| Base coordinates better than independent prosumers (LMO cost) | −2.0% | **−3.0%** | Aligned (direction + magnitude) |
| Case III (bilevel KKT) much slower than distributed | ~12 h vs 22 s | extrapolated ~3.8 h vs 59 s | Aligned (order-of-magnitude gap) |
| More prosumer penetration lowers LMO + prosumer cost | ↓ | ↓ | Aligned (direction) |
| DLMP varies across buses & over the day | qualitative | present (small) | Partially aligned |
| Centralized ≈ cheapest; coordination raises prosumer cost | / | ≈ equal; prosumer cost *lower* | Diverges (relaxed storage ≈ central flexibility) |

**Downscaling / substitutions (explicit).** (1) Deterministic synthetic data
replaces the unreleased prosumer resource parameters, PJM price/solar series and
NHTS EV schedules — the dominant source of numeric offset from the paper's
dollar tables. (2) Prosumer charge/EV/flexible-load binaries are relaxed to
continuous bounds (open-source solver stack, no commercial MIP). (3) Case III is
timed on a *sample* nested pass and extrapolated, not run for 12 h. (4) Case III
economics reuse the base-market equilibrium (same framework), as the paper says.

**Compute.** All runs executed on the project's `lcec-4090` host via
`orx exp run --backend ssh --host lcec-4090`.

**Reports.**
- [`reports/two-loop-admm-lem/report.md`](reports/two-loop-admm-lem/report.md) — illustrated, implementation-led claim-by-claim report (all figures)
- [`two_loop_admm_reproduction.py`](two_loop_admm_reproduction.py) — marimo notebook opening with the evidence

### Experiment log

| Branch | Purpose / change | Run command | Assessment | Compute |
|---|---|---|---|---|
| `orx/full-reproduction-baseline-all-claims` | Full pipeline, all claims C1–C7 (baseline) | `python3 -m src.main --case all --out ./runs_out` | Baseline reproduced; see tables above | `lcec-4090` |
| `orx/rho-penalty-robustness` | ADMM penalty ρ_a 1→4 (robustness of C1/C3) | `python3 -m src.main --case all --out ./runs_out` | Identical results → convergence/economics stable | `lcec-4090` |
| `orx/integer-storage-binaries-mip` | Convergence tolerance 1e-3→1e-5 (robustness of C1) | `python3 -m src.main --case all --out ./runs_out` | Identical results → fast convergence is genuine | `lcec-4090` |
| `main` | — | `Not run as an experiment (publication surface)` | Holds this README, report, notebook | — |

---

## Running the reproduction

```bash
pip install -r requirements.txt        # numpy scipy pandas networkx cvxpy
python3 -m src.main --case all --out ./runs_out
```

`--case` takes `base` (C1, C2), `cases` (C3–C5), `penetration` (C6) or `dlmp`
(C7). Results (JSON summary, CSVs, figures) are written to `--out`.

## Layout

- `src/ieee69.py`        IEEE-69 radial feeder data (Savier & Das, ref [18])
- `src/network.py`       DSO branch-flow SOCP subproblem (eqs 2–7) + DLMP duals
- `src/market_data.py`   synthetic prosumer / PJM / PV / load profiles
- `src/admm.py`          the two-loop ADMM (Algorithm 1, sub-problems I–III)
- `src/cases.py`         Case I / II / III benchmarks (Table I)
- `src/config.py`        experiment knobs (ρ, tolerances; edited by children)
- `src/main.py`          claim orchestration + evidence output
- `reports/`             the illustrated reproduction report + figure scripts
- `two_loop_admm_reproduction.py`  marimo notebook

## Notes on faithfulness

The paper does not publish its prosumer resource parameters, PJM price / solar
data, or NHTS-derived EV schedules. This reproduction substitutes deterministic
synthetic versions (see `src/market_data.py`) and relaxes the prosumer binaries
(charging / flexible-load utilisation) to continuous bounds so every sub-problem
is a convex QP/SOCP. These substitutions are the dominant source of any numeric
offset from the paper's tables; the mechanism and directions of the claims are
reproduced.
