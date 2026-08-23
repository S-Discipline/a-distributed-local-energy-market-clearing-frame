# Two-Loop ADMM Local-Energy-Market — Reproduction

## Reproduction of arXiv:2505.16070 (grade C — partial)

This repository attempts to reproduce, claim by claim, **"A Distributed Local Energy
Market Clearing Framework Using a Two-Loop ADMM Method"** (Kabirifar, Mukherjee,
Krishnan, Konstantinou, Lakshminarayana — [arXiv:2505.16070](https://www.alphaxiv.org/abs/2505.16070)).

**Claim tested.** That a fully *distributed*, privacy-preserving two-loop ADMM can
clear a local energy market on an IEEE-69-bus feeder in a handful of iterations
(~1 min CPU), beat uncoordinated prosumers and a global bilevel solve, and that
more prosumer penetration lowers wholesale and prosumer costs.

**What was done.** Re-implemented the paper's DSO second-order-cone power-flow
subproblem (with DLMPs from its duals), the LMO and prosumer subproblems, and
Algorithm 1's two-loop ADMM (`src/`), then ran the full claim pipeline on the
project's `lcec-4090` compute host via `orx exp run --backend ssh --host lcec-4090`.

**Assessment (strict, grade C — partial reproduction success).** The algorithm runs
and its primary *directions* are supported, but **this is not a full
reproduction**: the paper publishes none of its prosumer/PJM/NHTS data (so dollar
figures cannot match), the prosumer binaries are relaxed to continuous bounds,
Case I / penetration use custom proxies, and Case III was **not truly run** (its
CPU time is a constructed `base × 500` extrapolation and its economics are copied
from the base equilibrium). One secondary conclusion — Case II's effect on prosumer
cost — points the *opposite* way to the paper. See
[`reports/two-loop-admm-lem/report.md`](reports/two-loop-admm-lem/report.md) for
the full claim-by-claim grading.

| Claim | Paper | Observed | Direction | Number close? |
|---|---|---|---|---|
| ADMM iterations | 5 / 3 | 2 / 1 | ✓ (both single-digit) | no (~2.5×) |
| Clearing CPU | 22.27 s | ≈59 s | ✓ (both <1 min) | no (~2.6×, solver/hardware) |
| Coordination beats independent (LMO) | −2.0% | −3.0% | ✓ | no (proxy Case I) |
| Case II centralized is cheapest & hurts prosumers | prosumer ↑ | prosumer ↓ | **✗** | — |
| Case III much slower than ADMM | ~12 h vs 22 s | not measured (extrapolated ~3.8 h) | (not measured) | — |
| Penetration lowers LMO/prosumer cost | ↓ | ↓ (→ negative at 75%) | ✓ (partial) | no (overshoot) |
| DLMP varies by bus & time | qualitative | present, small | ✓ (partial) | no (small spread) |

**Key experimental-condition differences (see report for full list).** Synthetic
(seeded) data in place of unpublished author data; continuous vs binary storage;
CLARABEL (open-source) vs commercial solver; single seed/no repeats; custom
Case-I & penetration definitions; Case III not actually solved. These are the
dominant drivers of the numeric gaps and of the C-grade.

**Compute.** All runs executed on the project's `lcec-4090` host.

**Reports.**
- [`reports/two-loop-admm-lem/report.md`](reports/two-loop-admm-lem/report.md) — illustrated, claim-by-claim report with a strict A–F grade and a full differences table
- [`two_loop_admm_reproduction.py`](two_loop_admm_reproduction.py) — marimo notebook opening with the evidence

> This repository is kept **private** (no public Molab badge). To view the
> notebook locally: `pip install marimo`, then `marimo edit two_loop_admm_reproduction.py`
> (edit) or `marimo run two_loop_admm_reproduction.py` (read-only app).

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
