# Two-Loop ADMM Local-Energy-Market Reproduction

Reproduction of: M. Kabirifar, B. Mukherjee, S. G. Krishnan, C. Konstantinou,
S. Lakshminarayana, **"A Distributed Local Energy Market Clearing Framework
Using a Two-Loop ADMM Method"** (arXiv:2505.16070).

## Run

```bash
python3 -m src.main --case all --out ./runs_out
```

`--case` takes `base` (C1, C2), `cases` (C3-C5), `penetration` (C6) or
`dlmp` (C7). Results (JSON summary, CSVs, figures) are written to `--out`.

## Dependencies

```bash
pip install -r requirements.txt     # numpy scipy pandas networkx cvxpy
```

## Layout

- `src/ieee69.py`        IEEE-69 radial feeder data (Savier & Das, ref [18])
- `src/network.py`       DSO branch-flow SOCP subproblem (eqs 2-7) + DLMP duals
- `src/market_data.py`   synthetic prosumer / PJM / PV / load profiles
- `src/admm.py`          the two-loop ADMM (Algorithm 1, sub-problems I-III)
- `src/cases.py`         Case I / II / III benchmarks (Table I)
- `src/main.py`          claim orchestration + evidence output

## Notes on faithfulness

The paper does not publish its prosumer resource parameters, PJM price / solar
data, or NHTS-derived EV schedules. This reproduction substitutes deterministic
synthetic versions (see `src/market_data.py`) and relaxes the prosumer binaries
(charging / flexible-load utilisation) to continuous bounds so every
sub-problem is a convex QP/SOCP. These substitutions are the dominant source of
any numeric offset from the paper's tables; the mechanism and directions of the
claims are reproduced.
