"""Experiment configuration knob (children edit defaults; run command is fixed).

Keeps the fixed run command `python3 -m src.main --case all --out ./runs_out`
working on every node while letting child experiments vary the ADMM penalty
radius, convergence tolerance or restore integer binaries simply by editing the
defaults here and committing the change.
"""

# ADMM penalties appearing in the augmented Lagrangian (paper calls them rho_a,
# rho'_a). Baseline uses rho_a = 1.0, rho_l = 1.0.
RHO_A = 1.0
RHO_L = 1.0

# Restore the paper's integer charge/discharge + flexible-load utilisations?
# Baseline FALSE (relaxed to continuous, convex sub-problems). TRUE restores
# binary variables at the cost of a MIP-capable solver (e.g. SCIP/GUROBI).
MIP_BINARIES = False

# Solver options tuning (tightened here: 1e-3 -> 1e-5 to probe claim C1).
OUTER_EPS = 1e-5
INNER_EPS = 1e-5
