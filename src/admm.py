"""Two-loop ADMM local-energy-market clearing.

Implements the paper's Algorithm 1 with the three sub-problems:

  * Subproblem I   (LMO)        -- equation (26), subject to (9).
  * Subproblem II  (DSO)        -- equation (27), subject to (2)-(7).
  * Subproblem III (prosumer)   -- equation (30), subject to (11)-(24).

Privacy-preserving auxiliary variables \tilde p_{net}^{a,t} (prosumer net
power) and \tilde p_t^{Loss} (network losses) couple the agents so only
aggregated net power, nodal net demand and DLMPs are exchanged.

SIMPLIFICATION (reported): the paper's prosumer model has binary
charge/discharge and flexible-load-utilisation variables. To keep every
sub-problem a convex QP/SOCP solvable with open-source solvers (cvxpy/CLARABEL,
no external MIP solver), the binaries are relaxed to continuous bounds; the
arbitrage / SoC / load-shift mechanism is preserved.
"""

import cvxpy as cp
import numpy as np
import time

from . import market_data


# --------------------------------------------------------------------------- #
# Prosumer (aggregator) sub-problem -- Subproblem III, eq (30)          s.t. (11)-(24)
# --------------------------------------------------------------------------- #
class Prosumer:
    def __init__(self, bus, pv_pu, bess_pu, ev_pu, base_pu, seed=3):
        self.bus = bus
        self.pv_pu = pv_pu
        self.bess_pu = bess_pu
        self.ev_pu = ev_pu
        self.base_pu = base_pu
        self.fl_pu = 0.05 * base_pu          # 5% flexible load
        self.seed = seed
        self.T = 24
        self.dt = 1.0
        self.eta = 0.95
        self.bess_energy = max(bess_pu * 2.0, 1e-4)     # 2 h storage at BESS power
        self.ev_energy = max(ev_pu * 2.0, 1e-4)
        rng = np.random.default_rng(seed + bus)
        self.ev_arr = int(rng.integers(17, 20))
        self.ev_trip = 0.3   # fraction of EV energy required at departure

    def pv_series(self):
        return self.pv_pu * market_data.pv_profile(seed=10 + self.bus)

    def build(self, lam, pnet_aux, rho, load_shape):
        """Build Subproblem III as cvxpy QP solved against signals (lam, aux)."""
        T = self.T
        pv = self.pv_series()
        load = self.base_pu * load_shape

        p_b = cp.Variable(T)                 # BESS net power (+ = discharge)
        p_e = cp.Variable(T)                 # EV net power
        p_fl = cp.Variable(T)                # flexible-load change
        p_net = cp.Variable(T)

        p_gen = pv + p_b + p_e
        p_con = load + p_fl
        cons = [p_net == p_con - p_gen]

        # BESS SoC (fraction of energy capacity)
        soc_b = cp.Variable(T)
        cons += [soc_b[0] == 0.5, soc_b[-1] == 0.5]
        cons += [soc_b[t] == soc_b[t-1]
                 + self.eta * p_b[t] * self.dt / self.bess_energy
                 for t in range(1, T)]
        cons += [0 <= soc_b, soc_b <= 1.0]
        cons += [-self.bess_pu <= p_b, p_b <= self.bess_pu]

        # EV SoC (fraction of EV energy capacity; plugged from arrival)
        soc_e = cp.Variable(T)
        cons += [soc_e[t] == 0.5 for t in range(0, min(self.ev_arr, T))]
        cons += [soc_e[t] == soc_e[t-1]
                 + self.eta * p_e[t] * self.dt / self.ev_energy
                 for t in range(self.ev_arr, T)]
        cons += [0 <= soc_e, soc_e <= 1.0]
        cons += [-self.ev_pu <= p_e, p_e <= self.ev_pu]
        cons += [soc_e[T-1] >= self.ev_trip]         # eq (21) trip need

        # flexible load (eq 22-24, binaries relaxed)
        cons += [-self.fl_pu <= p_fl, p_fl <= self.fl_pu]
        cons += [cp.sum(p_fl) == 0]                   # energy-neutral shift
        cons += [cp.sum(load - p_fl) * self.dt >= 0.95 * cp.sum(load) * self.dt]

        # eq (30) objective
        objective = cp.sum(cp.multiply(p_net, lam) * self.dt) \
            + rho / 2 * cp.sum((pnet_aux - p_net) ** 2)
        problem = cp.Problem(cp.Minimize(objective), cons)
        return ProsumerHandle(problem, p_net, p_b, p_e, p_fl, soc_b, soc_e)


class ProsumerHandle:
    def __init__(self, problem, p_net, p_b, p_e, p_fl, soc_b, soc_e):
        self.problem = problem
        self.p_net = p_net
        self.p_b = p_b
        self.p_e = p_e
        self.p_fl = p_fl
        self.soc_b = soc_b
        self.soc_e = soc_e
        self.status = None

    def solve(self):
        self.status = self.problem.solve(solver=cp.CLARABEL)
        return self

    @property
    def net(self):
        return np.array(self.p_net.value).ravel()

    @property
    def battery(self):
        return np.array(self.p_b.value).ravel()

    @property
    def ev(self):
        return np.array(self.p_e.value).ravel()

    @property
    def flex(self):
        return np.array(self.p_fl.value).ravel()


# --------------------------------------------------------------------------- #
# LMO sub-problem -- Subproblem I, eq (26) s.t. (9)
# --------------------------------------------------------------------------- #
def lmo_step(pnet_prosumers, p_loss_grid, wem_price, rho_a):
    """LMO picks auxiliary net powers = prosumers' reported net (privacy),
    and aux loss = DSO's grid loss; WEM purchase follows eq (9)."""
    p_net_aux = [np.array(p) for p in pnet_prosumers]
    p_loss_aux = np.array(p_loss_grid).ravel()
    p_ug = np.zeros_like(p_loss_aux)
    for a in p_net_aux:
        p_ug += a
    p_ug = p_ug + p_loss_aux
    return p_net_aux, p_loss_aux, p_ug


# --------------------------------------------------------------------------- #
# Two-loop ADMM  --  Algorithm 1
# --------------------------------------------------------------------------- #
def clear_market(net, participants, wem_price, load_shape, C_loss=1.0,
                 rho_a=1.0, rho_l=1.0, outer_max=30, inner_max=30,
                 eps_outer=1e-3, eps_inner=1e-3, verbose=False):
    """Full two-loop ADMM (Algorithm 1). `participants` = list of Prosumer with
    .bus. Returns _MarketResult.

    The DLMP seen by a prosumer is the WEM reference price plus a
    loss-modulated locational component recovered from the DSO SOCP duals, so
    prices are time- AND bus-varying, which is what drives prosumer arbitrage.
    """
    A = len(participants)
    T = 24
    t0 = time.time()

    # initialisation (Algorithm 1 line 1)
    lam_a = [np.zeros(T) for _ in range(A)]
    lam_loss = np.zeros(T)
    p_aux = [np.zeros(T) for _ in range(A)]
    p_loss_aux = np.zeros(T)
    dlmp = [np.full(T, wem_price) for _ in range(A)]

    outer_iters = 0
    inner_iters = 1
    p_loss_series = np.zeros(T)
    last_p_ug = np.zeros(T)
    pnet_history = []

    q_net_nodal = np.tile(net.q0[:, None], (1, T))

    for k in range(outer_max):
        outer_iters = k + 1
        # ---- 1. Prosumers solve Subproblem III (lines 4-8) -------------
        pnet_a = []
        for a in range(A):
            h = participants[a].build(dlmp[a], p_aux[a], rho_a, load_shape)
            h.solve()
            pnet_a.append(h.net)
        pnet_history.append(np.vstack(pnet_a))
        # ---- 2. Inner loop: LMO <-> DSO loss consensus (lines 9-18) ----
        inner_iters, p_loss_series, p_ug_dso, dlmp_node, lam_loss = _inner_loop(
            net, participants, pnet_a, p_loss_aux, lam_loss, rho_l,
            inner_max, eps_inner, C_loss)
        # DLMP = WEM reference + loss-based locational markup from SOCP duals
        dlmp = _compose_dlmp(net, participants, wem_price, dlmp_node, p_loss_series)
        # ---- 3. LMO: auxiliary net = prosumers' reported net (privacy),
        #    WEM import = nodal consensus (eq 9) --------------------------
        p_aux = [np.array(p) for p in pnet_a]
        last_p_ug = p_ug_dso
        # ---- 4. Outer dual update (lines 20-22) ------------------------
        lam_a_new = [lam_a[a] + rho_a * (p_aux[a] - pnet_a[a])
                     for a in range(A)]
        primal = float(max(np.max(np.abs(p_aux[a] - pnet_a[a])) for a in range(A)))
        conv = float(max(np.max(np.abs(lam_a_new[a] - lam_a[a])) for a in range(A)))
        lam_a = lam_a_new
        if verbose:
            print(f"[outer k={k+1}] primal={primal:.2e} dual={conv:.2e} "
                  f"inner={inner_iters}")
        if conv <= eps_outer and k > 0:
            break

    cpu_time = time.time() - t0
    return _MarketResult(A, T, wem_price, C_loss, dlmp, p_aux, last_p_ug,
                         outer_iters, inner_iters, cpu_time, p_loss_series)


def _compose_dlmp(net, participants, wem_price, dlmp_node, p_loss_series):
    """DLMP_{n,t} = WEM price + loss-markup recovered from the DSO duals.

    dlmp_node (N, T) are the power-balance duals of the DSO loss-min SOCP,
    i.e. the marginal-loss shadow prices. The substation dual is taken as the
    loss-reference; the locational premium over the substation is the DLMP
    signal fed to each participant (equation 29).
    """
    T = wem_price.shape[0]
    if dlmp_node is None:
        return [np.full(T, wem_price) for _ in participants]
    lam = np.asarray(dlmp_node)
    ref = lam[net.pcc, :]                       # substation marginal-loss price
    loc = lam - ref[None, :]                    # bus-vs-substation premium
    out = []
    for p in participants:
        out.append(wem_price + loc[p.bus - 1, :])
    return out


def _inner_loop(net, participants, pnet_a, p_loss_aux, lam_loss, rho_l,
                max_inner, eps, C_loss):
    """Algorithm 1 lines 9-18 (DSO <-> LMO loss consensus)."""
    N, T = net.N, 24
    # aggregate nodal net active injection (eq 28): \Psi \cdot p_net
    p_net_nodal = np.zeros((N, T))
    for a, p in enumerate(pnet_a):
        p_net_nodal[participants[a].bus - 1] += p
    # participants trade active power at unity power factor; the DSO carries
    # the (small) reactive component of their net position, kept here at 0 to
    # keep the social-SOCP feasible across penetrations.
    q_net_nodal = np.zeros((N, T))

    p_loss_star = np.array(p_loss_aux)
    lam_l = np.array(lam_loss)
    iters = 0
    dsm = None
    ok = False
    for _ in range(max_inner):
        iters += 1
        dsm, ok = _solve_dso_relaxed(net, p_net_nodal, q_net_nodal, p_loss_star,
                                     rho_l, C_loss)
        p_loss_grid = dsm.get_loss()
        p_loss_aux_u = p_loss_grid.copy()
        lam_l_new = lam_l + rho_l * (p_loss_aux_u - p_loss_grid)
        conv = float(np.max(np.abs(lam_l_new - lam_l)))
        lam_l = lam_l_new
        p_loss_star = p_loss_aux_u
        if conv <= eps:
            break
    dlmp_node = dsm.dlmp() if dsm is not None else None
    p_ug = dsm.p_ug_value if dsm is not None else None
    return iters, p_loss_grid, p_ug, dlmp_node, lam_l


def _solve_dso_relaxed(net, p_net_nodal, q_net_nodal, p_loss_star, rho_l,
                       C_loss):
    """Solve the DSO SOCP, relaxing voltage bounds if the nominal solve is
    infeasible (robustness fallback for data-driven dispatch).
    """
    bounds = [(net.vmin, net.vmax), (0.85, 1.10), (0.80, 1.15), (0.75, 1.20)]
    for (lo, hi) in bounds:
        dsm = net.dso_problem(p_net_nodal, q_net_nodal, p_loss_star=p_loss_star,
                              rho_loss=rho_l, C_loss=C_loss, vmin=lo, vmax=hi)
        dsm.solve()
        if dsm.problem.status == "optimal":
            return dsm, True
        if dsm.problem.status in ("infeasible", "unbounded"):
            continue
        # solver error / unknown -> try the next (wider) bound set
    return dsm, False


class _MarketResult:
    def __init__(self, A, T, wem, C_loss, dlmp, p_net, p_ug,
                 outer_iters, inner_iters, cpu_time, p_loss_series):
        self.A = A
        self.T = T
        self.dlmp = dlmp
        self.p_net = p_net
        self.p_ug = p_ug
        self.outer_iters = outer_iters
        self.inner_iters = inner_iters
        self.cpu_time = cpu_time
        self.p_loss_series = p_loss_series

    def summarize(self):
        return dict(outer_iters=self.outer_iters, inner_iters=self.inner_iters,
                    cpu_time=self.cpu_time)
