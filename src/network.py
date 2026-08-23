"""Per-unit branch-flow (DistFlow) SOCP model of the radial feeder.

Implements the DSO's second-order-cone operating model, equations (2)-(7) of
the paper: active/reactive power balance (whose duals define the DLMP), branch
voltage-drop, the SOC relaxation, line capacity limits, voltage bounds, and the
total loss expression (7).

Per-unit on base MVA = 1.0 and base voltage 12.66 kV. Net nodal injections are
fixed inputs (sent by the LMO / DSO aggregation); the power-balance equality
duals v[balance] are the nodal DLMPs.
"""

import cvxpy as cp
import numpy as np

from . import ieee69


class Network:
    def __init__(self, base_mva=1.0, vmin=0.90, vmax=1.05, kv=12.66):
        self.base_mva = base_mva
        self.vmin = vmin
        self.vmax = vmax
        self.N = len(ieee69.LOAD)
        self.edges = [(f - 1, t - 1) for (f, t, _, _) in ieee69.BRANCHES]
        z_base = kv ** 2 / base_mva              # ohm at the system base
        self.R = np.array([r for (_, _, r, _) in ieee69.BRANCHES]) / z_base
        self.X = np.array([x for (_, _, _, x) in ieee69.BRANCHES]) / z_base
        children = {i: [] for i in range(self.N)}
        for (f, t) in self.edges:
            children[f].append(t)
        self.children = children
        self.pcc = ieee69.PCC - 1
        self.Smax = 30.0  # generous line capacity (no congestion in our setup)

    @property
    def p0(self):
        return np.array([ieee69.LOAD[b][0] for b in range(1, self.N + 1)]) / (
            1000.0 * self.base_mva)

    @property
    def q0(self):
        return np.array([ieee69.LOAD[b][1] for b in range(1, self.N + 1)]) / (
            1000.0 * self.base_mva)

    def _loss(self, l):
        return cp.sum(cp.multiply(self.R[:, None], l), axis=0)  # per-hour loss

    def dso_problem(self, p_net, q_net, p_loss_star=None, rho_loss=1.0,
                    C_loss=1.0, vmin=None, vmax=None):
        """DSO SOCP subproblem (Subproblem II).

        p_net, q_net : (N, T) per-unit net active/reactive injections (fixed).
        Returns a DSOBundle handle with solve() and value getters, including
        the nodal DLMPs recovered from the power-balance duals.
        """
        if vmin is None:
            vmin = self.vmin
        if vmax is None:
            vmax = self.vmax
        T = p_net.shape[1]
        pf = cp.Variable((len(self.edges), T))
        qf = cp.Variable((len(self.edges), T))
        l = cp.Variable((len(self.edges), T))
        v = cp.Variable((self.N, T))
        p_ug = cp.Variable(T)   # upstream-grid import at the PCC (eq 2)
        q_ug = cp.Variable(T)

        balance_p = []
        balance_q = []
        for i in range(self.N):
            inflow = sum(pf[e] for e, (f, t) in enumerate(self.edges) if t == i)
            outflow = sum(pf[e] for e, (f, t) in enumerate(self.edges) if f == i)
            if i == self.pcc:
                balance_p.append(inflow - outflow + p_net[i] - p_ug == 0)
            else:
                balance_p.append(inflow - outflow + p_net[i] == 0)
            inflow_q = sum(qf[e] for e, (f, t) in enumerate(self.edges) if t == i)
            outflow_q = sum(qf[e] for e, (f, t) in enumerate(self.edges) if f == i)
            if i == self.pcc:
                balance_q.append(inflow_q - outflow_q + q_net[i] - q_ug == 0)
            else:
                balance_q.append(inflow_q - outflow_q + q_net[i] == 0)

        drop = []
        for e, (f, t) in enumerate(self.edges):
            drop.append(v[f] - v[t] - 2*(self.R[e]*pf[e] + self.X[e]*qf[e])
                        + (self.R[e]**2 + self.X[e]**2)*l[e] == 0)
        soc = []
        for e, (f, _) in enumerate(self.edges):
            for t in range(T):
                soc.append(cp.SOC(v[f, t] + l[e, t],
                                  cp.hstack([2*pf[e, t], 2*qf[e, t],
                                             v[f, t] - l[e, t]])))
        cap = []
        for e in range(len(self.edges)):
            for t in range(T):
                cap.append(cp.norm(cp.hstack([pf[e, t], qf[e, t]]), 2)
                           <= self.Smax)
        vbounds = []
        for t in range(T):
            vbounds += [vmin ** 2 <= v[:, t], v[:, t] <= vmax ** 2]

        loss = self._loss(l)
        loss_cost = cp.sum(loss * C_loss)
        prox = (rho_loss / 2 * cp.sum((loss - p_loss_star)**2)
                if p_loss_star is not None else 0.0)

        objective = cp.Minimize(loss_cost + prox)
        cons = balance_p + balance_q + drop + soc + cap + vbounds + [
            v[self.pcc, :] == 1.0]
        problem = cp.Problem(objective, cons)
        return DSM(loss, v, p_ug, q_ug).bind(problem, balance_p, p_loss_star,
                                             p_net, q_net)


class DSM:
    """Handle exposing a solved DSO SOCP and its outputs."""
    def __init__(self, loss, v, p_ug, q_ug):
        self.loss = loss
        self.v = v
        self.p_ug = p_ug
        self.q_ug = q_ug
        self.problem = None
        self.balance_p = None

    def bind(self, problem, balance_p, p_loss_star, p_net, q_net):
        self.problem = problem
        self.balance_p = balance_p
        self.p_loss_star = p_loss_star
        self.p_net = p_net
        self.q_net = q_net
        return self

    def solve(self, **kw):
        return self.problem.solve(**kw)

    @property
    def p_ug_value(self):
        return np.asarray(self.p_ug.value).ravel()

    def get_loss(self):
        return np.asarray(self.loss.value).ravel()

    def get_loss_series(self):
        return np.asarray(self.loss.value).ravel()

    def get_voltages(self):
        return np.sqrt(np.asarray(self.v.value))

    def dlmp(self):
        """Nodal DLMPs from the power-balance duals (eq. 2 duals)."""
        lam = np.array([np.asarray(c.dual_value) for c in self.balance_p])
        return lam
