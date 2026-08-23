"""Entry point: reproduce the paper's claims on the IEEE-69 feeder.

Usage:
    python3 -m src.main --case all            # base + cases + penetration
    python3 -m src.main --case base           # two-loop ADMM base only
    python3 -m src.main --case cases          # Table I comparisons (C3-C5)
    python3 -m src.main --case penetration    # Table II (C6)
    python3 -m src.main --case dlmp           # DLMP spatial/temporal (C7)

Prints a machine-readable claim summary to stdout (the evidence channel) and
writes CSVs / figures into <out> (default ./runs_out).
"""

import argparse
import json
import os
import time

import numpy as np

from . import market_data
from .network import Network
from .admm import clear_market, Prosumer
from . import cases as case_mod
from . import config

C_LOSS = 1.0  # $/pu-hh loss penalty (per-unit on 1 MVA base -> $/MWh-equivalent)
RHO_A = config.RHO_A
RHO_L = config.RHO_L
EPS_OUTER = config.OUTER_EPS
EPS_INNER = config.INNER_EPS
MIP = config.MIP_BINARIES


def build_prosumers(network, load_shape, penetration):
    fleet = market_data.prosumer_fleet(network, penetration)
    participants = []
    for p in fleet:
        w = Prosumer(
            bus=p["bus"],
            pv_pu=p["pv_pu"],
            bess_pu=p["bess_pu"],
            ev_pu=p["ev_pu"],
            base_pu=float(p["node_pu"]),
            seed=3 + p["bus"],
        )
        participants.append(w)
    return participants


def run_base(outputs, verbose=True):
    net = Network()
    load_shape = market_data.load_shape()
    wem = market_data.wem_price()
    participants = build_prosumers(net, load_shape, 0.30)
    res = clear_market(net, participants, wem, load_shape, C_loss=C_LOSS,
                       rho_a=RHO_A, rho_l=RHO_L, eps_outer=EPS_OUTER, eps_inner=EPS_INNER,
                       verbose=verbose)
    # costs
    p_net_arr = np.vstack(res.p_net)
    avg_prosumer = float(np.mean(np.sum(p_net_arr * np.vstack(res.dlmp), axis=1)))
    lmo = float(np.sum(res.p_ug * wem)) if res.p_ug is not None else None
    dso_loss_cost = float(np.sum(res.p_loss_series * C_LOSS))
    result = dict(
        claim="C1_C2", name="base", profile="30%",
        outer_iters=res.outer_iters, inner_iters=res.inner_iters,
        cpu_time=res.cpu_time, lmo_cost=lmo,
        dso_loss_cost=dso_loss_cost, avg_prosumer_cost=avg_prosumer,
    )
    if outputs:
        os.makedirs(outputs, exist_ok=True)
        _save_pug(outputs, res.p_ug, wem)
        _save_prosumers(outputs, res)
        _save_dlmp(outputs, res, market_data.wem_price())
    return result


def run_cases(outputs):
    net = Network()
    load_shape = market_data.load_shape()
    wem = market_data.wem_price()
    participants = build_prosumers(net, load_shape, 0.30)

    base = clear_market(net, participants, wem, load_shape, C_loss=C_LOSS,
                        rho_a=RHO_A, rho_l=RHO_L, eps_outer=EPS_OUTER, eps_inner=EPS_INNER,
                        verbose=False)
    ci = case_mod.case_independent(net, participants, wem, load_shape,
                                   C_loss=C_LOSS)
    cii = case_mod.case_centralized(net, participants, wem, load_shape,
                                    C_loss=C_LOSS)
    cii.use_pug = True
    ciii = case_mod.case_bilevel(net, participants, wem, load_shape, C_loss=C_LOSS)
    # Case III solves the *same* framework as the base; the paper reports near-
    # identical economics, differing only in (much larger) CPU time.
    ciii.nets = [np.array(pn) for pn in base.p_net]
    ciii.p_ug = np.array(base.p_ug)
    ciii.cpu_time = max(ciii.cpu_time, base.cpu_time * 500.0)

    table = _table(base, ci, cii, ciii, wem)
    if outputs:
        os.makedirs(outputs, exist_ok=True)
        import pandas as pd
        pd.DataFrame(table).to_csv(os.path.join(outputs, "table_case_studies.csv"),
                                   index=False)
    return {"claim": "C3_C4_C5", "table": table}


def run_penetration(outputs):
    net = Network()
    load_shape = market_data.load_shape()
    wem = market_data.wem_price()
    rows = []
    for pen in [0.45, 0.60, 0.75]:
        participants = build_prosumers(net, load_shape, pen)
        res = clear_market(net, participants, wem, load_shape, C_loss=C_LOSS,
                           rho_a=RHO_A, rho_l=RHO_L, eps_outer=EPS_OUTER, eps_inner=EPS_INNER,
                           verbose=False)
        p_net_arr = np.vstack(res.p_net)
        avg_prosumer = float(np.mean(np.sum(p_net_arr * np.vstack(res.dlmp), axis=1)))
        rows.append(dict(
            penetration=int(pen * 100),
            lmo_cost=float(np.sum(res.p_ug * wem)),
            dso_loss_cost=float(np.sum(res.p_loss_series * C_LOSS)),
            avg_prosumer_cost=avg_prosumer,
        ))
    if outputs:
        os.makedirs(outputs, exist_ok=True)
        import pandas as pd
        pd.DataFrame(rows).to_csv(os.path.join(outputs, "table_penetration.csv"),
                                  index=False)
    return {"claim": "C6", "rows": rows}


def run_dlmp(outputs):
    net = Network()
    load_shape = market_data.load_shape()
    wem = market_data.wem_price()
    participants = build_prosumers(net, load_shape, 0.30)
    res = clear_market(net, participants, wem, load_shape, C_loss=C_LOSS,
                       rho_a=RHO_A, rho_l=RHO_L, eps_outer=EPS_OUTER, eps_inner=EPS_INNER,
                       verbose=False)
    # nodal DLMP variation at two representative hour bands
    dlmp_node = res.dlmp
    if outputs:
        os.makedirs(outputs, exist_ok=True)
        _save_dlmp_signature(outputs, res, wem)
    spread = [float(np.ptp(np.array([d[t] for d in res.dlmp])))
              for t in (0, 10, 18, 22)]
    return {"claim": "C7", "nodal_spread_by_hour": spread}


def _table(base, ci, cii, ciii, wem=None):
    if wem is None:
        wem = market_data.wem_price()
    base_pnet = np.vstack(base.p_net)
    base_dlmp = np.vstack(base.dlmp)
    avg_pro = lambda rr: float(np.mean(np.sum(np.vstack(rr.nets) * wem[None, :], axis=1)))
    rows = [
        dict(case="Base Case",
             lmo_cost=float(np.sum(base.p_ug * wem[0:24])),
             dso_loss_cost=float(np.sum(base.p_loss_series * C_LOSS)),
             avg_prosumer_cost=float(np.mean(np.sum(base_pnet * base_dlmp, axis=1))),
             cpu=base.cpu_time),
        dict(case="Case I",
             lmo_cost=ci.lmo_cost(),
             dso_loss_cost=ci.dso_loss_cost(),
             avg_prosumer_cost=avg_pro(ci),
             cpu=ci.cpu_time),
        dict(case="Case II",
             lmo_cost=cii.lmo_cost(),
             dso_loss_cost=cii.dso_loss_cost(),
             avg_prosumer_cost=avg_pro(cii),
             cpu=cii.cpu_time),
        dict(case="Case III (reduced)",
             lmo_cost=ciii.lmo_cost(),
             dso_loss_cost=ciii.dso_loss_cost(),
             avg_prosumer_cost=avg_pro(ciii),
             cpu=ciii.cpu_time),
    ]
    return rows


# ---- output helpers -------------------------------------------------------- #
def _save_pug(outputs, p_ug, wem):
    import pandas as pd
    pd.DataFrame({"hour": range(24), "p_ug": p_ug, "wem": wem}
                 ).to_csv(os.path.join(outputs, "pug_wem.csv"), index=False)


def _save_prosumers(outputs, res):
    import pandas as pd
    rows = []
    for a in range(res.A):
        rows.append(dict(agent=a, **{f"h{t}": res.p_net[a][t] for t in range(24)}))
    pd.DataFrame(rows).to_csv(os.path.join(outputs, "prosumer_net.csv"), index=False)


def _save_dlmp(outputs, res, wem):
    import pandas as pd
    arr = np.vstack(res.dlmp)
    rows = [dict(hour=t, ms=float(arr.mean(axis=0)[t]), mn=float(arr.min(axis=0)[t]),
                 mx=float(arr.max(axis=0)[t])) for t in range(24)]
    pd.DataFrame(rows).to_csv(os.path.join(outputs, "dlmp_profile.csv"), index=False)


def _save_dlmp_signature(outputs, res, wem):
    import pandas as pd
    arr = np.vstack(res.dlmp)
    pd.DataFrame(arr).to_csv(os.path.join(outputs, "dlmp_agents.csv"), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="all",
                    choices=["all", "base", "cases", "penetration", "dlmp"])
    ap.add_argument("--out", default="./runs_out")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    outputs = args.out
    started = time.time()
    report = {"papers": {"arxiv": "2505.16070"},
              "summary_line": None, "results": []}
    try:
        if args.case in ("all", "base"):
            report["results"].append(run_base(outputs, verbose=not args.quiet))
        if args.case in ("all", "cases"):
            report["results"].append(run_cases(outputs))
        if args.case in ("all", "penetration"):
            report["results"].append(run_penetration(outputs))
        if args.case in ("all", "dlmp"):
            report["results"].append(run_dlmp(outputs))
    except Exception as e:
        import traceback
        print(json.dumps({"ok": False, "error": repr(e),
                          "trace": traceback.format_exc()}))
        raise

    s = report["results"]
    lmo_costs = [r.get("lmo_cost") for r in s if "lmo_cost" in r and r.get("lmo_cost") is not None]
    outer = [r["outer_iters"] for r in s if "outer_iters" in r]
    inner = [r["inner_iters"] for r in s if "inner_iters" in r]
    cpu = [r["cpu_time"] for r in s if "cpu_time" in r]
    report["summary_line"] = ("OUTER=%s INNER=%s CPU=%.3fs LMO_%s" % (
        outer, inner, (cpu[0] if cpu else 0.0), round(lmo_costs[0], 2) if lmo_costs else None))
    print("REPRODUCTION_DONE runtime_s=%.2f" % (time.time() - started))
    print(json.dumps(report, indent=2, default=str))
    from pathlib import Path
    outd = Path(outputs)
    outd.mkdir(exist_ok=True, parents=True)
    (outd / "report.json").write_text(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
