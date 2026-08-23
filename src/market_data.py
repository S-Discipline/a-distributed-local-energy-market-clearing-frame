"""Synthetic market / prosumer input data.

The reproduced paper uses PJM wholesale prices, solar-irradiation data, NHTS
travel-derived EV schedules and per-customer resource parameter classes that are
not packaged with the paper. To keep every experiment self-contained and
reproducible, we synthesise these inputs with deterministic random seeds. The
shapes follow the paper:

  * 30% of the customers under each load node are active prosumers (penetration
    is a tunable fraction used for Table I; case study Table II varies it among
    {45, 60, 75}%).
  * Each active prosumer owns a BESS, an EV, a PV unit and flexible loads.
  * Flexible load is 5% of the prosumer's base load; 95% of daily energy must be
    met after flexibility usage.
  * Average power factor of load nodes is 0.85 lag.

All quantities returned to the market-clearing code are in per-unit on the
network base (see src/network.py). This substitution is reported explicitly in
the reproduction write-up and is the dominant source of any numeric offset from
the paper's tables.
"""

import numpy as np

from . import ieee69


def time_horizon():
    """24 hourly intervals (day-ahead market)."""
    return np.arange(24)


def wem_price(seed=0):
    """PJM-style wholesale price ($/MWh) with a mid-afternoon peak."""
    rng = np.random.default_rng(seed)
    H = 24
    base = 30.0
    diurnal = 18.0 * _smooth_peak(np.arange(H), peak_hour=18.0, width=4.0)
    noise = 2.0 * rng.normal(size=H)
    price = np.clip(base + diurnal + noise, 12.0, 95.0)
    return price  # $/MWh


def _smooth_peak(t, peak_hour, width):
    """Gaussian bump centred at peak_hour over a 24 h axis (radial wrap)."""
    d = np.abs((t - peak_hour + 12.0) % 24.0 - 12.0)
    return np.exp(-0.5 * (d / width) ** 2)


def load_shape(seed=1):
    """Normalised 24 h active-load profile (dimensionless multiplier)."""
    rng = np.random.default_rng(seed)
    t = np.arange(24)
    morning = _smooth_peak(t, 8.0, 2.5)
    evening = _smooth_peak(t, 19.0, 3.0)
    shape = 0.55 + 0.30 * morning + 0.60 * evening
    shape = shape + 0.03 * rng.normal(size=24)
    shape = shape / shape.mean()
    return shape


def pv_profile(seed=2):
    """Normalised 24 h PV output (0 during night, peak at solar noon)."""
    rng = np.random.default_rng(seed)
    t = np.arange(24)
    day = _smooth_peak(t, 13.0, 3.5) * (np.abs(t - 12.5) < 6.5)
    cloudy = 1.0 - 0.12 * rng.random(24)
    return day * cloudy


def bus_load_kw():
    """Base (peak) active load of every bus in kW from the feeder data."""
    return np.array([ieee69.LOAD[b][0] for b in range(1, ieee69.N + 1)])


def prosumer_fleet(net, penetration, resource_rng_seed=7):
    """Return the list of aggregated prosumers (one per load node).

    The feeder's customer base is FIXED across penetration levels: every node
    is served by an aggregator whose base load equals the node's load. The
    *active-prosumer penetration* governs how much DER (PV / BESS / EV)
    capacity covers that load: higher penetration means the same customers
    install more generation and storage, reducing purchases from the grid.
    """
    pen = penetration
    f = max(0.2, pen / 0.30)                 # DER-capacity multiplier
    prosumers = []
    base_demand = net.p0                       # per-unit base active per node
    for bus in range(1, net.N + 1):
        if bus == net.pcc + 1 or base_demand[bus - 1] <= 1e-9:
            continue
        node_pu = float(base_demand[bus - 1])   # fixed customer base
        pv_pu = 0.45 * node_pu * f              # DER grows with penetration
        bess_pu = 0.75 * pv_pu
        ev_pu = 0.35 * node_pu * f
        prosumers.append(dict(
            bus=bus,
            pen=pen,
            node_pu=node_pu,
            pv_pu=pv_pu,
            bess_pu=bess_pu,
            ev_pu=ev_pu,
            fl_pu=0.05 * node_pu,
        ))
    return prosumers
