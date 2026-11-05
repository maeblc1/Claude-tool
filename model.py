"""
Energy-saving investment model for a small Swiss business.

CONVENTIONS (state these in the report)
- Year 1 is the first year of operation and uses today's known tariff p0
  (e.g. the published 2027 ElCom tariff). Year t uses p0 * (1+g)^(t-1).
- g is the AVERAGE annual tariff growth over the horizon (a scenario
  parameter, not a forecast). r is the nominal discount rate, r = r_f + premium.
- Savings in year t:  k_t = k * (1 - delta)^(t-1)   (delta = annual decline)
- Net cash flow in year t:  CF_t = k_t * share * p_t - M   (M = extra yearly cost;
  share < 1 if part of the tariff, e.g. a demand charge, is not avoided by saving kWh)
- NPV(g, r) = -I0 + sum_{t=1..T} CF_t / (1+r)^t
- Money in CHF, energy in kWh, tariffs in CHF/kWh (ElCom gives Rp./kWh: divide by 100).

Mathematics used (course link):
- derivatives, gradient, Hessian, Taylor approximation (Lecture 5)
- quadratic forms, positive semidefiniteness of the covariance matrix (Lecture 3)
- delta method: Var(NPV) ~ grad' Sigma grad (Lectures 3-5)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np


# --------------------------------------------------------------------------
# Inputs and validation
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Project:
    I0: float            # net upfront investment (cost minus subsidy), CHF
    k: float             # kWh saved in year 1
    p0: float            # avoided tariff in year 1, CHF/kWh
    T: int               # lifetime in years
    M: float = 0.0       # additional yearly cost (maintenance etc.), CHF/year
    delta: float = 0.0   # yearly decline of savings, 0 <= delta < 1
    share: float = 1.0   # share of the published tariff actually avoided per saved kWh, 0 < share <= 1


def validate(project: Project, g: float, r: float) -> None:
    """Raise ValueError with a readable message for invalid inputs."""
    if not float(project.T).is_integer() or project.T < 1:
        raise ValueError("Lifetime T must be a whole number of years >= 1.")
    if project.I0 < 0:
        raise ValueError("Investment I0 cannot be negative.")
    if project.k <= 0:
        raise ValueError("Energy saved k must be positive.")
    if project.p0 <= 0:
        raise ValueError("Tariff p0 must be positive.")
    if not (0 <= project.delta < 1):
        raise ValueError("delta must satisfy 0 <= delta < 1.")
    if not (0 < project.share <= 1):
        raise ValueError("share (tariff share avoided) must satisfy 0 < share <= 1.")
    if g <= -1:
        raise ValueError("Tariff growth g must be greater than -100%.")
    if r <= -1:
        raise ValueError("Discount rate r must be greater than -100%.")


def check_covariance(Sigma, tol: float = 1e-12) -> np.ndarray:
    """Return Sigma as an array if it is a valid covariance matrix, else raise.
    Valid = symmetric, non-negative variances, positive semidefinite (all
    eigenvalues >= 0). A user-typed correlation above 1 in absolute value fails."""
    S = np.asarray(Sigma, dtype=float)
    if S.shape != (2, 2):
        raise ValueError("Sigma must be 2x2.")
    if not np.allclose(S, S.T, atol=1e-12):
        raise ValueError("Sigma must be symmetric.")
    if (np.diag(S) < 0).any():
        raise ValueError("Variances cannot be negative.")
    eig = np.linalg.eigvalsh(S)
    if eig.min() < -tol:
        raise ValueError(
            f"Sigma is not positive semidefinite (smallest eigenvalue {eig.min():.3g}): "
            "some combination of g and r would have negative variance. "
            "Check that |correlation| <= 1.")
    return S


def make_sigma(sd_g: float, sd_r: float, rho: float) -> np.ndarray:
    """Build Sigma from standard deviations and a correlation, with validation."""
    if sd_g < 0 or sd_r < 0:
        raise ValueError("Standard deviations cannot be negative.")
    if not (-1 <= rho <= 1):
        raise ValueError("Correlation must lie between -1 and 1.")
    return check_covariance([[sd_g**2, rho * sd_g * sd_r],
                             [rho * sd_g * sd_r, sd_r**2]])


def sigma_g_horizon(sd_annual: float, T: int) -> float:
    """sd of the horizon-average growth rate IF annual changes were independent:
    sd_annual / sqrt(T). This is an ASSUMPTION; with ~7 annual observations it
    is a rough guide only."""
    return sd_annual / math.sqrt(T)


# --------------------------------------------------------------------------
# Core model
# --------------------------------------------------------------------------
def _t(T: int) -> np.ndarray:
    return np.arange(1, int(T) + 1, dtype=float)


def cash_flows(project: Project, g: float) -> np.ndarray:
    t = _t(project.T)
    return project.k * (1 - project.delta) ** (t - 1) * project.p0 * project.share * (1 + g) ** (t - 1) - project.M


def npv(project: Project, g: float, r: float) -> float:
    validate(project, g, r)
    t = _t(project.T)
    return float(-project.I0 + np.sum(cash_flows(project, g) / (1 + r) ** t))


def gradient(project: Project, g: float, r: float) -> np.ndarray:
    """Analytic gradient of NPV with respect to (g, r)."""
    validate(project, g, r)
    t = _t(project.T)
    a = project.k * (1 - project.delta) ** (t - 1) * project.p0 * project.share
    cf = cash_flows(project, g)
    d_g = np.sum(a * (t - 1) * (1 + g) ** (t - 2) / (1 + r) ** t)
    d_r = -np.sum(t * cf / (1 + r) ** (t + 1))
    return np.array([d_g, d_r])


def hessian(project: Project, g: float, r: float) -> np.ndarray:
    """Analytic 2x2 Hessian of NPV with respect to (g, r)."""
    validate(project, g, r)
    t = _t(project.T)
    a = project.k * (1 - project.delta) ** (t - 1) * project.p0 * project.share
    cf = cash_flows(project, g)
    h_gg = np.sum(a * (t - 1) * (t - 2) * (1 + g) ** (t - 3) / (1 + r) ** t)
    h_gr = -np.sum(a * (t - 1) * t * (1 + g) ** (t - 2) / (1 + r) ** (t + 1))
    h_rr = np.sum(t * (t + 1) * cf / (1 + r) ** (t + 2))
    return np.array([[h_gg, h_gr], [h_gr, h_rr]])


def dnpv_dp0(project: Project, g: float, r: float) -> float:
    """NPV is LINEAR in the tariff level p0, so this derivative is constant
    and a Taylor expansion in p0 is exact. Equals k * B(g, r)."""
    return project.k * project.share * _B(project, g, r)


# --------------------------------------------------------------------------
# Break-even quantities (closed form because NPV is linear in p0, k, I0, M)
# --------------------------------------------------------------------------
def _A(project: Project, r: float) -> float:
    t = _t(project.T)
    return float(np.sum(1 / (1 + r) ** t))


def _B(project: Project, g: float, r: float) -> float:
    t = _t(project.T)
    return float(np.sum((1 - project.delta) ** (t - 1) * (1 + g) ** (t - 1) / (1 + r) ** t))


def break_even_tariff(project: Project, g: float, r: float) -> float:
    """Year-1 tariff p0* (CHF/kWh) at which NPV = 0."""
    validate(project, g, r)
    return (project.I0 + project.M * _A(project, r)) / (project.k * project.share * _B(project, g, r))


def break_even_savings(project: Project, g: float, r: float) -> float:
    """Yearly kWh saved at which NPV = 0."""
    validate(project, g, r)
    return (project.I0 + project.M * _A(project, r)) / (project.p0 * project.share * _B(project, g, r))


def max_affordable_investment(project: Project, g: float, r: float) -> float:
    """Highest net investment I0 with NPV >= 0."""
    validate(project, g, r)
    return project.p0 * project.share * project.k * _B(project, g, r) - project.M * _A(project, r)


def simple_payback(project: Project) -> float:
    """Undiscounted payback in years using year-1 cash flow (baseline method)."""
    cf1 = project.k * project.p0 * project.share - project.M
    return math.inf if cf1 <= 0 else project.I0 / cf1


def discounted_payback(project: Project, g: float, r: float):
    """First year in which cumulative discounted cash flow covers I0, else None."""
    validate(project, g, r)
    t = _t(project.T)
    cum = np.cumsum(cash_flows(project, g) / (1 + r) ** t)
    idx = np.where(cum >= project.I0)[0]
    return int(idx[0] + 1) if idx.size else None


# --------------------------------------------------------------------------
# Taylor approximation (compare with exact recomputation)
# --------------------------------------------------------------------------
def taylor_npv(project: Project, g0: float, r0: float, dg: float, dr: float, order: int = 2) -> float:
    base = npv(project, g0, r0)
    h = np.array([dg, dr])
    est = base + gradient(project, g0, r0) @ h
    if order >= 2:
        est += 0.5 * h @ hessian(project, g0, r0) @ h
    return float(est)


def taylor_report(project: Project, g0: float, r0: float, dg: float, dr: float) -> dict:
    exact = npv(project, g0 + dg, r0 + dr)
    first = taylor_npv(project, g0, r0, dg, dr, order=1)
    second = taylor_npv(project, g0, r0, dg, dr, order=2)
    return {"exact": exact, "first_order": first, "second_order": second,
            "err_first": first - exact, "err_second": second - exact}


# --------------------------------------------------------------------------
# Uncertainty: delta method and Monte Carlo
# --------------------------------------------------------------------------
def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def delta_method(project: Project, mu_g: float, mu_r: float, Sigma) -> dict:
    """Approximate mean, variance and P(NPV>0) of NPV when (g, r) ~ (mu, Sigma).
    mean ~ NPV(mu) + 0.5 tr(H Sigma);  var ~ grad' Sigma grad."""
    S = check_covariance(Sigma)
    base = npv(project, mu_g, mu_r)
    grad = gradient(project, mu_g, mu_r)
    H = hessian(project, mu_g, mu_r)
    mean = base + 0.5 * float(np.trace(H @ S))
    var = float(grad @ S @ grad)
    sd = math.sqrt(max(var, 0.0))
    p_pos = _norm_cdf(mean / sd) if sd > 0 else float(mean > 0)
    return {"npv_at_mean": base, "mean": mean, "var": var, "sd": sd, "p_positive": p_pos}


def monte_carlo(project: Project, mu_g: float, mu_r: float, Sigma,
                n: int = 20000, seed: int = 0) -> dict:
    """Exact recomputation of NPV for random draws (g, r) ~ N(mu, Sigma).
    Draws with g <= -99% or r <= -99% are clipped (reported as clipped_share)."""
    S = check_covariance(Sigma)
    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal([mu_g, mu_r], S, size=n)
    bad = (draws[:, 0] <= -0.99) | (draws[:, 1] <= -0.99)
    draws[:, 0] = np.maximum(draws[:, 0], -0.99)
    draws[:, 1] = np.maximum(draws[:, 1], -0.99)
    vals = np.array([npv(project, gi, ri) for gi, ri in draws])
    return {"mean": float(vals.mean()), "sd": float(vals.std(ddof=1)),
            "p_positive": float((vals > 0).mean()),
            "p05": float(np.quantile(vals, 0.05)), "p95": float(np.quantile(vals, 0.95)),
            "clipped_share": float(bad.mean()), "samples": vals}


# --------------------------------------------------------------------------
# Sensitivity (tornado) helper
# --------------------------------------------------------------------------
def tornado(project: Project, g: float, r: float, ranges: dict) -> list[tuple[str, float, float]]:
    """ranges: {'I0': (lo, hi), 'k': (lo, hi), 'p0': ..., 'T': ..., 'M': ..., 'g': ..., 'r': ...}
    Returns [(name, npv_at_lo, npv_at_hi)] sorted by swing, largest first."""
    out = []
    for name, (lo, hi) in ranges.items():
        vals = []
        for v in (lo, hi):
            if name == "g":
                vals.append(npv(project, v, r))
            elif name == "r":
                vals.append(npv(project, g, v))
            else:
                vals.append(npv(replace(project, **{name: v}), g, r))
        out.append((name, vals[0], vals[1]))
    return sorted(out, key=lambda x: abs(x[2] - x[1]), reverse=True)


# --------------------------------------------------------------------------
# Timing: replace a working appliance NOW, or WAIT n years until it fails?
# (RAVEL guide 1992 calls this the "Ersatzfrage": the best moment for a replacement.)
# --------------------------------------------------------------------------
def replace_now_vs_wait(project: Project, g: float, r: float, n: int) -> dict:
    """Compare two plans over a common horizon H = n + T years, where project.I0 is the FULL net
    purchase price of the new appliance and T its life:
      NOW : buy at year 0 (old appliance scrapped, no salvage), buy again at year T, count the
            unused share (T-n)/T of the second purchase as residual value at H.
      WAIT: keep the old appliance n more years (no savings), buy at year n, no residual at H.
    Savings in calendar year t use the tariff p0*(1+g)^(t-1) and the appliance age (delta applies).
    Returns PVs of each plan and now_minus_wait (> 0 means replacing now is better).
    ASSUMPTIONS: same price I0 for any future purchase; old appliance has no salvage value."""
    validate(project, g, r)
    T = int(project.T)
    if not (0 <= n <= T):
        raise ValueError("Remaining life n must satisfy 0 <= n <= T.")
    H = n + T

    def saving(t: int, start: int) -> float:
        age = t - start
        if age < 1 or age > T:
            return 0.0
        return (project.k * (1 - project.delta) ** (age - 1) * project.p0 * project.share
                * (1 + g) ** (t - 1) - project.M)

    d = lambda t: (1 + r) ** (-t)
    pv_now = -project.I0 - project.I0 * d(T) + project.I0 * (T - n) / T * d(H)
    pv_now += sum((saving(t, 0) if t <= T else saving(t, T)) * d(t) for t in range(1, H + 1))
    pv_wait = -project.I0 * d(n) + sum(saving(t, n) * d(t) for t in range(n + 1, H + 1))
    return {"n": n, "pv_now": pv_now, "pv_wait": pv_wait, "now_minus_wait": pv_now - pv_wait}


def timing_table(project: Project, g: float, r: float, ns) -> list[dict]:
    return [replace_now_vs_wait(project, g, r, n) for n in ns]


# --------------------------------------------------------------------------
# Measure presets (label every number: documented range vs assumption)
# --------------------------------------------------------------------------
PRESETS = {
    # Source for both presets: Swiss Federal Office of Energy (2022), "Potenzial und Massnahmen zur
    # Steigerung der Stromeffizienz bis 2025", measure 31a (commercial hobs), p. 59/82, which uses
    # EcoGastro monitoring: avg 120,000 kWh before -> 102,000 kWh after = 18,000 kWh/yr saved (from bills),
    # induction purchase price CHF 25,000 (ASSUMPTION of the report), premium ~40% (ASSUMPTION),
    # 1,000 new hobs/yr across 30,000 kitchens (=> 30-year life assumed), 7.1 M CHF/yr extra cost.
    # NOTE: these kitchens use ~120 MWh/yr, i.e. ElCom profile C3 (150 MWh), NOT C2 (30 MWh).
    "induction_incremental_sfoe": {
        "label": "Induction instead of a conventional hob WHEN a replacement is due anyway (extra cost only); ElCom profile C3",
        "I0_gross": {"value": 7143, "range": (3500, 10000), "status": "DERIVED: 25,000 x 0.40/1.40 = 7,143 (SFOE: 7.1 M CHF / 1,000 hobs); range = assumed +-50%"},
        "subsidy":  {"value": 0, "range": (0, 6000), "status": "BFE brochure 2017: up to CHF 6,000; current availability NOT verified -> base 0"},
        "k":        {"value": 18000, "range": (9000, 27000), "status": "SFOE 2022 p.59: observed average from EcoGastro bills; range = assumed +-50% (kitchen size varies)"},
        "T":        {"value": 20, "range": (15, 30), "status": "RAVEL guide (1992): 15-25 y for electrical/mechanical plant; SFOE 2022 assumes 30 y"},
        "M":        {"value": 0, "range": (0, 300), "status": "ASSUMPTION: no extra maintenance (find supplier source)"},
        "delta":    {"value": 0.0, "range": (0, 0.02), "status": "ASSUMPTION: no decline in savings"},
        "share":    {"value": 1.0, "range": (0.5, 1.0), "status": "ASSUMPTION: ElCom C3 average includes demand charges that kWh savings may not avoid; 1.0 = full average tariff"},
    },
    "induction_replace_now_sfoe": {
        "label": "Replace a still-working conventional hob NOW (full purchase price); ElCom profile C3",
        "I0_gross": {"value": 25000, "range": (15000, 40000), "status": "SFOE 2022 p.59: avg purchase price 25,000 (ASSUMPTION of report); supplier 4-zone range listed at CHF 15,148 ex VAT (UNVERIFIED by us). Ignores residual value of old hob"},
        "subsidy":  {"value": 0, "range": (0, 6000), "status": "see above; base 0"},
        "k":        {"value": 18000, "range": (9000, 27000), "status": "SFOE 2022 p.59"},
        "T":        {"value": 20, "range": (15, 30), "status": "RAVEL 15-25 y; SFOE 30 y"},
        "M":        {"value": 0, "range": (0, 300), "status": "ASSUMPTION"},
        "delta":    {"value": 0.0, "range": (0, 0.02), "status": "ASSUMPTION"},
        "share":    {"value": 1.0, "range": (0.5, 1.0), "status": "ASSUMPTION (see above)"},
    },
}


def project_from_preset(name: str, p0: float, overrides: dict | None = None) -> Project:
    pr = PRESETS[name]
    v = {key: pr[key]["value"] for key in ("I0_gross", "subsidy", "k", "T", "M", "delta", "share")}
    if overrides:
        v.update(overrides)
    return Project(I0=v["I0_gross"] - v["subsidy"], k=v["k"], p0=p0, T=int(v["T"]),
                   M=v["M"], delta=v["delta"], share=v["share"])
