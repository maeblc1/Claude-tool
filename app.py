"""
Energy-saving investment tool for a small Swiss business (Streamlit app).

Run locally:   streamlit run app.py
Files needed in the same folder: model.py, tariffs_C3_clean.csv, yields_10y_annual.csv

All calculations come from model.py (tested by test_model.py). Nothing here is hard-coded:
every number shown is recomputed from the sidebar inputs and the source data.
"""
from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import model as m

HERE = Path(__file__).parent
st.set_page_config(page_title="Induction investment tool", page_icon="⚡", layout="wide")


# --------------------------------------------------------------------------
# Data (cached)
# --------------------------------------------------------------------------
@st.cache_data
def load_data():
    tar = pd.read_csv(HERE / "tariffs_C3_clean.csv")
    yld = pd.read_csv(HERE / "yields_10y_annual.csv", index_col=0)
    muni = tar.groupby(["gemeindeNummer", "year"])["variable_tariff"].median().unstack()
    med = muni.median()
    gch = np.log(med).diff().dropna()
    dy = yld["mean"].diff().dropna()
    common = gch.index.intersection(dy.index)
    stats = {
        "mean_g": float(gch.mean()), "sd_g": float(gch.std(ddof=1)), "n_g": int(len(gch)),
        "sd_r": float(dy.std(ddof=1)) / 100, "n_r": int(len(dy)),
        "rho": float(np.corrcoef(gch[common], dy[common])[0, 1]), "n_rho": int(len(common)),
        "rf": float(yld.loc[yld.index.max(), "last"]) / 100, "rf_year": int(yld.index.max()),
        "med": {int(k): float(v) for k, v in med.items()},
    }
    return tar, yld, muni, stats


tar, yld, muni, stats = load_data()
YEARS = sorted(int(y) for y in tar["year"].unique())
LATEST = YEARS[-1]


# --------------------------------------------------------------------------
# Sidebar: inputs
# --------------------------------------------------------------------------
st.sidebar.title("Inputs")

st.sidebar.header("1. Decision")
framing = st.sidebar.radio(
    "Which decision?",
    ["Replacement is due anyway (extra cost of induction only)",
     "Replace a working hob now (full purchase price)"],
)
INCREMENTAL = framing.startswith("Replacement")
key = "induction_incremental_sfoe" if INCREMENTAL else "induction_replace_now_sfoe"
pr = m.PRESETS[key]

st.sidebar.header("2. Electricity tariff")
year = st.sidebar.selectbox("Tariff level used as year-1 tariff", YEARS[::-1], index=0,
                            help="Pick an earlier year to see how the same investment looked at that year's tariff (e.g. 2024 = the peak).")
where = st.sidebar.radio("Location", ["National median (profile C3)", "Choose a municipality"])
canton = gem = op = None
if where == "Choose a municipality":
    t_latest = tar[tar["year"] == LATEST]
    canton = st.sidebar.selectbox("Canton", sorted(t_latest["kanton"].unique()), index=sorted(t_latest["kanton"].unique()).index("VD") if "VD" in set(t_latest["kanton"]) else 0)
    gems = sorted(t_latest[t_latest["kanton"] == canton]["gemeindeName"].unique())
    gem = st.sidebar.selectbox("Municipality", gems)
    ops = sorted(t_latest[(t_latest["kanton"] == canton) & (t_latest["gemeindeName"] == gem)]["netzbetreiber"].unique())
    op = st.sidebar.selectbox("Network operator", ops) if len(ops) > 1 else ops[0]


def tariff_chf(y, canton, gem, op):
    """Variable tariff in CHF/kWh for the chosen year/location, or None if not available."""
    if gem is None:
        return float(muni[y].median()) / 100
    t = tar[(tar["year"] == y) & (tar["kanton"] == canton) & (tar["gemeindeName"] == gem) & (tar["netzbetreiber"] == op)]
    return float(t["variable_tariff"].median()) / 100 if len(t) else None


p0 = tariff_chf(year, canton, gem, op)
if p0 is None:
    st.sidebar.warning(f"No {year} tariff for this municipality/operator in the data; using the national median.")
    p0 = tariff_chf(year, None, None, None)
st.sidebar.caption(f"Variable tariff used: **{p0*100:.2f} Rp./kWh** (energy + grid + levies + surcharge)")

st.sidebar.header("3. The investment")
k = st.sidebar.number_input("kWh saved per year", min_value=100, max_value=100000,
                            value=int(pr["k"]["value"]), step=500, key=f"k_{key}",
                            help="SFOE (2022): average 18,000 kWh/year in monitored kitchens using about 120 MWh/year.")
cost = st.sidebar.number_input("Price (CHF)" if not INCREMENTAL else "Extra cost of induction (CHF)",
                               min_value=0, max_value=500000, value=int(pr["I0_gross"]["value"]), step=500, key=f"I0_{key}")
subsidy = st.sidebar.number_input("Subsidy (CHF)", min_value=0, max_value=500000, value=0, step=500,
                                  help="Base case 0: current availability of subsidies is not verified.")
T = st.sidebar.slider("Lifetime (years)", 5, 30, int(pr["T"]["value"]),
                      help="RAVEL (1992): 15–25 years for electrical/mechanical plant; SFOE (2022) assumes 30.")

st.sidebar.header("4. Discount rate and tariff growth")
RF = stats["rf"]
spread = st.sidebar.slider("Spread over the 10-year Confederation rate (points)", 0.0, 6.0, 2.5, 0.1,
                           help=f"Risk-free rate from the SNB data: {RF:.2%} (last observation of {stats['rf_year']}). The spread is an ASSUMPTION.")
r = RF + spread / 100
st.sidebar.caption(f"Discount rate r = {RF:.2%} + {spread:.1f} pts = **{r:.2%}**")

med = stats["med"]
fall = (med[2020] / med[LATEST]) ** (1 / 10) - 1
scen = st.sidebar.selectbox("Tariff growth scenario", [
    "Flat (0 % a year)",
    f"Historical average ({stats['mean_g']:+.1%} a year, 2020–{LATEST})",
    f"Fall back to the 2020 level over 10 years ({fall:+.1%} a year)",
    "Custom",
])
if scen.startswith("Flat"):
    g = 0.0
elif scen.startswith("Historical"):
    g = stats["mean_g"]
elif scen.startswith("Fall back"):
    g = fall
else:
    g = st.sidebar.slider("Tariff growth g (% a year)", -10.0, 10.0, 0.0, 0.5) / 100

with st.sidebar.expander("Advanced"):
    share = st.slider("Share of the tariff avoided per saved kWh", 0.3, 1.0, 1.0, 0.05,
                      help="ElCom's C3 average includes demand charges that saved kWh may not avoid.")
    M = st.number_input("Extra yearly cost (CHF)", 0, 20000, 0, 50)
    delta = st.slider("Yearly decline of savings", 0.0, 0.05, 0.0, 0.005)
    st.markdown("**Risk inputs** (defaults estimated from the data)")
    sd_annual = st.number_input("Annual sd of tariff changes", 0.0, 1.0, round(stats["sd_g"], 3), 0.01)
    sd_r = st.number_input("Annual sd of 10-year rate change (decimal)", 0.0, 0.05, round(stats["sd_r"], 4), 0.0005, format="%.4f")
    rho = st.number_input("Correlation tariff change / rate change", -2.0, 2.0, round(stats["rho"], 2), 0.05,
                          help="Values outside [-1, 1] are invalid and will be rejected.")

# Validate inputs up front
I0 = cost - subsidy
try:
    if I0 < 0:
        raise ValueError("The subsidy cannot exceed the price: the net investment would be negative.")
    P = m.Project(I0=float(I0), k=float(k), p0=float(p0), T=int(T), M=float(M), delta=float(delta), share=float(share))
    m.validate(P, g, r)
except ValueError as e:
    st.error(f"Invalid input: {e}")
    st.stop()

# Baseline (flat tariffs, default spread) versus the chosen scenario
R_BASE = RF + 0.025
summ = lambda Pj, gg, rr: {
    "NPV (CHF)": m.npv(Pj, gg, rr),
    "Break-even tariff (Rp./kWh)": m.break_even_tariff(Pj, gg, rr) * 100,
    "Break-even savings (kWh/yr)": m.break_even_savings(Pj, gg, rr),
    "Max affordable investment (CHF)": m.max_affordable_investment(Pj, gg, rr),
    "Simple payback (years)": m.simple_payback(Pj),
    "Discounted payback (years)": m.discounted_payback(Pj, gg, rr),
}
res = summ(P, g, r)
base = summ(P, 0.0, R_BASE)


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("⚡ Does an energy-saving investment pay off?")
st.markdown(
    "For **a small Swiss business** (ElCom profile C3, about 150 MWh/year, e.g. a restaurant kitchen), this tool helps decide "
    "**whether to replace a commercial hob with induction**, using **ElCom tariffs 2020–2027, SNB interest rates and an NPV model** "
    "with break-even, sensitivity and risk analysis."
)
with st.expander("How to use"):
    st.markdown(
        "1. Choose the decision and the location in the sidebar.\n"
        "2. Adjust the investment (savings, price, lifetime) and the scenario (discount rate, tariff growth).\n"
        "3. Read the **Result** tab first, then check **What drives it**, **Risk**, **Replace now or wait**, **Where in Switzerland** "
        "and **Method & sources**.\n\n"
        "Every figure is recomputed from your inputs. Inputs that come from sources and inputs that are assumptions are listed in *Method & sources*."
    )

tabs = st.tabs(["Result", "What drives it", "Risk", "Replace now or wait", "Where in Switzerland", "Method & sources"])


def fig_show(fig):
    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------------------------------
# Tab 1: Result
# --------------------------------------------------------------------------
with tabs[0]:
    c = st.columns(4)
    c[0].metric("NPV", f"CHF {res['NPV (CHF)']:,.0f}")
    c[1].metric("Break-even tariff", f"{res['Break-even tariff (Rp./kWh)']:.1f} Rp./kWh", f"today {p0*100:.1f}", delta_color="off")
    c[2].metric("Break-even savings", f"{res['Break-even savings (kWh/yr)']:,.0f} kWh/yr", f"yours {k:,.0f}", delta_color="off")
    dp = res["Discounted payback (years)"]
    c[3].metric("Discounted payback", f"{dp} years" if dp else "not within lifetime", f"simple {res['Simple payback (years)']:.1f} y", delta_color="off")

    if res["NPV (CHF)"] >= 0:
        st.success(f"Under these inputs the investment **pays off** (NPV ≥ 0). It would still pay off with savings down to "
                   f"{res['Break-even savings (kWh/yr)']:,.0f} kWh/yr or a tariff down to {res['Break-even tariff (Rp./kWh)']:.1f} Rp./kWh.")
    else:
        st.warning(f"Under these inputs the investment **does not pay off** (NPV < 0). It would need savings of at least "
                   f"{res['Break-even savings (kWh/yr)']:,.0f} kWh/yr or a tariff of at least {res['Break-even tariff (Rp./kWh)']:.1f} Rp./kWh.")

    st.subheader("Baseline versus your scenario")
    comp = pd.DataFrame({"Baseline (flat tariff, spread 2.5 pts)": base, "Your scenario": res}).T
    comp.columns = list(res.keys())
    st.dataframe(comp.T.style.format("{:,.1f}", na_rep="not within lifetime"), width="stretch")

    st.subheader("Cumulative discounted cash flow")
    t = np.arange(1, P.T + 1)
    cum = np.concatenate([[-P.I0], -P.I0 + np.cumsum(m.cash_flows(P, g) / (1 + r) ** t)])
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(np.arange(0, P.T + 1), cum, marker="o")
    ax.axhline(0, color="red", ls="--", lw=1)
    ax.set(xlabel="year", ylabel="CHF", title="Money recovered over time (discounted)")
    ax.grid(alpha=0.3)
    fig_show(fig)
    st.caption("The curve crosses zero in the year of the discounted payback. The final value equals the NPV.")


# --------------------------------------------------------------------------
# Tab 2: What drives it
# --------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Tornado: NPV swing when one input moves over its range")
    clip = lambda lo_hi: (max(lo_hi[0] - subsidy, 0.0), max(lo_hi[1] - subsidy, 0.0))
    rng = {"I0": clip(pr["I0_gross"]["range"]), "k": pr["k"]["range"], "T": pr["T"]["range"], "M": pr["M"]["range"],
           "share": pr["share"]["range"],
           "p0": (float(muni[year].quantile(0.1)) / 100, float(muni[year].quantile(0.9)) / 100),
           "g": (-0.05, 0.05), "r": (0.01, 0.06)}
    names = {"I0": "investment", "k": "kWh saved / year", "T": "lifetime", "M": "extra yearly cost", "share": "tariff share avoided",
             "p0": "tariff (p10–p90)", "g": "tariff growth (−5…+5 %)", "r": "discount rate (1…6 %)"}
    tor = m.tornado(P, g, r, rng)
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (n_, lo, hi) in enumerate(reversed(tor)):
        ax.barh(i, abs(hi - lo), left=min(lo, hi), alpha=0.75)
    ax.set_yticks(range(len(tor)))
    ax.set_yticklabels([names[x[0]] for x in reversed(tor)])
    ax.axvline(res["NPV (CHF)"], color="k", lw=1)
    ax.axvline(0, color="red", ls="--", lw=1)
    ax.set(xlabel="NPV (CHF)")
    ax.grid(alpha=0.3)
    fig_show(fig)
    st.caption("Black line = your NPV; red dashed = break-even (NPV 0). Ranges: documented ranges or assumed ±50 % (see Method & sources), tariff p10–p90 across municipalities, growth and rate as labelled.")

    st.subheader("Break-even frontier: tariff × kWh saved")
    ks = np.linspace(max(200, k * 0.05), max(k * 2, 5000), 200)
    ps = np.linspace(0.05, 0.40, 200)
    KK, PP = np.meshgrid(ks, ps)
    A_, B_ = m._A(P, r), m._B(P, g, r)
    NPV = -P.I0 - P.M * A_ + PP * P.share * KK * B_
    fig, ax = plt.subplots(figsize=(8, 4.5))
    lim = max(abs(NPV.min()), abs(NPV.max()))
    cs = ax.contourf(KK, PP * 100, NPV, levels=np.linspace(-lim, lim, 21), cmap="RdYlGn")
    ax.contour(KK, PP * 100, NPV, levels=[0], colors="k", linewidths=2)
    ax.plot(k, p0 * 100, "ko", ms=8)
    ax.annotate("you are here", (k, p0 * 100), (k * 1.05, p0 * 100 + 4))
    plt.colorbar(cs, label="NPV (CHF)")
    ax.set(xlabel="kWh saved per year", ylabel="tariff avoided (Rp./kWh)")
    fig_show(fig)
    st.caption("NPV is linear in the product tariff × kWh saved, so the break-even line is a hyperbola. Green = pays off.")


# --------------------------------------------------------------------------
# Tab 3: Risk
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_risk(P, g, r, sd_annual, sd_r, rho, n=5000):
    S = m.make_sigma(m.sigma_g_horizon(sd_annual, P.T), sd_r, rho)
    return S, m.delta_method(P, g, r, S), m.monte_carlo(P, g, r, S, n=n, seed=1)


with tabs[2]:
    st.subheader("Uncertainty in tariff growth and the discount rate")
    st.markdown(
        "Tariff growth `g` and the rate `r` are treated as random with covariance Σ estimated from history "
        f"(tariff changes: {stats['n_g']} annual observations; yield changes: {stats['n_r']}; correlation based on {stats['n_rho']}). "
        "The **delta method** approximates the NPV's mean and variance from the gradient and Hessian; "
        "the **exact** column recomputes the NPV for 5,000 random draws."
    )
    try:
        S, d, mc = run_risk(P, g, r, sd_annual, sd_r, rho)
    except ValueError as e:
        st.error(f"Invalid risk inputs: {e}")
    else:
        c = st.columns(3)
        c[0].metric("P(NPV > 0), exact", f"{mc['p_positive']:.0%}")
        c[1].metric("P(NPV > 0), delta method", f"{d['p_positive']:.0%}")
        c[2].metric("5%–95% range of NPV", f"{mc['p05']:,.0f} … {mc['p95']:,.0f}")
        tbl = pd.DataFrame({
            "Delta method": [d["mean"], d["sd"], d["p_positive"]],
            "Exact (Monte Carlo)": [mc["mean"], mc["sd"], mc["p_positive"]],
        }, index=["Mean NPV (CHF)", "Std. deviation (CHF)", "P(NPV > 0)"])
        st.dataframe(tbl.style.format("{:,.2f}"), width="stretch")
        x = np.linspace(mc["samples"].min(), mc["samples"].max(), 300)
        pdf = np.exp(-(x - d["mean"]) ** 2 / (2 * d["sd"] ** 2)) / (d["sd"] * math.sqrt(2 * math.pi)) if d["sd"] > 0 else None
        fig, ax = plt.subplots(figsize=(8, 3.6))
        ax.hist(mc["samples"], bins=60, density=True, alpha=0.5, label="exact (Monte Carlo)")
        if pdf is not None:
            ax.plot(x, pdf, "k", label="delta-method normal")
        ax.axvline(0, color="red", ls="--")
        ax.set(xlabel="NPV (CHF)")
        ax.legend()
        fig_show(fig)
        if mc["p_positive"] > 0.99:
            st.info("The probability of a positive NPV is essentially 1: uncertainty in tariff growth and the discount rate does not change the decision here. "
                    "The kWh actually saved matter more (see *What drives it*).")
        st.caption("The horizon-average growth has standard deviation sd/√T, which assumes independent annual changes (an assumption). Σ must be positive semidefinite; invalid correlations are rejected.")


# --------------------------------------------------------------------------
# Tab 4: Replace now or wait
# --------------------------------------------------------------------------
def breakeven_k_timing(Pj, gg, rr, n, lo=1.0, hi=100000.0):
    f = lambda kk: m.replace_now_vs_wait(replace(Pj, k=kk), gg, rr, n)["now_minus_wait"]
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if f(mid) > 0 else (mid, hi)
    return (lo + hi) / 2


with tabs[3]:
    st.subheader("Replace a working appliance now, or wait until it fails?")
    if INCREMENTAL:
        st.info("This comparison uses the full purchase price. Switch the decision in the sidebar to *Replace a working hob now*.")
    else:
        n = st.slider("Years the old hob could still run", 1, int(P.T), min(5, int(P.T)))
        out = pd.DataFrame(m.timing_table(P, g, r, range(0, int(P.T) + 1))).set_index("n")
        now_minus_wait = out.loc[n, "now_minus_wait"]
        hurdle = breakeven_k_timing(P, g, r, n)
        c = st.columns(3)
        c[0].metric(f"Replace now − wait {n} years", f"CHF {now_minus_wait:,.0f}")
        c[1].metric("Savings needed to replace now", f"{hurdle:,.0f} kWh/yr", f"yours {k:,.0f}", delta_color="off")
        c[2].metric("Savings for NPV = 0", f"{res['Break-even savings (kWh/yr)']:,.0f} kWh/yr")
        if now_minus_wait > 0:
            st.success("Replacing now beats waiting under these inputs.")
        else:
            st.warning("Waiting until the old hob fails is better under these inputs.")
        fig, ax = plt.subplots(figsize=(8, 3.6))
        ax.plot(out.index, out["now_minus_wait"], marker="o")
        ax.axhline(0, color="red", ls="--", lw=1)
        ax.axvline(n, color="grey", lw=1)
        ax.set(xlabel="remaining life of the old hob (years)", ylabel="PV(replace now) − PV(wait), CHF")
        ax.grid(alpha=0.3)
        fig_show(fig)
        st.caption("Plans are compared over the common horizon n + T with the full price in both. Assumptions: the same price for any future purchase; no salvage value for the old hob.")


# --------------------------------------------------------------------------
# Tab 5: Where in Switzerland
# --------------------------------------------------------------------------
with tabs[4]:
    st.subheader(f"Same investment, different tariffs ({year} tariff level)")
    ty = tar[tar["year"] == year]
    mm = ty.groupby(["kanton", "gemeindeNummer"])["variable_tariff"].median().reset_index()
    mm["p0"] = mm["variable_tariff"] / 100
    slope = m.dnpv_dp0(P, g, r)                       # NPV is linear in the tariff: exact for any tariff
    mm["NPV"] = res["NPV (CHF)"] + slope * (mm["p0"] - P.p0)
    mm["Break-even savings (kWh/yr)"] = [m.break_even_savings(replace(P, p0=float(x)), g, r) for x in mm["p0"]]
    pos = (mm["NPV"] >= 0).mean()
    c = st.columns(3)
    n_pos = int((mm["NPV"] >= 0).sum())
    c[0].metric("Municipalities with NPV ≥ 0", f"{n_pos:,} of {len(mm):,}", f"{pos:.1%}", delta_color="off")
    c[1].metric("Lowest break-even savings", f"{mm['Break-even savings (kWh/yr)'].min():,.0f} kWh/yr")
    c[2].metric("Highest break-even savings", f"{mm['Break-even savings (kWh/yr)'].max():,.0f} kWh/yr")
    metric = st.radio("Show by canton", ["Break-even savings (kWh/yr)", "NPV", "Tariff (Rp./kWh)"], horizontal=True)
    bc = mm.groupby("kanton").agg(**{"Break-even savings (kWh/yr)": ("Break-even savings (kWh/yr)", "median"),
                                      "NPV": ("NPV", "median"), "Tariff (Rp./kWh)": ("variable_tariff", "median")}).sort_values(metric)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(bc.index, bc[metric])
    if metric == "Break-even savings (kWh/yr)":
        ax.axhline(k, color="red", ls="--", label=f"your savings {k:,.0f} kWh/yr")
        ax.legend()
    elif metric == "NPV":
        ax.axhline(0, color="red", ls="--")
    ax.set(ylabel=metric)
    ax.grid(alpha=0.3, axis="y")
    plt.xticks(rotation=90)
    fig_show(fig)
    st.caption("Median across municipalities of each canton. NPV is exactly linear in the tariff, so each municipality's NPV follows from the current NPV and the tariff difference. "
               "Standard-product tariffs; larger customers may buy on the free market.")
    st.download_button("Download municipality results (CSV)", mm.round(3).to_csv(index=False).encode(), "municipality_results.csv", "text/csv")


# --------------------------------------------------------------------------
# Tab 6: Method & sources
# --------------------------------------------------------------------------
with tabs[5]:
    st.subheader("Model")
    st.latex(r"NPV(g,r) = -I_0 + \sum_{t=1}^{T} \frac{k\,(1-\delta)^{t-1}\,\mathrm{share}\,p_0\,(1+g)^{t-1} - M}{(1+r)^t}")
    st.markdown("Year 1 uses today's known tariff `p0`; year *t* uses `p0·(1+g)^(t−1)`. `I0` = price or extra cost minus subsidy.")
    st.markdown("**Break-even tariff** (closed form, because NPV is linear in `p0`): "
                "`p0* = (I0 + M·A) / (k·share·B)` with `A = Σ 1/(1+r)^t` and `B = Σ ((1−δ)(1+g))^(t−1)/(1+r)^t`.")

    st.subheader("Taylor approximation vs exact NPV")
    cA, cB = st.columns(2)
    dg = cA.slider("Change in growth g (points)", -5.0, 5.0, 3.0, 0.5) / 100
    dr = cB.slider("Change in discount rate r (points)", -3.0, 3.0, 1.0, 0.5) / 100
    try:
        rep = m.taylor_report(P, g, r, dg, dr)
        st.dataframe(pd.DataFrame({"CHF": rep}).style.format("{:,.0f}"), width="stretch")
        st.caption("First order uses the gradient, second order adds ½·hᵀHh with the Hessian.")
    except ValueError as e:
        st.error(str(e))

    st.subheader("Where each number comes from")
    rows = [{"parameter": kk, "value": pr[kk]["value"], "range": str(pr[kk]["range"]), "status": pr[kk]["status"]}
            for kk in ("I0_gross", "subsidy", "k", "T", "M", "delta", "share")]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.subheader("Data")
    st.markdown(
        "| Data | Provider | Period | Unit |\n|---|---|---|---|\n"
        "| Electricity tariffs, standard product, profile C3 | ElCom (via LINDAS raw tariff data) | tariff years 2020–2027 | Rp./kWh |\n"
        "| 10-year Confederation spot rate | Swiss National Bank data portal | daily to 2026-10-30 | % p.a. |\n"
        "| Savings, price, lifetime of induction hobs | SFOE (2022), measure 31a, p. 59 | EcoGastro monitoring | kWh, CHF |\n"
        "| Discount-rate and lifetime guidance | Leemann (1992), RAVEL | – | – |\n\n"
        "Retrieval dates: *fill in before submission.* Cleaning: variable tariff = energy + grid use + levies + grid surcharge in every year "
        "(metering costs are included in `total` only from 2026); 83 placeholder rows with tariff ≤ 1 Rp. removed."
    )
    st.subheader("Limits")
    st.markdown(
        "- Savings are an average of monitored kitchens (about 120 MWh/year); the price and the 40 % premium are assumptions of the SFOE report, not invoices.\n"
        "- The 2.5-point spread over the risk-free rate is an assumption (RAVEL: the rate should be at most the cost of new long-term borrowing).\n"
        "- Only 8 annual tariff points exist (7 changes), so growth volatility and the correlation with rates are rough.\n"
        "- ElCom's C3 figure is an average at a standard load; demand charges may not be avoided; large customers may use the free market.\n"
        "- Taxes, residual values and inflation effects are ignored. This is a decision aid, not financial advice."
    )
