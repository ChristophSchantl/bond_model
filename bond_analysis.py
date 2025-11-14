# bond_lab_lite.py
# ---------------------------------------------------------
# Bond Evaluation & Analysis Lab — LITE (clean & pro)
# ---------------------------------------------------------
# Features:
# 1) Pricing: YTM (APR/EAR), Clean/Dirty, Accrued, Duration/Convexity/DV01
# 2) Zero-Kurve ohne Upload: Flat / Steigend / Invers (kurz/ lang justierbar)
# 3) Z-Spread gegen gewählte Kurve
# 4) Credit: implizite Hazard-Rate (FRP), PD 1/3/5Y, Survival-Kurve
# 5) Carry: Pull-to-Par (24M)
# 6) Stress (Tornado): Yield ±100bp, Z-Spread ±100bp, Recovery 10/40, λ x0.5/x1.5

from __future__ import annotations
from datetime import date
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---- robust month arithmetic
try:
    from dateutil.relativedelta import relativedelta
except Exception:
    class relativedelta:  # minimal fallback
        def __init__(self, months=0, years=0): self._months = months + 12*years

# =========================
# Zero curve (no upload)
# =========================
def build_zero_curve(mode: str, bench_flat_pct: float, r_short_pct: float = 2.5, r_long_pct: float = 3.5):
    """Return (tenors in years, zero rates in decimals)."""
    tenors = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 30], dtype=float)
    if mode == "Flat":
        rates = np.full_like(tenors, bench_flat_pct/100.0, dtype=float)
    elif mode == "Steigend":
        rates = np.linspace(r_short_pct/100.0, r_long_pct/100.0, len(tenors)).astype(float)
    elif mode == "Invers":
        rates = np.linspace(r_long_pct/100.0, r_short_pct/100.0, len(tenors)).astype(float)
    else:
        raise ValueError("mode must be Flat | Steigend | Invers")
    return tenors, rates

def interp_zero(t_nodes: np.ndarray, r_nodes: np.ndarray, t: float) -> float:
    if t <= t_nodes[0]: return float(r_nodes[0])
    if t >= t_nodes[-1]: return float(r_nodes[-1])
    i = np.searchsorted(t_nodes, t) - 1
    t0, t1 = t_nodes[i], t_nodes[i+1]
    r0, r1 = r_nodes[i], r_nodes[i+1]
    w = (t - t0) / (t1 - t0 + 1e-12)
    return float(r0 + w*(r1 - r0))

def df_zero(t_nodes: np.ndarray, r_nodes: np.ndarray, t: float) -> float:
    return math.exp(-interp_zero(t_nodes, r_nodes, t) * t)

# =========================
# Day count, schedule, cashflows
# =========================
def year_fraction(d1: date, d2: date, conv: str, period_start: date | None = None, period_end: date | None = None) -> float:
    c = conv.upper()
    if c in ("ACT/365", "ACT/365F"): return (d2 - d1).days / 365.0
    if c in ("ACT/360",):           return (d2 - d1).days / 360.0
    if c in ("30/360", "30E/360"):
        y1,m1,d1_ = d1.year, d1.month, min(d1.day, 30)
        y2,m2,d2_ = d2.year, d2.month, min(d2.day, 30)
        return (360*(y2-y1) + 30*(m2-m1) + (d2_-d1_)) / 360.0
    # ACT/ACT (simple)
    return (d2 - d1).days / 365.0

def build_schedule(settlement: date, maturity: date, frequency: str):
    step = {"Annual":12, "Semiannual":6, "Quarterly":3, "Monthly":1}[frequency]
    dates = [maturity]
    cur = maturity
    safety = 0
    while True:
        nxt = cur - relativedelta(months=step)
        dates.append(nxt); cur = nxt; safety += 1
        if nxt <= settlement or safety > 1000: break
    dates = sorted([d for d in dates if d > settlement])
    prev_coupon = cur
    return dates, prev_coupon

def accrued_interest(settlement: date, prev_coupon: date, next_coupon: date,
                     coupon_rate: float, nominal: float, frequency: str, day_count: str) -> float:
    per_year = {"Annual":1,"Semiannual":2,"Quarterly":4,"Monthly":12}[frequency]
    coupon_amt = coupon_rate * nominal / per_year
    full = year_fraction(prev_coupon, next_coupon, day_count, prev_coupon, next_coupon)
    run  = year_fraction(prev_coupon, settlement, day_count, prev_coupon, next_coupon)
    frac = 0.0 if full <= 0 else max(0.0, min(1.0, run/full))
    return coupon_amt * frac

def build_cashflows(settlement: date, maturity: date, coupon_rate: float, nominal: float, frequency: str, day_count: str):
    per_year = {"Annual":1,"Semiannual":2,"Quarterly":4,"Monthly":12}[frequency]
    sched, prev_coupon = build_schedule(settlement, maturity, frequency)
    coupon_amt = coupon_rate * nominal / per_year
    flows = []
    for dt in sched:
        amt = coupon_amt + (nominal if dt == sched[-1] else 0.0)
        flows.append((dt, amt))
    next_coupon = sched[0] if sched else maturity
    ai = accrued_interest(settlement, prev_coupon, next_coupon, coupon_rate, nominal, frequency, day_count)
    return flows, ai, sched

# =========================
# Pricing & risk
# =========================
def discount_factor_y(y_apr: float, t: float, m: int) -> float:
    return 1.0 / ((1.0 + y_apr/m)**(m*t))

def price_from_yield(y_apr: float, settlement: date, cashflows, m: int) -> float:
    return sum(cf * discount_factor_y(y_apr, (dt - settlement).days/365.0, m) for dt, cf in cashflows)

def ytm_from_price(target_dirty: float, settlement: date, cashflows, m: int) -> float:
    lo, hi = 0.0, 10.0
    for _ in range(160):
        mid = 0.5*(lo+hi)
        p_mid = price_from_yield(mid, settlement, cashflows, m)
        p_lo  = price_from_yield(lo,  settlement, cashflows, m)
        if (p_lo-target_dirty)*(p_mid-target_dirty) <= 0: hi = mid
        else: lo = mid
    return 0.5*(lo+hi)

def ear_from_apr(apr: float, m: int) -> float:
    return (1.0 + apr/m)**m - 1.0

def duration_convexity(y_apr: float, settlement: date, cashflows, m: int, bp: float = 1e-4):
    p0 = price_from_yield(y_apr, settlement, cashflows, m)
    p_up = price_from_yield(y_apr + bp, settlement, cashflows, m)
    p_dn = price_from_yield(y_apr - bp, settlement, cashflows, m)
    dP = (p_dn - p_up) / 2.0
    mod_dur = (dP / p0) / bp
    convex  = (p_up + p_dn - 2.0*p0) / (p0 * bp * bp)
    dv01    = mod_dur * p0 * 1e-4
    return mod_dur, convex, dv01, p0

# =========================
# Z-Spread & Hazard (FRP)
# =========================
def pv_zero_plus_z(cashflows, settlement: date, t_nodes, r_nodes, z: float) -> float:
    pv = 0.0
    for dt, cf in cashflows:
        t = (dt - settlement).days/365.0
        r = interp_zero(t_nodes, r_nodes, t) + z
        pv += cf * math.exp(-r*t)
    return pv

def solve_z_spread(dirty_price: float, cashflows, settlement: date, t_nodes, r_nodes) -> float:
    lo, hi = -0.5, 2.0
    for _ in range(120):
        mid = 0.5*(lo+hi)
        pv_mid = pv_zero_plus_z(cashflows, settlement, t_nodes, r_nodes, mid)
        pv_lo  = pv_zero_plus_z(cashflows, settlement, t_nodes, r_nodes, lo)
        if (pv_lo - dirty_price)*(pv_mid - dirty_price) <= 0: hi = mid
        else: lo = mid
    return 0.5*(lo+hi)

def price_hazard_frp(cashflows, settlement: date, t_nodes, r_nodes, lam: float, R: float, nominal: float) -> float:
    pv, prev_t = 0.0, 0.0
    for dt, cf in cashflows:
        t  = (dt - settlement).days/365.0
        S  = math.exp(-lam*t)
        df = df_zero(t_nodes, r_nodes, t)
        pv += cf * df * S
        dS = math.exp(-lam*prev_t) - math.exp(-lam*t)
        pv += (R * nominal) * df * dS
        prev_t = t
    return pv

def solve_hazard_from_price(dirty_price: float, cashflows, settlement: date, t_nodes, r_nodes, R: float, nominal: float) -> float:
    lo, hi = 0.0, 5.0
    for _ in range(120):
        mid = 0.5*(lo+hi)
        pv_mid = price_hazard_frp(cashflows, settlement, t_nodes, r_nodes, mid, R, nominal)
        pv_lo  = price_hazard_frp(cashflows, settlement, t_nodes, r_nodes, lo, R, nominal)
        if (pv_lo - dirty_price)*(pv_mid - dirty_price) <= 0: hi = mid
        else: lo = mid
    return 0.5*(lo+hi)

def pull_to_par_path(y_apr: float, settlement: date, cashflows, m: int, months: int = 24):
    pts, cur = [], settlement
    for k in range(months+1):
        p = price_from_yield(y_apr, cur, cashflows, m)
        pts.append((k, p))
        cur = cur + relativedelta(months=1)
    return pts

# =========================
# UI
# =========================
st.set_page_config(page_title="Bond Lab — LITE", page_icon="💹", layout="wide")
st.title("💹 Bond Lab — LITE")

with st.sidebar:
    st.header("Instrument")
    name      = st.text_input("Name", value="Urbanek Real Estate GmbH 10% 25/32")
    nominal   = st.number_input("Nominal (pro Stück)",  value=100.0, step=1.0, format="%.2f")
    couponPct = st.number_input("Kupon (% p.a.)",      value=10.0, step=0.01, format="%.4f")
    freq      = st.selectbox("Kuponfrequenz", ["Monthly","Quarterly","Semiannual","Annual"], index=0)
    day_count = st.selectbox("Day-Count", ["ACT/ACT","ACT/365","ACT/360","30/360"], index=0)
    settlement= st.date_input("Settlement", value=date.today())
    maturity  = st.date_input("Fälligkeit", value=date(2032,2,3))

    st.divider()
    st.subheader("Preis & Kurve")
    price_type = st.radio("Preis ist", ["Clean","Dirty"], index=0, horizontal=True)
    price_pct  = st.number_input("Preis (% vom Nominal)", value=20.00, step=0.01, format="%.4f")
    bench_flat = st.number_input("Flat-Benchmark (% p.a.)", value=2.00, step=0.05)
    curve_mode = st.radio("Zero-Kurve", ["Flat","Steigend","Invers"], index=1, horizontal=True)
    if curve_mode != "Flat":
        c1, c2 = st.columns(2)
        r_short = c1.number_input("Kurzfrist-Rate (%)", value=2.50, step=0.05, format="%.2f")
        r_long  = c2.number_input("Langfrist-Rate  (%)", value=(3.50 if curve_mode=="Steigend" else 2.00), step=0.05, format="%.2f")
    else:
        r_short = 2.50; r_long = 3.50

    st.divider()
    st.subheader("Credit")
    recovery = st.slider("Recovery (Anteil Par)", 0.0, 0.8, 0.20, 0.01)

# presets (optional)
coupon = couponPct/100.0
m      = {"Annual":1,"Semiannual":2,"Quarterly":4,"Monthly":12}[freq]

# Build flows & accrued
flows, accrued, sched = build_cashflows(settlement, maturity, coupon, nominal, freq, day_count)
accrued_pct = accrued/nominal*100.0
if price_type == "Clean":
    dirty_pct = price_pct + accrued_pct
else:
    dirty_pct = price_pct
dirty_abs = dirty_pct/100.0*nominal

# Pricing & risk
y_apr = ytm_from_price(dirty_abs, settlement, flows, m)
y_ear = ear_from_apr(y_apr, m)
mod_dur, convex, dv01, model_dirty = duration_convexity(y_apr, settlement, flows, m)
model_dirty_pct = model_dirty/nominal*100.0
model_clean_pct = (model_dirty - accrued)/nominal*100.0

# Zero curve
t_nodes, r_nodes = build_zero_curve(curve_mode, bench_flat, r_short, r_long)

# Tabs
tab_prc, tab_curve, tab_credit, tab_stress = st.tabs(["Pricing", "Curve & Z-Spread", "Credit", "Stress"])

# --- Pricing
with tab_prc:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("YTM (APR)", f"{y_apr*100:.2f}%")
        st.metric("YTM (EAR)", f"{y_ear*100:.2f}%")
    with c2:
        st.metric("Accrued (pro 100)", f"{accrued_pct:.4f}%")
        st.metric("Model Dirty", f"{model_dirty_pct:.2f}%")
        st.metric("Model Clean", f"{model_clean_pct:.2f}%")
    with c3:
        st.metric("Modified Duration", f"{mod_dur:.3f}")
        st.metric("Convexity", f"{convex:.3f}")
        st.metric("DV01", f"{dv01:.4f}")

    st.divider()
    st.subheader("Cashflows (jährlich, Coupon vs Tilgung) & PV")

    # annual aggregate for clarity
    last_dt = sched[-1]
    rows = []
    for dt, amt in flows:
        is_mat = (dt == last_dt)
        cpn = amt - (nominal if is_mat else 0.0)
        prin = nominal if is_mat else 0.0
        t = (pd.Timestamp(dt) - pd.Timestamp(settlement)).days/365.0
        dfy = math.exp(-math.log(1 + y_apr/m) * (m*t))
        rows.append({"year": pd.Timestamp(dt).year,
                     "coupon": cpn, "principal": prin,
                     "pv_coupon": cpn*dfy, "pv_principal": prin*dfy})
    agg = pd.DataFrame(rows).groupby("year", as_index=False).sum()

    fig_cf = go.Figure()
    fig_cf.add_bar(x=agg["year"], y=agg["coupon"],    name="Coupons")
    fig_cf.add_bar(x=agg["year"], y=agg["principal"], name="Tilgung", opacity=0.6)
    fig_cf.update_layout(template="plotly_white", height=320, barmode="group", yaxis_title="Betrag")
    st.plotly_chart(fig_cf, use_container_width=True)

    fig_pv = go.Figure()
    fig_pv.add_bar(x=agg["year"], y=agg["pv_coupon"],    name="PV Coupons")
    fig_pv.add_bar(x=agg["year"], y=agg["pv_principal"], name="PV Tilgung")
    fig_pv.update_layout(template="plotly_white", height=300, barmode="stack", yaxis_title="Present Value")
    st.plotly_chart(fig_pv, use_container_width=True)

    st.subheader("Price ↔ Yield")
    ys = np.linspace(0.0, max(2.0, y_apr*1.2), 220)
    prices = [price_from_yield(y, settlement, flows, m)/nominal*100.0 for y in ys]
    fig_py = go.Figure()
    fig_py.add_trace(go.Scatter(x=ys*100, y=prices, name="Dirty Price", mode="lines"))
    fig_py.add_trace(go.Scatter(x=[y_apr*100], y=[model_dirty_pct], name="Aktuell", mode="markers", marker=dict(size=10)))
    fig_py.update_layout(template="plotly_white", height=320, xaxis_title="Yield (APR, %)", yaxis_title="Price (% of Par)")
    st.plotly_chart(fig_py, use_container_width=True)

# --- Curve & Spread
with tab_curve:
    z = solve_z_spread(dirty_abs, flows, settlement, t_nodes, r_nodes)
    st.metric("Z-Spread", f"{z*10000:.0f} bp")

    ts = np.linspace(0.0, max((sched[-1]-settlement).days/365.0, t_nodes[-1]), 60)
    base = [interp_zero(t_nodes, r_nodes, t)*100 for t in ts]
    plus = [(interp_zero(t_nodes, r_nodes, t)+z)*100 for t in ts]
    fig_curve = go.Figure()
    fig_curve.add_trace(go.Scatter(x=ts, y=base, name="Zero", mode="lines"))
    fig_curve.add_trace(go.Scatter(x=ts, y=plus, name="Zero + Z", mode="lines"))
    fig_curve.update_layout(template="plotly_white", height=320, xaxis_title="t (Jahre)", yaxis_title="Rate (% p.a.)")
    st.plotly_chart(fig_curve, use_container_width=True)

# --- Credit
with tab_credit:
    lam = solve_hazard_from_price(dirty_abs, flows, settlement, t_nodes, r_nodes, recovery, nominal)
    c1,c2,c3 = st.columns(3)
    c1.metric("Implizite Hazard λ", f"{lam*100:.2f}% p.a.")
    c2.metric("PD 1Y", f"{(1-math.exp(-lam*1.0))*100:.2f}%")
    c3.metric("PD 3Y", f"{(1-math.exp(-lam*3.0))*100:.2f}%")
    st.metric("PD 5Y", f"{(1-math.exp(-lam*5.0))*100:.2f}%")

    # Survival curve
    T_end = (sched[-1] - settlement).days/365.0
    Ts = np.linspace(0, T_end, 60)
    S  = [math.exp(-lam*t) for t in Ts]
    figS = go.Figure(go.Scatter(x=Ts, y=S, mode="lines", name="S(t)"))
    figS.update_layout(template="plotly_white", height=300, xaxis_title="t (Jahre)", yaxis_title="Survival S(t)")
    st.plotly_chart(figS, use_container_width=True)

    # Pull-to-par (24M)
    st.subheader("Pull-to-Par (24 Monate, konst. APR)")
    path = pull_to_par_path(y_apr, settlement, flows, m, months=24)
    fig_pull = go.Figure(go.Scatter(x=[k for k,_ in path], y=[p/nominal*100.0 for _,p in path], mode="lines+markers"))
    fig_pull.add_hline(y=100.0, line_dash="dot", annotation_text="Par")
    fig_pull.update_layout(template="plotly_white", height=300, xaxis_title="Monate", yaxis_title="Price (% of Par)")
    st.plotly_chart(fig_pull, use_container_width=True)

# --- Stress
with tab_stress:
    st.subheader("Tornado — ΔPrice (Dirty) in Punkten")
    scenarios = []

    y_up = price_from_yield(y_apr+0.01, settlement, flows, m)
    y_dn = price_from_yield(y_apr-0.01, settlement, flows, m)
    scenarios += [("Yield +100bp", (y_up-model_dirty)/nominal*100.0),
                  ("Yield -100bp", (y_dn-model_dirty)/nominal*100.0)]

    z0 = solve_z_spread(model_dirty, flows, settlement, t_nodes, r_nodes)
    p_sp_up = pv_zero_plus_z(flows, settlement, t_nodes, r_nodes, z0+0.01)
    p_sp_dn = pv_zero_plus_z(flows, settlement, t_nodes, r_nodes, z0-0.01)
    scenarios += [("Z-Spread +100bp", (p_sp_up-model_dirty)/nominal*100.0),
                  ("Z-Spread -100bp", (p_sp_dn-model_dirty)/nominal*100.0)]

    lam0 = solve_hazard_from_price(model_dirty, flows, settlement, t_nodes, r_nodes, recovery, nominal)
    for R in [0.10, 0.40]:
        pR = price_hazard_frp(flows, settlement, t_nodes, r_nodes, lam0, R, nominal)
        scenarios.append((f"Recovery {int(R*100)}%", (pR-model_dirty)/nominal*100.0))
    for s in [0.5, 1.5]:
        ps = price_hazard_frp(flows, settlement, t_nodes, r_nodes, lam0*s, recovery, nominal)
        scenarios.append((f"λ x{s:.1f}", (ps-model_dirty)/nominal*100.0))

    sc = pd.DataFrame(scenarios, columns=["Szenario","ΔPrice"]).sort_values("ΔPrice")
    fig_t = go.Figure(go.Bar(x=sc["ΔPrice"], y=sc["Szenario"], orientation="h"))
    fig_t.update_layout(template="plotly_white", height=380, xaxis_title="ΔPrice (Prozentpunkte)",
                        margin=dict(l=120, r=40, t=30, b=40))
    st.plotly_chart(fig_t, use_container_width=True)

st.caption("© Bond Lab — LITE. Diskrete Verzinsung (m=Frequenz). Hazard via FRP-Approximation. Z-Spread gegen gewählte Zero-Kurve.")
