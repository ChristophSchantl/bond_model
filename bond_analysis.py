# bond_lab_pro.py
# ---------------------------------------------------------
# Bond Evaluation & Analysis Lab — PRO
# ---------------------------------------------------------
# Features:
# - Zero-Kurve ohne Upload: Flat(bench), Standard steigend/invers, Manuell (Editor)
# - Z-Spread (gegen gewählte Zero-Kurve)
# - Hazard-Rate / Survival (impliziert aus Preis & Recovery)
# - Key Rate Duration (lokale Kurven-Bumps)
# - Carry & Roll-Down + Pull-to-Par-Pfad
# - Stress/Tornado: Yield, Spread, Recovery, Hazard
# - Charts: Cashflow-Timeline (Coupons vs. Tilgung), PV-Stack, PV-Donut, kumulativer PV,
#           Price↔Yield, Zero vs Zero+Z, KRD-Bar, Survival, Pull-to-Par, Tornado

from datetime import date
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Monatsarithmetik
try:
    from dateutil.relativedelta import relativedelta
except Exception:
    class relativedelta:  # Fallback
        def __init__(self, months=0, years=0):
            self._months = months + years*12

# ---------- Standard-Zero-Kurven (ohne Upload) ----------
def build_standard_curve(mode: str, bench_flat: float, r_short=2.50, r_long=3.50):
    """
    mode: "Flat (Bench)" | "Standard steigend" | "Standard invers"
    bench_flat, r_short, r_long: in Prozent
    returns: (t_nodes [Jahre], r_nodes [dezimal])
    """
    tenors = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 30], dtype=float)
    if mode == "Flat (Bench)":
        rates = np.full_like(tenors, bench_flat/100.0, dtype=float)
    else:
        rates = np.linspace(r_short/100.0, r_long/100.0, len(tenors)).astype(float)
    return tenors, rates

# ------------------------------
# Day-count & Schedule
# ------------------------------
def year_fraction(d1: date, d2: date, convention: str, period_start: date = None, period_end: date = None) -> float:
    c = convention.upper()
    if c in ("ACT/365", "ACT/365F", "ACT/365 FIXED"):
        return (d2 - d1).days / 365.0
    if c in ("ACT/360",):
        return (d2 - d1).days / 360.0
    if c in ("30/360", "30U/360", "30E/360"):
        d1_y, d1_m, d1_d = d1.year, d1.month, min(d1.day, 30)
        d2_y, d2_m, d2_d = d2.year, d2.month, min(d2.day, 30)
        return (360*(d2_y-d1_y)+30*(d2_m-d1_m)+(d2_d-d1_d))/360.0
    if c in ("ACT/ACT", "ACT/ACT-ISDA"):
        if (period_start is None) or (period_end is None):
            return (d2 - d1).days / 365.0
        return (d2 - d1).days / (period_end - period_start).days
    return (d2 - d1).days / 365.0

def build_schedule(settlement: date, maturity: date, frequency: str):
    freq_to_months = {"Annual":12, "Semiannual":6, "Quarterly":3, "Monthly":1}
    step = freq_to_months[frequency]
    dates = [maturity]
    cur = maturity
    safety = 0
    while True:
        nxt = cur - relativedelta(months=step)
        dates.append(nxt); cur = nxt; safety += 1
        if nxt <= settlement or safety > 1000:
            break
    dates = sorted([d for d in dates if d > settlement])
    prev_coupon = cur
    return dates, prev_coupon

def accrued_interest(settlement, prev_coupon, next_coupon, coupon_rate, nominal, frequency, day_count):
    per_year = {"Annual":1,"Semiannual":2,"Quarterly":4,"Monthly":12}[frequency]
    coupon_amt = coupon_rate*nominal/per_year
    full = year_fraction(prev_coupon, next_coupon, day_count, prev_coupon, next_coupon)
    run  = year_fraction(prev_coupon, settlement,   day_count, prev_coupon, next_coupon)
    frac = 0.0 if full<=0 else max(0.0, min(1.0, run/full))
    return coupon_amt*frac

def build_cashflows(settlement, maturity, coupon_rate, nominal, frequency, day_count):
    per_year = {"Annual":1,"Semiannual":2,"Quarterly":4,"Monthly":12}[frequency]
    schedule, prev_coupon = build_schedule(settlement, maturity, frequency)
    coupon_amt = coupon_rate*nominal/per_year
    flows=[]
    for dt_pay in schedule:
        amt = coupon_amt
        if dt_pay == schedule[-1]:
            amt += nominal
        flows.append((dt_pay, amt))
    next_coupon = schedule[0] if schedule else maturity
    ai = accrued_interest(settlement, prev_coupon, next_coupon, coupon_rate, nominal, frequency, day_count)
    return flows, ai, schedule, prev_coupon, next_coupon

# ------------------------------
# Pricing / Yield
# ------------------------------
def discount_factor(y_annual: float, t_years: float, m: int) -> float:
    return 1.0/((1.0+y_annual/m)**(m*t_years))

def price_from_yield(y_annual, settlement, cashflows, m):
    pv=0.0
    for dt_pay, amt in cashflows:
        t=(dt_pay-settlement).days/365.0
        pv += amt * discount_factor(y_annual, t, m)
    return pv

def ytm_from_price(target_dirty, settlement, cashflows, m):
    lo, hi = 0.0, 10.0
    for _ in range(200):
        mid = 0.5*(lo+hi)
        p_mid = price_from_yield(mid, settlement, cashflows, m)
        p_lo  = price_from_yield(lo,  settlement, cashflows, m)
        if (p_lo - target_dirty)*(p_mid - target_dirty) <= 0: hi=mid
        else: lo=mid
    return 0.5*(lo+hi)

def ear_from_apr(apr, m):
    return (1.0+apr/m)**m - 1.0

def duration_convexity(y, settlement, cashflows, m, bp=1e-4):
    p0 = price_from_yield(y, settlement, cashflows, m)
    p_up = price_from_yield(y+bp, settlement, cashflows, m)
    p_dn = price_from_yield(y-bp, settlement, cashflows, m)
    dP = (p_dn - p_up)/2.0
    mod_dur = (dP/p0)/bp
    convex  = (p_up + p_dn - 2.0*p0)/(p0*bp*bp)
    dv01 = mod_dur * p0 * 1e-4
    return mod_dur, convex, dv01, p0

# ------------------------------
# Zero-Kurve & Z-Spread
# ------------------------------
def interp_rate(t_nodes, r_nodes, t):
    if t<=t_nodes[0]: return float(r_nodes[0])
    if t>=t_nodes[-1]: return float(r_nodes[-1])
    i = np.searchsorted(t_nodes, t)-1
    t0,t1 = t_nodes[i], t_nodes[i+1]
    r0,r1 = r_nodes[i], r_nodes[i+1]
    w = (t - t0)/(t1 - t0 + 1e-12)
    return float(r0 + w*(r1 - r0))

def df_from_zero_curve(t_nodes, r_nodes, t):
    r = interp_rate(t_nodes, r_nodes, t)
    return math.exp(-r*t)

def pv_with_zero_and_z(cashflows, settlement, t_nodes, r_nodes, z):
    pv=0.0
    for dt, cf in cashflows:
        t=(dt-settlement).days/365.0
        r = interp_rate(t_nodes, r_nodes, t) + z
        pv += cf*math.exp(-r*t)
    return pv

def solve_z_spread(dirty_price, cashflows, settlement, t_nodes, r_nodes):
    lo, hi = -0.5, 2.0
    for _ in range(140):
        mid=0.5*(lo+hi)
        pv = pv_with_zero_and_z(cashflows, settlement, t_nodes, r_nodes, mid)
        pv_lo = pv_with_zero_and_z(cashflows, settlement, t_nodes, r_nodes, lo)
        if (pv_lo - dirty_price)*(pv - dirty_price) <= 0: hi=mid
        else: lo=mid
    return 0.5*(lo+hi)

# ------------------------------
# Hazard / Survival (FRP)
# ------------------------------
def price_with_hazard_FRP(cashflows, settlement, t_nodes, r_nodes, lam, R, nominal):
    pv=0.0; prev_t=0.0
    for dt, cf in cashflows:
        t=(dt-settlement).days/365.0
        S = math.exp(-lam*t)
        df = df_from_zero_curve(t_nodes, r_nodes, t)
        pv += cf*df*S
        dS = math.exp(-lam*prev_t) - math.exp(-lam*t)
        pv += (R*nominal) * df * dS
        prev_t = t
    return pv

def solve_hazard_from_price(dirty_price, cashflows, settlement, t_nodes, r_nodes, R, nominal):
    lo, hi = 0.0, 5.0
    for _ in range(140):
        mid=0.5*(lo+hi)
        pv  = price_with_hazard_FRP(cashflows, settlement, t_nodes, r_nodes, mid, R, nominal)
        pv_lo = price_with_hazard_FRP(cashflows, settlement, t_nodes, r_nodes, lo, R, nominal)
        if (pv_lo - dirty_price)*(pv - dirty_price) <= 0: hi=mid
        else: lo=mid
    return 0.5*(lo+hi)

# ------------------------------
# Key Rate Duration & Carry
# ------------------------------
def pv_on_curve(cashflows, settlement, t_nodes, r_nodes):
    return sum(cf*df_from_zero_curve(t_nodes, r_nodes, (dt-settlement).days/365.0) for dt, cf in cashflows)

def bump_curve_local(t_nodes, r_nodes, center, bump_bp=25, width=1.0):
    bump=bump_bp/10000.0; r_b=r_nodes.copy().astype(float)
    for i,t in enumerate(t_nodes):
        w=max(0.0, 1.0-abs(t-center)/width); r_b[i]=r_nodes[i]+bump*w
    return r_b

def key_rate_duration(cashflows, settlement, t_nodes, r_nodes, tenors=(1,2,5,7,10), bump_bp=25):
    base=pv_on_curve(cashflows, settlement, t_nodes, r_nodes)
    out={}
    for T in tenors:
        r_up=bump_curve_local(t_nodes, r_nodes, T, bump_bp=bump_bp, width=1.0)
        pv_up=pv_on_curve(cashflows, settlement, t_nodes, r_up)
        out[f"KRD {T}y"]=(pv_up-base)/base/(bump_bp/10000.0)
    return out

def survival_price_path(y_apr, settlement, cashflows, m, months=24):
    pts=[]; cur=settlement
    for k in range(months+1):
        p=price_from_yield(y_apr, cur, cashflows, m); pts.append((k,p)); cur=cur+relativedelta(months=1)
    return pts

# ------------------------------
# UI
# ------------------------------
st.set_page_config(page_title="Bond Evaluation & Analysis Lab — PRO", page_icon="💹", layout="wide")
st.title("💹 Bond Evaluation & Analysis Lab — PRO")
st.caption("YTM/YTW, Z-Spread, Hazard/Survival, KRD, Carry/Roll, Stress — mit professionellen Visualisierungen.")

with st.sidebar:
    st.header("Instrument")
    name = st.text_input("Name", value="Urbanek Real Estate GmbH 10% 25/32")
    nominal = st.number_input("Nominal (pro Stück)", value=100.0, step=1.0, format="%.2f")
    coupon_pct = st.number_input("Kupon (% p.a.)", value=10.0, step=0.01, format="%.4f")
    frequency = st.selectbox("Kuponfrequenz", ["Monthly","Quarterly","Semiannual","Annual"], index=0)
    day_count = st.selectbox("Day-Count", ["ACT/ACT","ACT/365","ACT/360","30/360"], index=0)
    today = date.today()
    settlement = st.date_input("Settlement", value=today)
    maturity = st.date_input("Fälligkeit", value=date(2032,2,3))

    st.divider()
    st.subheader("Preis & Benchmark")
    price_type = st.radio("Preis ist", ["Clean","Dirty"], index=0, horizontal=True)
    price_pct = st.number_input("Preis (% vom Nominal)", value=20.00, step=0.01, format="%.4f")
    bench_flat = st.number_input("Flat-Benchmark (% p.a.)", value=2.00, step=0.05)

    st.divider()
    st.subheader("Zero-Kurve (ohne Upload)")
    curve_mode = st.selectbox(
        "Quelle",
        ["Flat (Bench)", "Standard steigend", "Standard invers", "Manuell (Editor)"],
        index=1
    )
    if curve_mode in ["Flat (Bench)", "Standard steigend", "Standard invers"]:
        if curve_mode == "Flat (Bench)":
            t_nodes, r_nodes = build_standard_curve("Flat (Bench)", bench_flat)
            st.caption(f"Flat @ {bench_flat:.2f}% p.a. über alle Laufzeiten.")
        else:
            cA, cB = st.columns(2)
            with cA:
                r_short = st.number_input("Kurzfrist-Rate (0–2J) %", value=2.50, step=0.05, format="%.2f")
            with cB:
                default_long = 3.50 if curve_mode == "Standard steigend" else 2.00
                r_long  = st.number_input("Langfrist-Rate (10–30J) %", value=default_long, step=0.05, format="%.2f")
            t_nodes, r_nodes = build_standard_curve(curve_mode, bench_flat, r_short, r_long)
            st.caption(f"{curve_mode}: {r_short:.2f}% ➜ {r_long:.2f}% (linear über Standard-Tenors).")
    else:
        st.caption("Manuelle Kurve: Tenors in Jahren, Rate in % (bearbeitbar).")
        df_default = pd.DataFrame({
            "t_years": [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 30],
            "rate_%": [2.50, 2.55, 2.60, 2.70, 2.85, 3.00, 3.10, 3.20, 3.30, 3.35]
        })
        df_edit = st.data_editor(df_default, num_rows="dynamic", use_container_width=True)
        df_clean = df_edit.dropna().sort_values("t_years")
        t_nodes = df_clean["t_years"].to_numpy(dtype=float)
        r_nodes = (df_clean["rate_%"].to_numpy(dtype=float))/100.0

    st.divider()
    st.subheader("Credit Annahmen")
    recov = st.slider("Recovery (R, Anteil vom Par)", 0.0, 0.8, 0.20, 0.01)

    st.divider()
    preset = st.checkbox("Urbanek-Preset setzen", True)

if preset:
    coupon_pct = 10.0; frequency="Monthly"; price_pct=20.00; maturity=date(2032,2,3)

coupon = coupon_pct/100.0
m = {"Annual":1,"Semiannual":2,"Quarterly":4,"Monthly":12}[frequency]

# Cashflows & Accrued
cashflows, ai_abs, schedule, prev_coupon, next_coupon = build_cashflows(settlement, maturity, coupon, nominal, frequency, day_count)
ai_pct = ai_abs/nominal*100.0

# Clean/Dirty
if price_type=="Clean":
    clean_pct=price_pct; dirty_pct=price_pct+ai_pct
else:
    dirty_pct=price_pct; clean_pct=max(0.0, price_pct-ai_pct)
dirty_abs = dirty_pct/100.0*nominal

# Renditen & Risiko
y_apr = ytm_from_price(dirty_abs, settlement, cashflows, m)
y_ear = ear_from_apr(y_apr, m)
mod_dur, convex, dv01, model_dirty = duration_convexity(y_apr, settlement, cashflows, m)
model_clean_pct=(model_dirty-ai_abs)/nominal*100.0
model_dirty_pct=model_dirty/nominal*100.0

tab_pricing, tab_spreads, tab_credit, tab_carry, tab_stress = st.tabs(["Pricing","Spreads","Credit / Hazard","Carry / Roll","Stress"])

# ---------- PRICING ----------
with tab_pricing:
    c1,c2,c3=st.columns(3)
    with c1:
        st.subheader("Renditen")
        st.metric("Nominal APR (p.a.)", f"{y_apr*100:.2f}%")
        st.metric("Effektiver Jahreszins (EAR)", f"{y_ear*100:.2f}%")
    with c2:
        st.subheader("Preise")
        st.metric("Accrued (pro 100)", f"{ai_pct:.4f}%")
        st.metric("Model Dirty Price", f"{model_dirty_pct:.2f}%")
        st.metric("Model Clean Price", f"{model_clean_pct:.2f}%")
    with c3:
        st.subheader("Risiko")
        st.metric("Modified Duration", f"{mod_dur:.3f}")
        st.metric("Convexity", f"{convex:.3f}")
        st.metric("DV01", f"{dv01:.4f}")

    st.divider()
    st.subheader("Cashflow-Timeline & PV-Profile")

    # Build CF table with components & PV
    rows=[]; last_dt = cashflows[-1][0]; settle_ts = pd.Timestamp(settlement)
    for dt, amt in cashflows:
        is_mat = (dt == last_dt)
        cpn = amt - (nominal if is_mat else 0.0)
        prin = (nominal if is_mat else 0.0)
        t = (pd.Timestamp(dt) - settle_ts).days/365.0
        dfy = np.exp(-np.log(1 + y_apr/m) * (m*t))
        rows.append({
            "date": pd.Timestamp(dt),
            "year": pd.Timestamp(dt).year,
            "coupon": cpn,
            "principal": prin,
            "pv_coupon": cpn*dfy,
            "pv_principal": prin*dfy
        })
    cf = pd.DataFrame(rows)

    colA, colB = st.columns([2,1])
    with colA:
        agg_mode = st.radio("Aggregation", ["Monatlich", "Jährlich"], horizontal=True, index=1)
    with colB:
        yscale = st.radio("Skalierung", ["Linear", "Log"], horizontal=True)

    if agg_mode == "Jährlich":
        plot_df = (cf.groupby("year", as_index=False)[["coupon","principal","pv_coupon","pv_principal"]].sum())
        x_vals = plot_df["year"].astype(str)
    else:
        plot_df = cf.copy()
        x_vals = plot_df["date"]

    # (1) Coupons vs. Tilgung (Doppelachse)
    fig_cf = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
    coup_y = plot_df["coupon"] if yscale=="Linear" else plot_df["coupon"].replace(0, np.nan)
    prin_y = plot_df["principal"] if yscale=="Linear" else plot_df["principal"].replace(0, np.nan)
    fig_cf.add_bar(x=x_vals, y=coup_y, name="Coupons", secondary_y=False)
    fig_cf.add_bar(x=x_vals, y=prin_y, name="Tilgung", opacity=0.55, secondary_y=True)
    fig_cf.update_layout(template="plotly_white", height=360, barmode="group")
    fig_cf.update_yaxes(title_text="Coupons", secondary_y=False, type=("log" if yscale=="Log" else "linear"))
    fig_cf.update_yaxes(title_text="Tilgung", secondary_y=True, type=("log" if yscale=="Log" else "linear"))
    st.plotly_chart(fig_cf, use_container_width=True)

    # (2) PV gestapelt
    fig_pv = go.Figure()
    fig_pv.add_bar(x=x_vals, y=plot_df["pv_coupon"], name="PV Coupons")
    fig_pv.add_bar(x=x_vals, y=plot_df["pv_principal"], name="PV Tilgung")
    fig_pv.update_layout(template="plotly_white", height=320, barmode="stack", yaxis_title="Present Value")
    st.plotly_chart(fig_pv, use_container_width=True)

    # (3) PV-Donut
    pv_tot = plot_df[["pv_coupon","pv_principal"]].sum()
    fig_pie = go.Figure(go.Pie(labels=["PV Coupons","PV Tilgung"], values=pv_tot.values, hole=0.45))
    fig_pie.update_layout(template="plotly_white", height=300)
    st.plotly_chart(fig_pie, use_container_width=True)

    # (4) kumulativer PV
    cf_sorted = cf.sort_values("date").copy()
    cf_sorted["pv_total"] = cf_sorted["pv_coupon"] + cf_sorted["pv_principal"]
    cf_sorted["pv_cum"] = cf_sorted["pv_total"].cumsum()
    fig_cum = go.Figure(go.Scatter(x=cf_sorted["date"], y=cf_sorted["pv_cum"], mode="lines", name="Cum PV"))
    fig_cum.update_layout(template="plotly_white", height=300, yaxis_title="PV kumuliert")
    st.plotly_chart(fig_cum, use_container_width=True)

    # Price ↔ Yield
    st.subheader("Price ↔ Yield")
    ys=np.linspace(0.0, max(2.0, y_apr*1.2), 220)
    prices=[price_from_yield(y, settlement, cashflows, m)/nominal*100.0 for y in ys]
    fig_py=go.Figure()
    fig_py.add_trace(go.Scatter(x=ys*100, y=prices, mode="lines", name="Dirty Price"))
    fig_py.add_trace(go.Scatter(x=[y_apr*100], y=[model_dirty_pct], mode="markers", name="Aktuell", marker=dict(size=10)))
    fig_py.update_layout(height=360, template="plotly_white", xaxis_title="Yield (APR, % p.a.)", yaxis_title="Price (% of Nominal)")
    st.plotly_chart(fig_py, use_container_width=True)

# ---------- SPREADS ----------
with tab_spreads:
    # Immer verfügbar, da Kurve gewählt/editiert wird
    z = solve_z_spread(dirty_abs, cashflows, settlement, t_nodes, r_nodes)
    st.metric("Z-Spread", f"{z*10000:.0f} bp")

    ts=np.linspace(0, max((schedule[-1]-settlement).days/365.0, t_nodes[-1]), 50)
    base=[interp_rate(t_nodes, r_nodes, t)*100 for t in ts]
    plus=[(interp_rate(t_nodes, r_nodes, t)+z)*100 for t in ts]
    fig_curve=go.Figure()
    fig_curve.add_trace(go.Scatter(x=ts, y=base, mode="lines", name="Zero"))
    fig_curve.add_trace(go.Scatter(x=ts, y=plus, mode="lines", name="Zero + Z"))
    fig_curve.update_layout(height=340, template="plotly_white", xaxis_title="t (Jahre)", yaxis_title="Rate (% p.a.)")
    st.plotly_chart(fig_curve, use_container_width=True)

    tenors = st.multiselect("KRD Tenors (Jahre)", [0.5,1,2,3,5,7,10], default=[1,2,5,7,10])
    bump = st.slider("Bump (bp)", 1, 100, 25, 1)
    krd = key_rate_duration(cashflows, settlement, t_nodes, r_nodes, tenors=tuple(tenors), bump_bp=bump)
    if krd:
        krd_df=pd.DataFrame({"tenor":list(krd.keys()), "krd":list(krd.values())})
        fig_krd=go.Figure(); fig_krd.add_bar(x=krd_df["tenor"], y=krd_df["krd"])
        fig_krd.update_layout(height=300, template="plotly_white", xaxis_title="Key Rate", yaxis_title="Duration (per 1)")
        st.plotly_chart(fig_krd, use_container_width=True)

# ---------- CREDIT / HAZARD ----------
with tab_credit:
    lam = solve_hazard_from_price(dirty_abs, cashflows, settlement, t_nodes, r_nodes, recov, nominal)
    c1,c2,c3=st.columns(3)
    c1.metric("Impl. Hazard λ", f"{lam*100:.2f}% p.a.")
    c2.metric("PD 1Y", f"{(1-math.exp(-lam*1.0))*100:.2f}%")
    c3.metric("PD 3Y", f"{(1-math.exp(-lam*3.0))*100:.2f}%")
    st.metric("PD 5Y", f"{(1-math.exp(-lam*5.0))*100:.2f}%")

    Ts=np.linspace(0, (schedule[-1]-settlement).days/365.0, 60)
    S=[math.exp(-lam*t) for t in Ts]
    figS=go.Figure(); figS.add_trace(go.Scatter(x=Ts, y=S, mode="lines", name="Survival S(t)"))
    figS.update_layout(height=340, template="plotly_white", xaxis_title="t (Jahre)", yaxis_title="S(t)")
    st.plotly_chart(figS, use_container_width=True)

    Rs=np.linspace(0.0, 0.6, 13)
    lam_R=[solve_hazard_from_price(dirty_abs, cashflows, settlement, t_nodes, r_nodes, R, nominal) for R in Rs]
    figLamR=go.Figure(); figLamR.add_trace(go.Scatter(x=Rs*100, y=[l*100 for l in lam_R], mode="lines+markers"))
    figLamR.update_layout(height=300, template="plotly_white", xaxis_title="Recovery (%)", yaxis_title="Implizite λ (% p.a.)")
    st.plotly_chart(figLamR, use_container_width=True)

# ---------- CARRY / ROLL ----------
with tab_carry:
    st.subheader("Pull-to-Par (Survival, konst. APR)")
    path=survival_price_path(y_apr, settlement, cashflows, m, months=24)
    fig_pull=go.Figure()
    fig_pull.add_trace(go.Scatter(x=[k for k,_ in path], y=[p/nominal*100.0 for _,p in path], mode="lines+markers"))
    fig_pull.add_hline(y=100.0, line_dash="dot", annotation_text="Par")
    fig_pull.update_layout(height=340, template="plotly_white", xaxis_title="Monate", yaxis_title="Price (% of Nominal)")
    st.plotly_chart(fig_pull, use_container_width=True)

    h6=settlement+relativedelta(months=6); h12=settlement+relativedelta(months=12)
    p6=price_from_yield(y_apr, h6, cashflows, m)/nominal*100.0
    p12=price_from_yield(y_apr, h12, cashflows, m)/nominal*100.0
    carry_1y=coupon*100.0; rd6=p6-model_dirty_pct; rd12=p12-model_dirty_pct
    ca,cb,cc=st.columns(3)
    ca.metric("Carry 1Y (Kupons)", f"{carry_1y:.2f}%")
    cb.metric("Roll-down 6M", f"{rd6:.2f} pts")
    cc.metric("Roll-down 12M", f"{rd12:.2f} pts")

# ---------- STRESS ----------
with tab_stress:
    st.subheader("Tornado — ΔPrice (Dirty) unter Stress")
    scenarios=[]
    # Yield ±100bp
    y_up=price_from_yield(y_apr+0.01, settlement, cashflows, m)
    y_dn=price_from_yield(y_apr-0.01, settlement, cashflows, m)
    scenarios.append(("Yield +100bp", (y_up-model_dirty)/nominal*100.0))
    scenarios.append(("Yield -100bp", (y_dn-model_dirty)/nominal*100.0))
    # Z-Spread ±100bp
    z0=solve_z_spread(model_dirty, cashflows, settlement, t_nodes, r_nodes)
    p_sp_up=pv_with_zero_and_z(cashflows, settlement, t_nodes, r_nodes, z0+0.01)
    p_sp_dn=pv_with_zero_and_z(cashflows, settlement, t_nodes, r_nodes, z0-0.01)
    scenarios.append(("Z-Spread +100bp", (p_sp_up-model_dirty)/nominal*100.0))
    scenarios.append(("Z-Spread -100bp", (p_sp_dn-model_dirty)/nominal*100.0))
    # Recovery & Hazard
    lam0=solve_hazard_from_price(model_dirty, cashflows, settlement, t_nodes, r_nodes, recov, nominal)
    for R in [0.10,0.40]:
        pR=price_with_hazard_FRP(cashflows, settlement, t_nodes, r_nodes, lam0, R, nominal)
        scenarios.append((f"Recovery {int(R*100)}%", (pR-model_dirty)/nominal*100.0))
    for s in [0.5,1.5]:
        lam_s=lam0*s
        ps=price_with_hazard_FRP(cashflows, settlement, t_nodes, r_nodes, lam_s, recov, nominal)
        scenarios.append((f"λ x{s:.1f}", (ps-model_dirty)/nominal*100.0))

    sc_df=pd.DataFrame(scenarios, columns=["Szenario","ΔPrice (pts)"]).sort_values("ΔPrice (pts)")
    fig_t=go.Figure(); fig_t.add_bar(x=sc_df["ΔPrice (pts)"], y=sc_df["Szenario"], orientation="h")
    fig_t.update_layout(height=400, template="plotly_white", xaxis_title="ΔPrice (Prozentpunkte)", margin=dict(l=120,r=40,t=40,b=40))
    st.plotly_chart(fig_t, use_container_width=True)

st.caption("© Bond Evaluation & Analysis Lab — PRO. Diskrete Verzinsung (m=Frequenz); Hazard: FRP-Approximation; Z-Spread vs. gewählte Zero-Kurve.")
