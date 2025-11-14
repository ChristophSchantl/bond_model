# bond_lab.py
# ---------------------------------------------------------
# Bond Evaluation & Analysis Lab
# Professionelles, kompaktes Streamlit-Tool zur Bewertung
# von Anleihen mit exakter YTM-Lösung, Accrued, Duration,
# DV01, Convexity, Price↔Yield-Kurve und 1Y-Default/Recovery.
# ---------------------------------------------------------

from datetime import date, datetime, timedelta

# Optional, aber empfohlen (für saubere Monats-Schritte)
try:
    from dateutil.relativedelta import relativedelta
except Exception:  # Fallback, falls dateutil fehlt
    class relativedelta:  # type: ignore
        def __init__(self, months=0, years=0):
            self._months = months + years * 12
    def _add_months(dt, months):
        y = dt.year + (dt.month - 1 + months) // 12
        m = (dt.month - 1 + months) % 12 + 1
        d = min(
            dt.day,
            [31, 29 if y % 4 == 0 and (y % 100 != 0 or y % 400 == 0) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][
                m - 1
            ],
        )
        return date(y, m, d)

import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ------------------------------
# Day-count & Schedule Helpers
# ------------------------------
def year_fraction(d1: date, d2: date, convention: str, period_start: date = None, period_end: date = None) -> float:
    """Day-count fraction (d1 -> d2)."""
    c = convention.upper()
    if c in ("ACT/365", "ACT/365F", "ACT/365 FIXED"):
        return (d2 - d1).days / 365.0
    if c in ("ACT/360",):
        return (d2 - d1).days / 360.0
    if c in ("30/360", "30U/360", "30E/360"):
        # 30/360 US (vereinfachte Variante)
        d1_y, d1_m, d1_d = d1.year, d1.month, min(d1.day, 30)
        d2_y, d2_m, d2_d = d2.year, d2.month, min(d2.day, 30)
        return (360 * (d2_y - d1_y) + 30 * (d2_m - d1_m) + (d2_d - d1_d)) / 360.0
    if c in ("ACT/ACT", "ACT/ACT-ISDA"):
        if (period_start is None) or (period_end is None):
            return (d2 - d1).days / 365.0
        return (d2 - d1).days / (period_end - period_start).days
    # Default
    return (d2 - d1).days / 365.0


def build_schedule(settlement: date, maturity: date, frequency: str):
    """Erzeuge zukünftige Zahlungsdaten > settlement, durch Rückwärtslaufen ab Fälligkeit."""
    freq_to_months = {"Annual": 12, "Semiannual": 6, "Quarterly": 3, "Monthly": 1}
    if frequency not in freq_to_months:
        raise ValueError("Unsupported frequency.")
    step = freq_to_months[frequency]

    dates = [maturity]
    cur = maturity
    # Rückwärts erzeugen, bis wir einen Termin vor/gleich Settlement erreichen
    safety = 0
    while True:
        nxt = cur - relativedelta(months=step)
        dates.append(nxt)
        cur = nxt
        safety += 1
        if nxt <= settlement or safety > 1000:
            break

    dates = sorted([d for d in dates if d > settlement])
    # Vorheriger Coupontermin (für Accrued)
    prev = cur  # das ist der erste Termin <= settlement
    return dates, prev


def accrued_interest(settlement: date, prev_coupon: date, next_coupon: date,
                     coupon_rate: float, nominal: float, frequency: str, day_count: str) -> float:
    """Berechne Stückzinsen des aktuellen Kupon-Periods."""
    per_year = {"Annual": 1, "Semiannual": 2, "Quarterly": 4, "Monthly": 12}[frequency]
    coupon_amt = coupon_rate * nominal / per_year

    full = year_fraction(prev_coupon, next_coupon, day_count, prev_coupon, next_coupon)
    run  = year_fraction(prev_coupon, settlement,   day_count, prev_coupon, next_coupon)
    frac = 0.0 if full <= 0 else max(0.0, min(1.0, run / full))
    return coupon_amt * frac


def build_cashflows(settlement: date, maturity: date, coupon_rate: float, nominal: float,
                    frequency: str, day_count: str):
    """Liste der zukünftigen Cashflows (ab settlement)."""
    per_year = {"Annual": 1, "Semiannual": 2, "Quarterly": 4, "Monthly": 12}[frequency]
    schedule, prev_coupon = build_schedule(settlement, maturity, frequency)
    coupon_amt = coupon_rate * nominal / per_year

    flows = []
    for dt_pay in schedule:
        amt = coupon_amt
        if dt_pay == schedule[-1]:
            amt += nominal
        flows.append((dt_pay, amt))

    next_coupon = schedule[0] if schedule else maturity
    ai = accrued_interest(settlement, prev_coupon, next_coupon, coupon_rate, nominal, frequency, day_count)
    return flows, ai, schedule, prev_coupon, next_coupon


# ------------------------------
# Pricing / Yield / Risk
# ------------------------------
def discount_factor(y_annual: float, t_years: float, comp_per_year: int) -> float:
    """Diskrete Verzinsung m = comp_per_year: DF = 1 / (1 + y/m)^(m*t)."""
    m = comp_per_year
    return 1.0 / ((1.0 + y_annual / m) ** (m * t_years))


def price_from_yield(y_annual: float, settlement: date, cashflows: list, comp_per_year: int) -> float:
    """Dirty Price (pro Nominal 1.0) bei gegebenem APR-y_annual."""
    pv = 0.0
    for dt_pay, amt in cashflows:
        t = (dt_pay - settlement).days / 365.0
        pv += amt * discount_factor(y_annual, t, comp_per_year)
    return pv


def ytm_from_price(target_dirty: float, settlement: date, cashflows: list, comp_per_year: int) -> float:
    """Exakte YTM via Bisection (APR, p.a., diskret m-kompoundiert)."""
    low, high = 0.0, 10.0  # bis 1000% p.a.
    for _ in range(200):
        mid = 0.5 * (low + high)
        p_mid = price_from_yield(mid, settlement, cashflows, comp_per_year)
        p_low = price_from_yield(low, settlement, cashflows, comp_per_year)
        if (p_low - target_dirty) * (p_mid - target_dirty) <= 0:
            high = mid
        else:
            low = mid
    return 0.5 * (low + high)


def effective_annual_from_apr(apr: float, m: int) -> float:
    """EAR = (1 + APR/m)^m - 1"""
    return (1.0 + apr / m) ** m - 1.0


def duration_convexity(y_annual: float, settlement: date, cashflows: list, comp_per_year: int, bp: float = 1e-4):
    """Finite-Difference um y_annual: Modified Duration, Convexity, DV01 (Dirty-Basis)."""
    p0 = price_from_yield(y_annual, settlement, cashflows, comp_per_year)
    p_up = price_from_yield(y_annual + bp, settlement, cashflows, comp_per_year)
    p_dn = price_from_yield(y_annual - bp, settlement, cashflows, comp_per_year)

    dP = (p_dn - p_up) / 2.0
    mod_dur = (dP / p0) / bp  # %-Preisänderung je 1.0 Renditepunkt (d.h. je 100%)
    convex = (p_up + p_dn - 2.0 * p0) / (p0 * (bp ** 2))
    dv01 = mod_dur * p0 * 1e-4  # Preisänderung für 1 bp
    return mod_dur, convex, dv01, p0


def make_price_yield_curve(settlement, cashflows, comp_per_year, y_min=0.0, y_max=2.0, n=180):
    ys = np.linspace(y_min, y_max, n)
    prices = [price_from_yield(y, settlement, cashflows, comp_per_year) for y in ys]
    return ys, prices


def expected_one_year_return(clean_price_pct, coupon_rate, nominal, frequency,
                             prob_default, recovery_pct):
    """Einfaches 1Y-Modell: Kuponjahr + Wiederverkauf ~ gleicher Clean-Preis; Default mit Recovery."""
    per_year = {"Annual": 1, "Semiannual": 2, "Quarterly": 4, "Monthly": 12}[frequency]
    coupon_income = coupon_rate * nominal * 1.0  # ein Kuponjahr
    start = clean_price_pct / 100.0 * nominal
    end_same = start
    payoff_survive = coupon_income + (end_same - start)
    payoff_default = (recovery_pct * nominal) - start
    exp_payoff = (1.0 - prob_default) * payoff_survive + prob_default * payoff_default
    return exp_payoff / start  # relative Rendite über 1 Jahr


# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Bond Evaluation & Analysis Lab", page_icon="💹", layout="wide")
st.title("💹 Bond Evaluation & Analysis Lab")
st.caption("Exakte YTM (APR & EAR), Accrued, Duration/DV01/Convexity, Price↔Yield-Kurve und 1Y-Default/Recovery.")

with st.sidebar:
    st.header("Instrument")
    name = st.text_input("Name", value="Urbanek Real Estate GmbH 10% 25/32")
    nominal = st.number_input("Nominal (pro Stück)", value=100.0, step=1.0, format="%.2f")
    coupon_pct = st.number_input("Kupon (% p.a.)", value=10.0, step=0.01, format="%.4f")
    frequency = st.selectbox("Kuponfrequenz", ["Monthly", "Quarterly", "Semiannual", "Annual"], index=0)
    day_count = st.selectbox("Day-Count", ["ACT/ACT", "ACT/365", "ACT/360", "30/360"], index=0)

    today = date.today()
    settlement = st.date_input("Settlement", value=today)
    maturity = st.date_input("Fälligkeit", value=date(2032, 2, 3))

    st.divider()
    st.subheader("Preis & Benchmark")
    price_type = st.radio("Preis ist", ["Clean", "Dirty"], index=0, horizontal=True)
    price_pct = st.number_input("Preis (% vom Nominal)", value=20.00, step=0.01, format="%.4f")
    bench_y = st.number_input("Benchmark Yield (% p.a.)", value=0.00, step=0.01)

    st.divider()
    st.subheader("1-Jahres-Szenario")
    p_def = st.slider("Default-Wahrscheinlichkeit 1Y", 0.0, 1.0, 0.20, 0.01)
    recovery = st.slider("Recovery (in % vom Nominal)", 0.0, 1.0, 0.20, 0.01)

    st.divider()
    preset = st.checkbox("Urbanek-Preset setzen", True,
                         help="10% p.a., monatlich, Kurs 20, Fälligkeit 03.02.2032.")

if preset:
    coupon_pct = 10.0
    frequency = "Monthly"
    price_pct = 20.00
    maturity = date(2032, 2, 3)

coupon_rate = coupon_pct / 100.0
comp_per_year = {"Annual": 1, "Semiannual": 2, "Quarterly": 4, "Monthly": 12}[frequency]

# Cashflows & Accrued
cashflows, ai_abs, schedule, prev_coupon, next_coupon = build_cashflows(
    settlement, maturity, coupon_rate, nominal, frequency, day_count
)

# Clean/Dirty angleichen (alles pro 100 Nominal anzeigen)
ai_pct = ai_abs / nominal * 100.0
if price_type == "Clean":
    clean_pct = price_pct
    dirty_pct = price_pct + ai_pct
else:
    dirty_pct = price_pct
    clean_pct = max(0.0, price_pct - ai_pct)

# Solver auf Dirty-Preis in absoluter Größe (pro 100 Nominal)
target_dirty_abs = dirty_pct / 100.0 * nominal

# YTM lösen (APR)
ytm_apr = ytm_from_price(target_dirty_abs, settlement, cashflows, comp_per_year)
ytm_ear = effective_annual_from_apr(ytm_apr, comp_per_year)

# Risiko-Kennzahlen am gelösten YTM
mod_dur, convex, dv01, dirty_model_abs = duration_convexity(ytm_apr, settlement, cashflows, comp_per_year)
clean_model_pct = (dirty_model_abs - ai_abs) / nominal * 100.0
dirty_model_pct = dirty_model_abs / nominal * 100.0

# Header-Block
c1, c2, c3 = st.columns(3)
with c1:
    st.subheader("Renditen")
    st.metric("Nominal APR (p.a.)", f"{ytm_apr*100:.2f}%")
    st.metric("Effektive Jahresrendite (EAR/YTM)", f"{ytm_ear*100:.2f}%")
    st.caption("EAR = (1 + APR/m)^m - 1 (m = Kuponfrequenz/Jahr).")

with c2:
    st.subheader("Preise")
    st.metric("Accrued Interest (pro 100)", f"{ai_pct:.4f}%")
    st.metric("Model Dirty Price", f"{dirty_model_pct:.2f}%")
    st.metric("Model Clean Price", f"{clean_model_pct:.2f}%")

with c3:
    st.subheader("Risiko")
    st.metric("Modified Duration", f"{mod_dur:.3f}")
    st.metric("Convexity", f"{convex:.3f}")
    st.metric("DV01 (pro 1bp)", f"{dv01:.4f}")
    if bench_y > 0.0:
        g_spread_bp = (ytm_apr - bench_y/100.0) * 10000
        st.metric("G-Spread (bp)", f"{g_spread_bp:.0f}")

# Cashflow-Tabelle
st.divider()
st.subheader("Cashflow-Schedule")
cf_rows = []
for dt_pay, amt in cashflows:
    cf_rows.append({
        "Payment Date": dt_pay,
        "Type": "Maturity+Coupon" if dt_pay == cashflows[-1][0] else "Coupon",
        "Amount": amt
    })
cf_df = pd.DataFrame(cf_rows)
st.dataframe(cf_df, use_container_width=True, hide_index=True)

# Price ↔ Yield Kurve
st.divider()
st.subheader("Price ↔ Yield")
ys, ps = make_price_yield_curve(settlement, cashflows, comp_per_year,
                                y_min=0.0, y_max=max(2.0, ytm_apr*1.2), n=220)
fig = go.Figure()
fig.add_trace(go.Scatter(x=[y*100 for y in ys], y=[p/nominal*100 for p in ps],
                         mode="lines", name="Dirty Price"))
fig.add_trace(go.Scatter(x=[ytm_apr*100], y=[dirty_model_pct], mode="markers",
                         marker=dict(size=10), name="Aktuell"))
fig.update_layout(xaxis_title="Yield (APR, % p.a.)",
                  yaxis_title="Price (% of Nominal)",
                  height=420, template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

# 1Y-Szenario
st.divider()
st.subheader("1-Jahres-Szenario (Expected Return)")
exp_1y = expected_one_year_return(clean_price_pct=clean_pct, coupon_rate=coupon_rate, nominal=nominal,
                                  frequency=frequency, prob_default=p_def, recovery_pct=recovery)
st.write(f"**Erwartete 1Y-Rendite** bei p(Default)={p_def:.0%}, Recovery={recovery:.0%}: "
         f"**{exp_1y*100:.2f}%**")

# Export
st.download_button(
    "Download Cashflows CSV",
    cf_df.to_csv(index=False).encode("utf-8"),
    file_name="cashflows.csv",
    mime="text/csv"
)

st.info(
    "Hinweise: Solver arbeitet auf Dirty-Preis und diskreter Verzinsung (m=Frequenz). "
    "Accrued gemäß gewählter Day-Count. Bei exotischen Stubs kann es zu kleinen Abweichungen kommen."
)
st.caption("© Bond Evaluation & Analysis Lab – für professionelle Anwender.")
