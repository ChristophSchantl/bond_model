# streamlit_app.py
# Bond Lab — LITE (Pro) : Pricing, Duration, Convexity, DV01 & Realized Yield (Reinvestment @ money market rate)

import streamlit as st
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

# -----------------------------
# Helpers
# -----------------------------
def price_from_ytm(face, coupon_rate, freq, years, ytm):
    """
    Price per 100 face (clean), discounting with YTM (APR) compounded 'freq' times.
    coupon_rate as % of face per year (e.g. 5.0 for 5%).
    ytm APR (e.g. 0.057 for 5.7%), freq in {1,2,4}.
    """
    c = coupon_rate / 100 * face / freq                # coupon per period in currency
    r = ytm / freq                                     # per-period rate
    n = int(round(years * freq))
    if n <= 0:
        # maturity today
        return face
    pv_coupons = 0.0
    for k in range(1, n + 1):
        pv_coupons += c / ((1 + r) ** k)
    pv_redemption = face / ((1 + r) ** n)
    return pv_coupons + pv_redemption

def ytm_from_price(face, coupon_rate, freq, years, clean_price, tol=1e-10, max_iter=200):
    """
    Solve APR YTM via bisection. Robust für High-Yield/Distressed.
    Returns APR (e.g. 0.5726 for 57.26%).
    """
    # plausible bracket: [-99.9%, 1000%]
    lo, hi = -0.9990, 10.0
    f_lo = price_from_ytm(face, coupon_rate, freq, years, lo) - clean_price
    f_hi = price_from_ytm(face, coupon_rate, freq, years, hi) - clean_price
    # widen if needed
    it_widen = 0
    while f_lo * f_hi > 0 and it_widen < 40:
        hi *= 2.0
        f_hi = price_from_ytm(face, coupon_rate, freq, years, hi) - clean_price
        it_widen += 1

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = price_from_ytm(face, coupon_rate, freq, years, mid) - clean_price
        if abs(f_mid) < tol or (hi - lo) < tol:
            return max(-0.999, mid)
        if f_lo * f_mid < 0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
    return max(-0.999, mid)

def macaulay_duration(face, coupon_rate, freq, years, ytm):
    """
    Macaulay Duration in Jahren, aus Zahlungsreihe (sauberer als Rückrechnung).
    """
    c = coupon_rate / 100 * face / freq
    r = ytm / freq
    n = int(round(years * freq))
    if n <= 0:
        return 0.0
    times = np.array([k / freq for k in range(1, n + 1)])
    cashflows = np.array([c] * (n - 1) + [c + face])
    disc = (1 + r) ** (np.arange(1, n + 1))
    pv = cashflows / disc
    price = pv.sum()
    w = pv / price
    D_mac = np.sum(w * times)
    return D_mac, price

def modified_duration_from_macaulay(D_mac, ytm, freq):
    return D_mac / (1 + ytm / freq)

def convexity(face, coupon_rate, freq, years, ytm):
    """
    Standard discrete convexity (per 100 face), annualized in Jahren^2.
    """
    c = coupon_rate / 100 * face / freq
    r = ytm / freq
    n = int(round(years * freq))
    if n <= 0:
        return 0.0
    conv = 0.0
    price = 0.0
    for k in range(1, n + 1):
        cf = c if k < n else (c + face)
        df = (1 + r) ** k
        price += cf / df
        conv += cf * k * (k + 1) / (df * (1 + r) ** 2)
    # period-based convexity -> annualized (divide by freq^2)
    return conv / price / (freq ** 2)

def dv01(mod_duration, dirty_price):
    """
    DV01 per 100 face in currency units per 1bp (0.01%).
    """
    return mod_duration * dirty_price * 0.0001

def effective_annual_rate(apr, m=1):
    return (1 + apr / m) ** m - 1

def realized_yield(face, coupon_rate, freq, years, clean_price, accrued_pct, reinvest_apr):
    """
    Realized Yield (annualized) bei Wiederanlage aller Kupons zum reinvest_apr (APR).
    - Preisbasis: Dirty (clean + accrued).
    - Kupon-Frequenz freq (1,2,4).
    """
    c = coupon_rate / 100 * face / freq
    n = int(round(years * freq))
    pr_dirty = clean_price + accrued_pct/100 * face

    r_re = reinvest_apr / freq
    # Accumulate coupons with reinvestment to maturity
    fv_coupons = 0.0
    for k in range(1, n + 1):
        periods_to_maturity = n - k
        fv_coupons += c * ((1 + r_re) ** periods_to_maturity)
    fv_total = fv_coupons + face  # redemption at maturity
    # Annualized realized yield over 'years'
    if years <= 0:
        return np.nan
    r_realized = (fv_total / pr_dirty) ** (1 / years) - 1
    return r_realized

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Bond Lab — LITE (Pro)", layout="wide")
st.title("Bond Lab — LITE (Pro)")
st.caption("Pricing • Duration • Convexity • DV01 • Realized Yield (Reinvestment @ Geldmarktzins)")

col1, col2, col3 = st.columns(3)
with col1:
    face = st.number_input("Nominal (Face Value)", 50.0, 10000.0, 100.0, step=50.0)
    clean_price = st.number_input("Clean Price (per 100)", 0.0, 200.0, 20.00, step=0.10)
    accrued_pct = st.number_input("Accrued (Stückzinsen, % von 100)", 0.0, 10.0, 0.3056, step=0.0001)

with col2:
    coupon_rate = st.number_input("Kupon p.a. (%)", 0.0, 100.0, 0.0, step=0.25)
    years = st.number_input("Restlaufzeit (Jahre)", 0.0, 50.0, 3.0, step=0.25)
    freq = st.selectbox("Kuponfrequenz", options=[1,2,4], index=0, help="1=jhrl, 2=halbj., 4=quartal")

with col3:
    reinvest_apr_input = st.number_input("Reinvestitionszins (APR, Geldmarkt, %)", -50.0, 200.0, 5.0, step=0.25)
    show_sensitivity = st.checkbox("Sensitivität der Realized Yield vs. Reinvest-Zins anzeigen", True)
    reinv_min = st.number_input("Sensitivität: min Reinvest-Zins (%)", -10.0, 200.0, 0.0, step=0.5)
    reinv_max = st.number_input("Sensitivität: max Reinvest-Zins (%)", -10.0, 500.0, 30.0, step=0.5)

dirty_price = clean_price + accrued_pct/100 * face

st.markdown("---")

# -----------------------------
# Core calculations
# -----------------------------
if years > 0:
    # YTM from clean price (APR)
    ytm_apr = ytm_from_price(face, coupon_rate, freq, years, clean_price)  # APR
    ytm_ear = effective_annual_rate(ytm_apr, m=1)

    # Durations
    D_mac, model_clean_chk = macaulay_duration(face, coupon_rate, freq, years, ytm_apr)
    D_mod = modified_duration_from_macaulay(D_mac, ytm_apr, freq)

    # Convexity & DV01
    conv = convexity(face, coupon_rate, freq, years, ytm_apr)
    dv01_val = dv01(D_mod, dirty_price)

    # Realized yield (APR) at money-market reinvestment
    r_realized_apr = realized_yield(face, coupon_rate, freq, years, clean_price, accrued_pct, reinvest_apr_input/100.0)
    r_realized_ear = effective_annual_rate(r_realized_apr, m=1)

    # -----------------------------
    # Display headline metrics
    # -----------------------------
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("YTM (APR)", f"{ytm_apr*100:,.2f} %")
        st.metric("YTM (EAR)", f"{ytm_ear*100:,.2f} %")
        st.metric("Accrued (pro 100)", f"{accrued_pct:.4f} %")
    with m2:
        st.metric("Model Clean (per 100)", f"{model_clean_chk:,.2f}")
        st.metric("Model Dirty (per 100)", f"{model_clean_chk + accrued_pct/100*face:,.2f}")
        st.metric("DV01 (per 100)", f"{dv01_val:,.4f}")
    with m3:
        st.metric("Modified Duration", f"{D_mod:,.3f}")
        st.metric("Macaulay Duration (J)", f"{D_mac:,.3f}")
        st.metric("Convexity", f"{conv:,.3f}")

    st.subheader("Realized Yield (Wiederanlage zum Geldmarktzins)")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Reinvestitionszins (APR)", f"{reinvest_apr_input:,.2f} %")
        st.metric("Realized Yield (APR)", f"{r_realized_apr*100:,.2f} %")
    with c2:
        st.metric("Dirty Price (per 100)", f"{dirty_price:,.2f}")
        st.metric("Realized Yield (EAR)", f"{r_realized_ear*100:,.2f} %")

    st.caption("Hinweis: Realized Yield setzt die Wiederanlage jedes Kupons zum angegebenen Geldmarktzins voraus. Liegt dieser deutlich unter der YTM, fällt die effektive Rendite entsprechend niedriger aus.")

    # -----------------------------
    # Sensitivity plot/table
    # -----------------------------
    if show_sensitivity:
        grid = np.linspace(reinv_min/100.0, reinv_max/100.0, 61)
        ry_apr = [realized_yield(face, coupon_rate, freq, years, clean_price, accrued_pct, z) for z in grid]
        df = pd.DataFrame({
            "Reinvestitionszins (APR, %)": grid * 100.0,
            "Realized Yield (APR, %)": np.array(ry_apr) * 100.0
        })
        st.dataframe(df.style.format({"Reinvestitionszins (APR, %)":"{:.2f}", "Realized Yield (APR, %)":"{:.2f}"}), use_container_width=True)

        fig = plt.figure(figsize=(7,4))
        plt.plot(df["Reinvestitionszins (APR, %)"], df["Realized Yield (APR, %)"])
        plt.xlabel("Reinvestitionszins (APR, %)")
        plt.ylabel("Realized Yield (APR, %)")
        plt.title("Sensitivität: Realized Yield vs. Reinvestitionszins")
        st.pyplot(fig)

else:
    st.warning("Bitte eine Restlaufzeit > 0 angeben.")
