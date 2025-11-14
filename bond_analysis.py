# bond_lab_lite_old_style.py
# Klassischer Stil – Bond Pricing & Duration Tool inkl. Realized Yield bei Geldmarktzins

import streamlit as st
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

# -----------------------------
# Funktionen
# -----------------------------
def price_from_ytm(face, coupon_rate, freq, years, ytm):
    c = coupon_rate / 100 * face / freq
    r = ytm / freq
    n = int(round(years * freq))
    pv = sum([c / ((1 + r) ** t) for t in range(1, n + 1)]) + face / ((1 + r) ** n)
    return pv

def ytm_from_price(face, coupon_rate, freq, years, price, tol=1e-10, max_iter=200):
    low, high = -0.999, 10.0
    for _ in range(max_iter):
        mid = (low + high) / 2
        p = price_from_ytm(face, coupon_rate, freq, years, mid)
        if abs(p - price) < tol:
            return mid
        if p > price:
            low = mid
        else:
            high = mid
    return mid

def macaulay_duration(face, coupon_rate, freq, years, ytm):
    c = coupon_rate / 100 * face / freq
    r = ytm / freq
    n = int(round(years * freq))
    times = np.arange(1, n + 1)
    cashflows = np.array([c] * (n - 1) + [c + face])
    pv = cashflows / ((1 + r) ** times)
    price = pv.sum()
    w = pv / price
    D_mac = np.sum(w * times / freq)
    return D_mac, price

def modified_duration(D_mac, ytm, freq):
    return D_mac / (1 + ytm / freq)

def convexity(face, coupon_rate, freq, years, ytm):
    c = coupon_rate / 100 * face / freq
    r = ytm / freq
    n = int(round(years * freq))
    conv = 0.0
    price = 0.0
    for k in range(1, n + 1):
        cf = c if k < n else (c + face)
        df = (1 + r) ** k
        price += cf / df
        conv += cf * k * (k + 1) / (df * (1 + r) ** 2)
    return conv / price / (freq ** 2)

def dv01(D_mod, price):
    return D_mod * price * 0.0001

def effective_annual_rate(apr, m=1):
    return (1 + apr / m) ** m - 1

def realized_yield(face, coupon_rate, freq, years, clean_price, accrued_pct, reinvest_apr):
    c = coupon_rate / 100 * face / freq
    n = int(round(years * freq))
    dirty = clean_price + accrued_pct/100 * face
    r_re = reinvest_apr / freq
    fv_coupons = sum([c * ((1 + r_re) ** (n - t)) for t in range(1, n + 1)])
    fv_total = fv_coupons + face
    return (fv_total / dirty) ** (1 / years) - 1

# -----------------------------
# Streamlit Oberfläche
# -----------------------------
st.set_page_config(page_title="Bond Lab — Classic", layout="centered")
st.title("Bond Lab — Classic Version")
st.write("Berechnet YTM, Duration, Convexity, DV01 und Realized Yield bei Wiederanlage zum Geldmarktzins.")

st.sidebar.header("Eingabeparameter")

face = st.sidebar.number_input("Nominalwert (Face)", 50.0, 10000.0, 100.0, step=50.0)
clean_price = st.sidebar.number_input("Clean Price (per 100)", 0.0, 200.0, 20.00, step=0.10)
accrued = st.sidebar.number_input("Accrued (% von 100)", 0.0, 10.0, 0.3056, step=0.0001)
coupon = st.sidebar.number_input("Kupon p.a. (%)", 0.0, 100.0, 0.0, step=0.25)
years = st.sidebar.number_input("Restlaufzeit (Jahre)", 0.0, 50.0, 3.0, step=0.25)
freq = st.sidebar.selectbox("Kuponfrequenz", [1, 2, 4], index=0)
reinvest_rate = st.sidebar.number_input("Reinvestitionszins (APR, %)", -50.0, 200.0, 5.0, step=0.25)

dirty_price = clean_price + accrued / 100 * face

if years > 0:
    ytm_apr = ytm_from_price(face, coupon, freq, years, clean_price)
    ytm_ear = effective_annual_rate(ytm_apr, 1)
    D_mac, model_clean = macaulay_duration(face, coupon, freq, years, ytm_apr)
    D_mod = modified_duration(D_mac, ytm_apr, freq)
    conv = convexity(face, coupon, freq, years, ytm_apr)
    dv01_val = dv01(D_mod, dirty_price)
    r_real_apr = realized_yield(face, coupon, freq, years, clean_price, accrued, reinvest_rate/100)
    r_real_ear = effective_annual_rate(r_real_apr, 1)

    st.subheader("Ergebnisse")
    st.write(f"**YTM (APR):** {ytm_apr*100:,.2f} %")
    st.write(f"**YTM (EAR):** {ytm_ear*100:,.2f} %")
    st.write(f"**Accrued (pro 100):** {accrued:.4f} %")
    st.write(f"**Model Clean / Dirty:** {model_clean:,.2f} / {model_clean + accrued/100*face:,.2f}")
    st.write(f"**Macaulay Duration:** {D_mac:,.3f} Jahre")
    st.write(f"**Modified Duration:** {D_mod:,.3f}")
    st.write(f"**Convexity:** {conv:,.3f}")
    st.write(f"**DV01 (pro 100):** {dv01_val:,.4f}")
    st.write("---")
    st.write(f"**Reinvestitionszins (APR):** {reinvest_rate:,.2f} %")
    st.write(f"**Realized Yield (APR):** {r_real_apr*100:,.2f} %")
    st.write(f"**Realized Yield (EAR):** {r_real_ear*100:,.2f} %")

    st.write("---")
    st.caption("Hinweis: Realized Yield berücksichtigt Wiederanlage der Kupons zum angegebenen Geldmarktzins. "
               "Liegt der Geldmarktzins unter der YTM, fällt die reale Rendite entsprechend niedriger aus.")

    # Sensitivität
    st.subheader("Sensitivität: Realized Yield vs. Reinvest-Zins")
    rates = np.linspace(0, 30, 61)
    ry_apr = [realized_yield(face, coupon, freq, years, clean_price, accrued, r/100) for r in rates]
    df = pd.DataFrame({"Reinvest-Zins (%)": rates, "Realized Yield (APR, %)": np.array(ry_apr)*100})
    st.line_chart(df.set_index("Reinvest-Zins (%)"))

else:
    st.warning("Bitte Restlaufzeit > 0 Jahre eingeben.")
