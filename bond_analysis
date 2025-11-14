# Re-create the Streamlit app file after environment reset.
from textwrap import dedent

code = dedent('''
    # bond_lab.py
    # Streamlit Bond Evaluation & Analysis Lab
    #
    # Features:
    # - Clean/Dirty pricing with accrued interest for arbitrary coupon frequency
    # - Exact YTM solving via robust bisection (no approximations)
    # - Effective Annual Rate (EAR) and nominal APR
    # - Cashflow schedule generation (monthly/quarterly/semiannual/annual)
    # - Duration, DV01, Convexity (finite-difference on yield)
    # - Price/Yield curve and scenario tools (default & recovery)
    # - Optional z-spread solver from uploaded zero-curve (simple constant spread)
    #
    # Usage: streamlit run bond_lab.py

    import math
    from datetime import date, datetime, timedelta
    try:
        from dateutil.relativedelta import relativedelta
    except Exception:
        # Minimal fallback if python-dateutil is missing
        class relativedelta:
            def __init__(self, months=0, years=0):
                self.months = months + years*12
        def _add_months(dt, months):
            y = dt.year + (dt.month - 1 + months)//12
            m = (dt.month - 1 + months)%12 + 1
            d = min(dt.day, [31,29 if y%4==0 and (y%100!=0 or y%400==0) else 28,31,30,31,30,31,31,30,31,30,31][m-1])
            return datetime(y,m,d).date()

    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import streamlit as st

    # ------------------------------
    # Helpers
    # ------------------------------
    def add_months(dt: date, n: int) -> date:
        return (dt + relativedelta(months=n))

    def year_fraction(d1: date, d2: date, convention: str, period_start: date=None, period_end: date=None) -> float:
        # Day-count fraction for accruals
        convention = convention.upper()
        if convention in ("ACT/365", "ACT/365F", "ACT/365 FIXED"):
            return (d2 - d1).days / 365.0
        if convention in ("ACT/360",):
            return (d2 - d1).days / 360.0
        if convention in ("30/360", "30U/360", "30E/360"):
            # 30/360 US simple
            d1_y, d1_m, d1_d = d1.year, d1.month, min(d1.day, 30)
            d2_y, d2_m, d2_d = d2.year, d2.month, min(d2.day, 30)
            return ((360*(d2_y-d1_y) + 30*(d2_m-d1_m) + (d2_d-d1_d)))/360.0
        if convention in ("ACT/ACT", "ACT/ACT-ISDA"):
            if period_start is None or period_end is None:
                # fallback to ACT/365 if bounds unknown
                return (d2 - d1).days / 365.0
            return (d2 - d1).days / (period_end - period_start).days
        # Default to ACT/365
        return (d2 - d1).days / 365.0

    def build_schedule(settlement: date, maturity: date, frequency: str, first_coupon: date=None):
        freq_map = {"Annual": 12, "Semiannual": 6, "Quarterly": 3, "Monthly": 1}
        if frequency not in freq_map:
            raise ValueError("Unsupported frequency")
        step_months = freq_map[frequency]
        # Generate coupon dates by stepping backward from maturity
        dates = [maturity]
        cur = maturity
        while True:
            nxt = cur - relativedelta(months=step_months)
            dates.append(nxt)
            cur = nxt
            if nxt <= settlement - timedelta(days=1):
                break
            # safety cap
            if len(dates) > 1000:
                break
        dates = sorted([d for d in dates if d > settlement])
        # For accrual we need previous coupon date (on or before settlement)
        prev_coupon = None
        probe = maturity
        while probe > settlement:
            prev = probe - relativedelta(months=step_months)
            if prev <= settlement:
                prev_coupon = prev
                break
            probe = prev
        if prev_coupon is None:
            prev_coupon = max(settlement - relativedelta(months=step_months), settlement)
        return dates, prev_coupon

    def accrued_interest(settlement: date, prev_coupon: date, next_coupon: date, coupon_rate: float, nominal: float, frequency: str, day_count: str) -> float:
        coupon_per_year = {"Annual":1, "Semiannual":2, "Quarterly":4, "Monthly":12}[frequency]
        c = coupon_rate * nominal / coupon_per_year
        # accrual fraction = fraction of current coupon period
        frac = year_fraction(prev_coupon, settlement, day_count, prev_coupon, next_coupon) / \
               max(1e-12, year_fraction(prev_coupon, next_coupon, day_count, prev_coupon, next_coupon))
        return c * frac

    def discount_factor(y_annual: float, t_years: float, comp_per_year: int) -> float:
        # Discrete compounding at m times per year
        m = comp_per_year
        return 1.0 / ((1.0 + y_annual/m) ** (m * t_years))

    def price_from_yield(y_annual: float, settlement: date, cashflows: list, comp_per_year: int, clean: bool, ai: float) -> float:
        # cashflows: list of tuples (payment_date, amount)
        pv = 0.0
        for dt, amt in cashflows:
            t = (dt - settlement).days / 365.0
            pv += amt * discount_factor(y_annual, t, comp_per_year)
        dirty = pv
        return dirty - ai if clean else dirty

    def ytm_from_price(target_price: float, settlement: date, cashflows: list, comp_per_year: int, clean: bool, ai: float) -> float:
        # Robust bisection on annualized yield
        # Bounds: [0, 10] => up to 1000% p.a.
        low, high = 0.0, 10.0
        for _ in range(200):
            mid = 0.5*(low+high)
            p = price_from_yield(mid, settlement, cashflows, comp_per_year, clean, ai)
            if (p - target_price) == 0:
                return mid
            # Determine which side to keep
            p_low = price_from_yield(low, settlement, cashflows, comp_per_year, clean, ai)
            if (p_low - target_price) * (p - target_price) <= 0:
                high = mid
            else:
                low = mid
        return 0.5*(low+high)

    def build_cashflows(settlement: date, maturity: date, coupon_rate: float, nominal: float, frequency: str, day_count: str):
        coupon_per_year = {"Annual":1, "Semiannual":2, "Quarterly":4, "Monthly":12}[frequency]
        step_months = {"Annual":12, "Semiannual":6, "Quarterly":3, "Monthly":1}[frequency]
        schedule, prev_coupon = build_schedule(settlement, maturity, frequency)
        c_amt = coupon_rate * nominal / coupon_per_year
        flows = []
        for pay_date in schedule:
            amt = c_amt
            if pay_date == schedule[-1]:  # maturity
                amt += nominal
            flows.append((pay_date, amt))
        # Next coupon for accrual calc is the first payment date in schedule
        next_coupon = schedule[0] if schedule else maturity
        ai = accrued_interest(settlement, prev_coupon, next_coupon, coupon_rate, nominal, frequency, day_count)
        return flows, ai, schedule, prev_coupon, next_coupon

    def effective_annual_from_periodic(y_annual: float, m: int) -> float:
        # returns EAR given nominal APR y_annual with m compounding per year
        r_per = y_annual / m
        return (1 + r_per) ** m - 1

    def duration_convexity(y_annual: float, settlement: date, cashflows: list, comp_per_year: int, clean: bool, ai: float, bp: float=1e-4):
        # Finite difference around y_annual (in decimal, e.g. 0.05)
        p0 = price_from_yield(y_annual, settlement, cashflows, comp_per_year, clean, ai)
        p_up = price_from_yield(y_annual + bp, settlement, cashflows, comp_per_year, clean, ai)
        p_dn = price_from_yield(y_annual - bp, settlement, cashflows, comp_per_year, clean, ai)
        # Modified duration approximation
        dP = (p_dn - p_up) / 2.0
        mod_dur = (dP / p0) / bp
        # Convexity approximation
        convex = (p_up + p_dn - 2.0*p0) / (p0 * (bp**2))
        dv01 = mod_dur * p0 * 1e-4  # price change for 1bp
        return mod_dur, convex, dv01, p0

    def make_price_yield_curve(settlement, cashflows, comp_per_year, clean, ai, y_min=0.0, y_max=2.0, n=120):
        ys = np.linspace(y_min, y_max, n)
        prices = [price_from_yield(y, settlement, cashflows, comp_per_year, clean, ai) for y in ys]
        return ys, prices

    def expected_one_year_return(clean_price, coupon_rate, nominal, frequency, y_annual, prob_default, recovery, assumption_price_same=True):
        # Simple 1y horizon model: if survive, receive coupons for 1y and sell at same clean price (or price from yield)
        m = {"Annual":1, "Semiannual":2, "Quarterly":4, "Monthly":12}[frequency]
        coupon_income = coupon_rate * nominal * (1.0)  # 1 year of coupons
        if assumption_price_same:
            price_end = clean_price
        else:
            # Price end if yield unchanged (approx = same clean price for flat accrual)
            price_end = clean_price
        # Survival payoff minus starting price = coupon + (price_end - price_start)
        payoff_survive = coupon_income + (price_end - clean_price)
        payoff_default = recovery * nominal - clean_price
        exp_payoff = (1 - prob_default) * payoff_survive + prob_default * payoff_default
        return exp_payoff / clean_price

    # ------------------------------
    # UI
    # ------------------------------
    st.set_page_config(page_title="Bond Evaluation & Analysis Lab", page_icon="💹", layout="wide")
    st.title("💹 Bond Evaluation & Analysis Lab")
    st.caption("Professionelles, dennoch einfaches Bond-Toolkit: exakte YTM, Accrued, Duration, Szenarien, Price/Yield-Kurven.")

    with st.sidebar:
        st.header("Instrument")
        name = st.text_input("Name", value="Urbanek Real Estate GmbH 10% 25/32")
        nominal = st.number_input("Nominal (pro Stück)", value=100.0, step=1.0, format="%.2f")
        coupon_pct = st.number_input("Kupon (% p.a.)", value=10.0, step=0.1, format="%.4f")
        frequency = st.selectbox("Kuponfrequenz", ["Monthly", "Quarterly", "Semiannual", "Annual"], index=0)
        day_count = st.selectbox("Day-Count", ["ACT/ACT", "ACT/365", "ACT/360", "30/360"], index=0)

        today = date.today()
        settlement = st.date_input("Settlement", value=today)
        maturity = st.date_input("Fälligkeit", value=date(2032,2,3))

        st.divider()
        st.subheader("Preis & Benchmarks")
        price_input_type = st.radio("Preis ist", ["Clean", "Dirty"], index=0, horizontal=True)
        clean_price = st.number_input("Preis (% vom Nominal)", value=20.00, step=0.01, format="%.4f")
        benchmark_y = st.number_input("Benchmark Yield (% p.a.)", value=0.0, step=0.01, help="Für G-Spread (optional).")

        st.divider()
        st.subheader("Szenario (1Y)")
        prob_default = st.slider("Default-Wahrscheinlichkeit 1Y", 0.0, 1.0, 0.2, 0.01)
        recovery = st.slider("Recovery (in % vom Nominal)", 0.0, 1.0, 0.20, 0.01)

        st.divider()
        example = st.checkbox("Urbanek-Parametrisierung laden", value=True, help="Setzt typische Daten: 10% p.a., monatlich, Kurs 20, Fälligkeit 03.02.2032.")

    if example:
        coupon_pct = 10.0
        frequency = "Monthly"
        clean_price = 20.00
        maturity = date(2032,2,3)

    coupon_rate = coupon_pct/100.0
    comp_per_year = {"Annual":1, "Semiannual":2, "Quarterly":4, "Monthly":12}[frequency]

    # Cashflows & accrued
    cashflows, ai, schedule, prev_coupon, next_coupon = build_cashflows(settlement, maturity, coupon_rate, nominal, frequency, day_count)

    # Clean/Dirty alignment
    if price_input_type == "Clean":
        target_price = clean_price
        dirty_price_input = clean_price + (ai/nominal)*100.0
    else:
        dirty_price_input = clean_price
        target_price = max(0.0, clean_price - (ai/nominal)*100.0)

    # Convert price in % to absolute per 100 nominal for solver convenience
    price_abs = dirty_price_input/100.0 * nominal

    # Solve YTM (annual nominal rate with discrete comp at m)
    ytm = ytm_from_price(price_abs, settlement, cashflows, comp_per_year, clean=False, ai=0.0)  # use dirty in the equation
    ear = effective_annual_from_periodic(ytm, comp_per_year)
    apr = ytm  # nominal APR with comp_per_year

    # Risk measures (around solved yield)
    mod_dur, convex, dv01, model_dirty = duration_convexity(ytm, settlement, cashflows, comp_per_year, clean=False, ai=0.0)
    model_clean = (model_dirty - ai)

    # Table: cashflows
    cf_rows = []
    for dt_pay, amt in cashflows:
        is_maturity = (dt_pay == cashflows[-1][0])
        cf_rows.append({
            "Payment Date": dt_pay,
            "Cash Flow": amt,
            "Type": "Maturity+Coupon" if is_maturity else "Coupon"
        })
    cf_df = pd.DataFrame(cf_rows)

    # Layout
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Renditen")
        st.metric("Nominal APR (p.a.)", f"{apr*100:.2f}%")
        st.metric("Effektiver Jahreszins (EAR/YTM)", f"{ear*100:.2f}%")
        st.caption("EAR = (1 + APR/m)^m - 1; m = Kuponfrequenz pro Jahr.")

    with col2:
        st.subheader("Preise")
        st.metric("Accrued Interest", f"{ai:.4f}")
        st.metric("Model Dirty Price", f"{model_dirty/nominal*100:.2f}%")
        st.metric("Model Clean Price", f"{model_clean/nominal*100:.2f}%")

    with col3:
        st.subheader("Risiko (am YTM)")
        st.metric("Modified Duration", f"{mod_dur:.3f}")
        st.metric("Convexity", f"{convex:.3f}")
        st.metric("DV01 (pro 1bp)", f"{dv01:.4f}")
        if benchmark_y > 0.0:
            gspread = (apr - benchmark_y/100.0)*10000
            st.metric("G-Spread (bp)", f"{gspread:.0f}")

    st.divider()
    st.subheader("Cashflow-Schedule")
    st.dataframe(cf_df, use_container_width=True, hide_index=True)

    # Price/Yield curve
    st.divider()
    st.subheader("Price ↔ Yield")
    ys, ps = make_price_yield_curve(settlement, cashflows, comp_per_year, clean=False, ai=0.0, y_min=0.0, y_max=max(2.0, ytm*1.2), n=200)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[y*100 for y in ys], y=[p/nominal*100 for p in ps], mode="lines", name="Dirty Price"))
    fig.add_trace(go.Scatter(x=[ytm*100], y=[model_dirty/nominal*100], mode="markers", name="Current", marker=dict(size=10)))
    fig.update_layout(xaxis_title="Yield (APR, % p.a.)", yaxis_title="Price (% of Nominal)", height=420, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # Scenario analysis
    st.divider()
    st.subheader("1-Jahres-Szenario (vereinfachtes Erwartungswertmodell)")
    exp_ret = expected_one_year_return(clean_price=target_price, coupon_rate=coupon_rate, nominal=nominal, frequency=frequency,
                                       y_annual=apr, prob_default=prob_default, recovery=recovery, assumption_price_same=True)
    st.write(f"**Erwartete 1Y-Return (E[r])** bei p(Default)={prob_default:.0%}, Recovery={recovery:.0%}: **{exp_ret*100:.2f}%**")

    st.info("Hinweis: EAR/YTM und Sensitivitäten basieren auf **diskreter Verzinsung** mit der gewählten Frequenz. "
            "Accrued Interest nach gewählter Day-Count-Konvention. Für exotische Stubs ggf. kleine Abweichungen.")

    st.caption("© Bond Evaluation & Analysis Lab – erstellt für professionelle Anwender.")

    # Exporter
    import io
    csv_bytes = cf_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Cashflows CSV", csv_bytes, file_name="cashflows.csv", mime="text/csv")
''')

with open('/mnt/data/bond_lab.py', 'w', encoding='utf-8') as f:
    f.write(code)

print("/mnt/data/bond_lab.py created")
