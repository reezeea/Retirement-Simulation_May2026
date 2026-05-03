"""
Singapore Retirement Simulator – Main Streamlit Application
Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.simulation.profile import UserProfile, OneOffExpense, RetirementIncome
from src.simulation.assets import Asset, AssetType, Liquidity, Portfolio, ASSET_DEFAULTS
from src.simulation.monte_carlo import run_monte_carlo, SimulationResult
from src.cpf.cpf_model import CPFAccounts, simulate_cpf_trajectory
from src.utils.presets import PRESETS
from src.utils.export import export_csv, export_summary_text

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SG Retirement Simulator",
    page_icon="🇸🇬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Styling ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}
.stApp {
    background: #0f1117;
    color: #e8eaf0;
}
h1, h2, h3 {
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: -0.03em;
}
.metric-card {
    background: #1a1d27;
    border: 1px solid #2e3248;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.5rem;
}
.metric-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #6b7280;
    font-family: 'IBM Plex Mono', monospace;
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    font-family: 'IBM Plex Mono', monospace;
    margin-top: 0.2rem;
}
.success-green { color: #10d48e; }
.warning-yellow { color: #f59e0b; }
.danger-red { color: #ef4444; }
.tag {
    display: inline-block;
    background: #1e2235;
    border: 1px solid #3b4263;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.7rem;
    font-family: 'IBM Plex Mono', monospace;
    color: #8892c0;
    margin-right: 4px;
}
.section-header {
    border-left: 3px solid #4f6ef7;
    padding-left: 0.75rem;
    margin: 1.5rem 0 1rem 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #a0a8cc;
}
.insight-box {
    background: #161a2b;
    border: 1px solid #2a3158;
    border-left: 4px solid #4f6ef7;
    border-radius: 6px;
    padding: 1rem 1.25rem;
    margin: 0.5rem 0;
    font-size: 0.92rem;
}
.stButton>button {
    background: #4f6ef7;
    color: white;
    border: none;
    border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    padding: 0.6rem 1.5rem;
    transition: all 0.2s;
}
.stButton>button:hover {
    background: #6b86ff;
    transform: translateY(-1px);
}
</style>
""", unsafe_allow_html=True)


# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 2rem 0 1rem 0;">
    <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; color: #4f6ef7; letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 0.5rem;">🇸🇬 Singapore Financial Planning</div>
    <h1 style="margin: 0; font-size: 2.2rem; color: #e8eaf0;">Retirement Simulator</h1>
    <p style="color: #6b7280; margin-top: 0.5rem; font-size: 0.95rem;">Monte Carlo simulation engine tailored for Singapore's CPF system, inflation, and cost of living.</p>
</div>
""", unsafe_allow_html=True)

# ─── Preset Loader ────────────────────────────────────────────────────────────
with st.expander("⚡ Load a Preset Profile", expanded=False):
    col_p1, col_p2 = st.columns([3, 1])
    with col_p1:
        preset_choice = st.selectbox(
            "Choose a preset",
            options=["— Custom (start blank) —"] + list(PRESETS.keys()),
            label_visibility="collapsed",
        )
    with col_p2:
        load_preset = st.button("Load Preset", use_container_width=True)

    st.caption("Presets pre-fill all fields. You can still edit any value after loading.")

if "profile" not in st.session_state:
    st.session_state.profile = None
if "result" not in st.session_state:
    st.session_state.result = None

if load_preset and preset_choice != "— Custom (start blank) —":
    st.session_state.profile = PRESETS[preset_choice]()
    st.session_state.result = None
    st.success(f"✅ Loaded preset: **{preset_choice}**")

p = st.session_state.profile  # may be None (custom mode)


def fv(key, default):
    """Get field value from preset profile or return default."""
    if p is not None:
        return getattr(p, key, default)
    return default


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR: SIMULATION SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Simulation Settings")
    n_sims = st.select_slider(
        "Monte Carlo Runs",
        options=[1000, 2000, 5000, 10000],
        value=fv("n_simulations", 5000),
    )
    use_fat_tails = st.toggle("Fat-Tail Returns (Student-t)", value=fv("use_fat_tails", True),
                               help="Uses Student-t distribution for heavier tails vs Normal distribution.")
    st.divider()
    st.markdown("### 🧠 Mode")
    sim_mode = st.radio(
        "Simulation Mode",
        ["Standard", "DieWithZero", "Legacy"],
        index=["Standard", "DieWithZero", "Legacy"].index(fv("simulation_mode", "Standard")),
        help="Standard: maximize longevity. Die With Zero: spend down to 0. Legacy: preserve a target amount.",
    )
    legacy_target = 0.0
    if sim_mode == "Legacy":
        legacy_target = st.number_input("Legacy Target (SGD)", value=float(fv("legacy_target", 500_000)), step=50_000.0, format="%.0f")

    st.divider()
    st.markdown("### 💸 Withdrawal Strategy")
    w_strategy = st.radio(
        "Strategy",
        ["Fixed", "PercentPortfolio", "Guardrail"],
        index=["Fixed", "PercentPortfolio", "Guardrail"].index(fv("withdrawal_strategy", "Fixed")),
    )
    w_pct = 4.0
    g_upper, g_lower = 5.5, 4.0
    if w_strategy == "PercentPortfolio":
        w_pct = st.slider("Withdrawal %", 2.0, 8.0, float(fv("withdrawal_percent", 4.0)), 0.1)
    elif w_strategy == "Guardrail":
        g_upper = st.slider("Upper Guardrail %", 4.0, 8.0, float(fv("guardrail_upper", 5.5)), 0.1)
        g_lower = st.slider("Lower Guardrail %", 2.0, 5.0, float(fv("guardrail_lower", 4.0)), 0.1)

    st.divider()
    st.markdown("### 🇸🇬 CPF Settings")
    cpf_rs = st.selectbox("Retirement Sum", ["BRS", "FRS", "ERS"], index=["BRS", "FRS", "ERS"].index(fv("cpf_retirement_sum", "FRS")))
    cpflife_plan = st.selectbox("CPF LIFE Plan", ["Basic", "Standard", "Escalating"], index=["Basic", "Standard", "Escalating"].index(fv("cpflife_payout_plan", "Standard")))
    cpflife_age = st.number_input("CPF LIFE Start Age", min_value=65, max_value=70, value=int(fv("cpflife_payout_age", 65)))


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ═══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "👤 Profile",
    "💰 Income & Expenses",
    "🏦 Assets & CPF",
    "📋 Liabilities",
    "📅 Events & Cashflows",
    "📊 Results",
    "📈 Analysis",
])

tab_profile, tab_income, tab_assets, tab_liabilities, tab_events, tab_results, tab_analysis = tabs


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: PERSONAL PROFILE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_profile:
    st.markdown('<div class="section-header">Personal Profile</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        current_age = st.number_input("Current Age", min_value=18, max_value=80, value=int(fv("current_age", 35)), step=1)
    with c2:
        retirement_age = st.number_input("Desired Retirement Age", min_value=current_age + 1, max_value=85, value=int(fv("retirement_age", 65)), step=1)
    with c3:
        lifespan = st.number_input("Expected Lifespan", min_value=retirement_age + 1, max_value=110, value=int(fv("lifespan", 85)), step=1,
                                    help="Singaporean life expectancy ~84M / 88F. Use 90–95 for conservative planning.")
    with c4:
        gender = st.selectbox("Gender", ["Male", "Female", "Prefer not to say"], index=["Male", "Female", "Prefer not to say"].index(fv("gender", "Male")))

    c5, c6, c7 = st.columns(3)
    with c5:
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"], index=["Single", "Married", "Divorced"].index(fv("marital_status", "Single")))
    with c6:
        num_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=int(fv("num_dependents", 0)), step=1,
                                          help="Children, elderly parents, etc.")
    with c7:
        annual_obligations = st.number_input("Annual Financial Obligations (SGD)", min_value=0.0, value=float(fv("annual_obligations", 0.0)), step=1000.0, format="%.0f",
                                              help="Supporting parents, other dependents (annual)")

    c8, c9 = st.columns(2)
    with c8:
        expected_inheritance = st.number_input("Expected Inheritance (SGD)", min_value=0.0, value=float(fv("expected_inheritance", 0.0)), step=10_000.0, format="%.0f")
    with c9:
        inheritance_age = st.number_input("Age at Inheritance", min_value=current_age, max_value=lifespan, value=int(fv("inheritance_age", 60)), step=1)

    st.markdown('<div class="section-header">Inflation Assumption</div>', unsafe_allow_html=True)
    infl_col1, infl_col2 = st.columns(2)
    with infl_col1:
        inflation_rate = st.slider("Inflation Rate (% p.a.)", min_value=0.5, max_value=6.0, value=float(fv("inflation_rate", 2.5)), step=0.1,
                                    help="Singapore long-term CPI ~2–3%. Use 3–4% for conservative planning.")
    with infl_col2:
        st.metric("Real Purchasing Power in 30 years", f"{(1/(1+inflation_rate/100)**30)*100:.0f}%",
                  help="What SGD 1 is worth in 30 years at this inflation rate")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: INCOME & EXPENSES
# ═══════════════════════════════════════════════════════════════════════════════
with tab_income:
    st.markdown('<div class="section-header">Employment Income</div>', unsafe_allow_html=True)
    i1, i2 = st.columns(2)
    with i1:
        current_income = st.number_input("Your Annual Income (SGD)", min_value=0.0, value=float(fv("current_annual_income", 0.0)), step=1_000.0, format="%.0f")
        income_growth = st.slider("Income Growth Rate (% p.a.)", 0.0, 10.0, float(fv("income_growth_rate", 3.0)), 0.1)
    with i2:
        spouse_income = 0.0
        spouse_growth = 3.0
        if marital_status == "Married":
            spouse_income = st.number_input("Spouse Annual Income (SGD)", min_value=0.0, value=float(fv("spouse_annual_income", 0.0)), step=1_000.0, format="%.0f")
            spouse_growth = st.slider("Spouse Income Growth (% p.a.)", 0.0, 10.0, float(fv("spouse_income_growth_rate", 3.0)), 0.1)
        else:
            st.info("Add spouse income if married.")

    st.markdown('<div class="section-header">Living Expenses</div>', unsafe_allow_html=True)
    e1, e2 = st.columns(2)
    with e1:
        pre_ret_expenses = st.number_input("Annual Expenses – Pre-Retirement (SGD)", min_value=0.0, value=float(fv("annual_expenses_pre_retirement", 0.0)), step=1_000.0, format="%.0f",
                                            help="Include housing costs, food, transport, insurance, childcare, etc.")
        st.caption(f"≈ SGD {pre_ret_expenses/12:,.0f}/month")
    with e2:
        post_ret_expenses = st.number_input("Annual Expenses – Post-Retirement (SGD)", min_value=0.0, value=float(fv("annual_expenses_post_retirement", 0.0)), step=1_000.0, format="%.0f",
                                             help="Typically lower (~70–80% of pre-retirement). Healthcare costs rise.")
        st.caption(f"≈ SGD {post_ret_expenses/12:,.0f}/month")

    # Expense breakdown helper
    with st.expander("🧮 Expense Calculator (Singapore Reference)", expanded=False):
        st.caption("Typical monthly expenses for a Singapore household (2024 estimates)")
        ec_col1, ec_col2, ec_col3 = st.columns(3)
        with ec_col1:
            hdb_loan = st.number_input("HDB/Mortgage (monthly)", 0.0, value=1500.0, step=100.0)
            utilities = st.number_input("Utilities & Telco", 0.0, value=200.0, step=50.0)
            groceries = st.number_input("Groceries & Dining", 0.0, value=800.0, step=50.0)
        with ec_col2:
            transport = st.number_input("Transport (MRT/Car)", 0.0, value=300.0, step=50.0)
            insurance = st.number_input("Insurance Premiums", 0.0, value=400.0, step=50.0)
            childcare = st.number_input("Childcare / Tuition", 0.0, value=500.0 * num_dependents, step=100.0)
        with ec_col3:
            entertainment = st.number_input("Entertainment & Leisure", 0.0, value=400.0, step=50.0)
            healthcare = st.number_input("Healthcare", 0.0, value=200.0, step=50.0)
            misc = st.number_input("Miscellaneous", 0.0, value=300.0, step=50.0)
        total_est = (hdb_loan + utilities + groceries + transport + insurance + childcare + entertainment + healthcare + misc) * 12
        st.metric("Estimated Annual Expenses", f"SGD {total_est:,.0f}", f"SGD {total_est/12:,.0f}/month")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: ASSETS & CPF
# ═══════════════════════════════════════════════════════════════════════════════
with tab_assets:
    st.markdown('<div class="section-header">CPF Accounts</div>', unsafe_allow_html=True)
    cpf_col1, cpf_col2, cpf_col3 = st.columns(3)
    with cpf_col1:
        cpf_oa = st.number_input("CPF Ordinary Account (SGD)", min_value=0.0,
                                  value=float(fv("cpf_accounts", CPFAccounts()).oa), step=1_000.0, format="%.0f")
        st.caption("Interest: 2.5% p.a. | Usable for housing & investments")
    with cpf_col2:
        cpf_sa = st.number_input("CPF Special Account (SGD)", min_value=0.0,
                                  value=float(fv("cpf_accounts", CPFAccounts()).sa), step=1_000.0, format="%.0f")
        st.caption("Interest: 4.0% p.a. | Locked until retirement")
    with cpf_col3:
        cpf_ma = st.number_input("CPF MediSave Account (SGD)", min_value=0.0,
                                  value=float(fv("cpf_accounts", CPFAccounts()).ma), step=1_000.0, format="%.0f")
        st.caption("Interest: 4.0% p.a. | Healthcare use only")

    st.markdown('<div class="section-header">Investment Assets</div>', unsafe_allow_html=True)
    st.caption("Add each asset class you own. Default returns/volatility are editable.")

    if "assets_list" not in st.session_state:
        if p is not None:
            st.session_state.assets_list = [
                {
                    "name": a.name,
                    "asset_type": a.asset_type.value,
                    "value": a.current_value,
                    "return": a.expected_return,
                    "volatility": a.volatility,
                    "liquidity": a.liquidity.value,
                    "currency": a.currency,
                    "rental_yield": a.rental_yield,
                }
                for a in p.portfolio.assets
                if a.asset_type not in (AssetType.CPF_OA, AssetType.CPF_SA, AssetType.CPF_MA)
            ]
        else:
            st.session_state.assets_list = []

    asset_type_options = [at.value for at in AssetType if at not in (AssetType.CPF_OA, AssetType.CPF_SA, AssetType.CPF_MA)]

    # Add new asset
    with st.expander("➕ Add Asset", expanded=len(st.session_state.assets_list) == 0):
        na_col1, na_col2, na_col3 = st.columns(3)
        with na_col1:
            new_name = st.text_input("Asset Name", placeholder="e.g. STI ETF, My Condo")
            new_type = st.selectbox("Asset Type", asset_type_options)
        with na_col2:
            new_value = st.number_input("Current Value (SGD)", min_value=0.0, step=1_000.0, format="%.0f")
            new_currency = st.selectbox("Currency", ["SGD", "USD", "EUR", "GBP", "AUD", "JPY"])
        with na_col3:
            # Auto-fill defaults
            at_enum = AssetType(new_type)
            d = ASSET_DEFAULTS[at_enum]
            new_return = st.number_input("Expected Return (% p.a.)", value=d["return"], step=0.1, format="%.1f")
            new_vol = st.number_input("Volatility (% p.a.)", value=d["volatility"], step=0.5, format="%.1f")
        st.caption(f"💡 {d['note']}")
        new_liquidity = st.select_slider("Liquidity", ["Liquid", "Semi-Liquid", "Illiquid"], value=d["liquidity"].value)

        if st.button("Add Asset") and new_name and new_value > 0:
            st.session_state.assets_list.append({
                "name": new_name, "asset_type": new_type,
                "value": new_value, "return": new_return,
                "volatility": new_vol, "liquidity": new_liquidity,
                "currency": new_currency, "rental_yield": 0.0,
            })
            st.rerun()

    # Display current assets
    if st.session_state.assets_list:
        st.markdown("**Current Assets:**")
        for i, asset in enumerate(st.session_state.assets_list):
            col_a, col_b, col_c, col_d, col_e = st.columns([3, 2, 1.5, 1.5, 1])
            with col_a:
                st.markdown(f"**{asset['name']}** <span class='tag'>{asset['asset_type']}</span> <span class='tag'>{asset['liquidity']}</span>", unsafe_allow_html=True)
            with col_b:
                st.markdown(f"SGD **{asset['value']:,.0f}**")
            with col_c:
                st.markdown(f"📈 {asset['return']:.1f}%")
            with col_d:
                st.markdown(f"〰️ {asset['volatility']:.1f}% vol")
            with col_e:
                if st.button("🗑️", key=f"del_asset_{i}"):
                    st.session_state.assets_list.pop(i)
                    st.rerun()

        total_assets_val = sum(a["value"] for a in st.session_state.assets_list)
        st.metric("Total Investable Assets", f"SGD {total_assets_val:,.0f}")
    else:
        st.info("No assets added yet. Use the form above to add your assets.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: LIABILITIES
# ═══════════════════════════════════════════════════════════════════════════════
with tab_liabilities:
    st.markdown('<div class="section-header">Liabilities & Debts</div>', unsafe_allow_html=True)

    if "liabilities_list" not in st.session_state:
        if p is not None and p.portfolio.liabilities:
            st.session_state.liabilities_list = [
                {"name": l.name, "balance": l.balance, "rate": l.interest_rate,
                 "monthly_payment": l.monthly_payment, "tenure": l.tenure_years, "type": l.liability_type}
                for l in p.portfolio.liabilities
            ]
        else:
            st.session_state.liabilities_list = []

    with st.expander("➕ Add Liability", expanded=len(st.session_state.liabilities_list) == 0):
        ll1, ll2, ll3 = st.columns(3)
        with ll1:
            lib_name = st.text_input("Liability Name", placeholder="e.g. HDB Mortgage")
            lib_type = st.selectbox("Type", ["Mortgage", "Car Loan", "Personal Loan", "Credit Card", "Other"])
        with ll2:
            lib_balance = st.number_input("Outstanding Balance (SGD)", min_value=0.0, step=5_000.0, format="%.0f")
            lib_rate = st.number_input("Interest Rate (% p.a.)", min_value=0.0, max_value=30.0, value=2.6, step=0.1)
        with ll3:
            lib_monthly = st.number_input("Monthly Payment (SGD)", min_value=0.0, step=100.0, format="%.0f")
            lib_tenure = st.number_input("Remaining Tenure (years)", min_value=0, max_value=40, value=20)
        if st.button("Add Liability") and lib_name and lib_balance > 0:
            st.session_state.liabilities_list.append({
                "name": lib_name, "balance": lib_balance, "rate": lib_rate,
                "monthly_payment": lib_monthly, "tenure": lib_tenure, "type": lib_type,
            })
            st.rerun()

    if st.session_state.liabilities_list:
        for i, lib in enumerate(st.session_state.liabilities_list):
            lc1, lc2, lc3, lc4 = st.columns([3, 2, 2, 1])
            with lc1:
                st.markdown(f"**{lib['name']}** <span class='tag'>{lib['type']}</span>", unsafe_allow_html=True)
            with lc2:
                st.markdown(f"Balance: **SGD {lib['balance']:,.0f}**")
            with lc3:
                st.markdown(f"Rate: {lib['rate']:.1f}% | SGD {lib['monthly_payment']:,.0f}/mo")
            with lc4:
                if st.button("🗑️", key=f"del_lib_{i}"):
                    st.session_state.liabilities_list.pop(i)
                    st.rerun()
        total_liab = sum(l["balance"] for l in st.session_state.liabilities_list)
        total_monthly_liab = sum(l["monthly_payment"] for l in st.session_state.liabilities_list)
        lm1, lm2 = st.columns(2)
        with lm1:
            st.metric("Total Liabilities", f"SGD {total_liab:,.0f}")
        with lm2:
            st.metric("Total Monthly Payments", f"SGD {total_monthly_liab:,.0f}")
    else:
        st.info("No liabilities added.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5: ONE-OFF EVENTS & RETIREMENT INCOME
# ═══════════════════════════════════════════════════════════════════════════════
with tab_events:
    st.markdown('<div class="section-header">One-Off / Irregular Expenses</div>', unsafe_allow_html=True)

    if "one_off_list" not in st.session_state:
        if p is not None:
            st.session_state.one_off_list = [
                {"name": e.name, "amount": e.amount, "age": e.age_due, "inflation_adj": e.inflation_adjusted}
                for e in p.one_off_expenses
            ]
        else:
            st.session_state.one_off_list = []

    with st.expander("➕ Add One-Off Expense", expanded=False):
        oe1, oe2, oe3, oe4 = st.columns([3, 2, 1.5, 1.5])
        with oe1:
            oe_name = st.text_input("Description", placeholder="e.g. Child's University, Wedding")
        with oe2:
            oe_amount = st.number_input("Amount (SGD, today's dollars)", min_value=0.0, step=5_000.0, format="%.0f")
        with oe3:
            oe_age = st.number_input("At Age", min_value=current_age, max_value=lifespan, value=50, step=1)
        with oe4:
            oe_infl = st.checkbox("Inflation-adjust", value=True)
        if st.button("Add Expense") and oe_name and oe_amount > 0:
            st.session_state.one_off_list.append({"name": oe_name, "amount": oe_amount, "age": oe_age, "inflation_adj": oe_infl})
            st.rerun()

    if st.session_state.one_off_list:
        for i, ex in enumerate(st.session_state.one_off_list):
            ec1, ec2, ec3, ec4 = st.columns([3, 2, 1.5, 1])
            with ec1:
                st.write(f"**{ex['name']}**")
            with ec2:
                st.write(f"SGD {ex['amount']:,.0f}")
            with ec3:
                st.write(f"Age {ex['age']}")
            with ec4:
                if st.button("🗑️", key=f"del_oe_{i}"):
                    st.session_state.one_off_list.pop(i)
                    st.rerun()

    st.markdown('<div class="section-header">Post-Retirement Income Sources</div>', unsafe_allow_html=True)
    st.caption("Part-time work, rental income, annuities, dividends (beyond CPF LIFE)")

    if "ret_income_list" not in st.session_state:
        if p is not None:
            st.session_state.ret_income_list = [
                {"name": ri.name, "amount": ri.annual_amount, "start_age": ri.start_age,
                 "end_age": ri.end_age, "inflation_adj": ri.inflation_adjusted}
                for ri in p.retirement_incomes
            ]
        else:
            st.session_state.ret_income_list = []

    with st.expander("➕ Add Retirement Income", expanded=False):
        ri1, ri2, ri3, ri4, ri5 = st.columns([3, 2, 1.5, 1.5, 1.5])
        with ri1:
            ri_name = st.text_input("Income Source", placeholder="e.g. Part-time Consulting, Rental")
        with ri2:
            ri_amount = st.number_input("Annual Amount (SGD)", min_value=0.0, step=1_000.0, format="%.0f")
        with ri3:
            ri_start = st.number_input("Start Age", min_value=retirement_age, max_value=lifespan, value=retirement_age)
        with ri4:
            ri_end = st.number_input("End Age", min_value=ri_start, max_value=999, value=75)
        with ri5:
            ri_infl = st.checkbox("Inflation-adj ", value=True)
        if st.button("Add Income") and ri_name and ri_amount > 0:
            st.session_state.ret_income_list.append({"name": ri_name, "amount": ri_amount, "start_age": ri_start, "end_age": ri_end, "inflation_adj": ri_infl})
            st.rerun()

    if st.session_state.ret_income_list:
        for i, ri in enumerate(st.session_state.ret_income_list):
            rc1, rc2, rc3, rc4 = st.columns([3, 2, 2, 1])
            with rc1:
                st.write(f"**{ri['name']}**")
            with rc2:
                st.write(f"SGD {ri['amount']:,.0f}/yr")
            with rc3:
                st.write(f"Age {ri['start_age']} → {ri['end_age']}")
            with rc4:
                if st.button("🗑️", key=f"del_ri_{i}"):
                    st.session_state.ret_income_list.pop(i)
                    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD PROFILE + RUN SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════
def build_profile_from_ui() -> UserProfile:
    from src.simulation.assets import Liability

    assets_objs = []
    for a in st.session_state.get("assets_list", []):
        at = AssetType(a["asset_type"])
        assets_objs.append(Asset(
            name=a["name"], asset_type=at,
            current_value=a["value"], expected_return=a["return"],
            volatility=a["volatility"], liquidity=Liquidity(a["liquidity"]),
            currency=a.get("currency", "SGD"),
        ))

    liab_objs = []
    for l in st.session_state.get("liabilities_list", []):
        liab_objs.append(Liability(
            name=l["name"], balance=l["balance"], interest_rate=l["rate"],
            monthly_payment=l["monthly_payment"], tenure_years=l["tenure"],
            liability_type=l["type"],
        ))

    one_off_objs = [
        OneOffExpense(name=e["name"], amount=e["amount"], age_due=e["age"], inflation_adjusted=e["inflation_adj"])
        for e in st.session_state.get("one_off_list", [])
    ]

    ret_income_objs = [
        RetirementIncome(name=ri["name"], annual_amount=ri["amount"],
                         start_age=ri["start_age"], end_age=ri["end_age"],
                         inflation_adjusted=ri["inflation_adj"])
        for ri in st.session_state.get("ret_income_list", [])
    ]

    return UserProfile(
        current_age=current_age,
        retirement_age=retirement_age,
        lifespan=lifespan,
        gender=gender,
        marital_status=marital_status,
        num_dependents=num_dependents,
        annual_obligations=annual_obligations,
        expected_inheritance=expected_inheritance,
        inheritance_age=inheritance_age,
        current_annual_income=current_income,
        income_growth_rate=income_growth,
        spouse_annual_income=spouse_income,
        spouse_income_growth_rate=spouse_growth,
        annual_expenses_pre_retirement=pre_ret_expenses,
        annual_expenses_post_retirement=post_ret_expenses,
        inflation_rate=inflation_rate,
        one_off_expenses=one_off_objs,
        retirement_incomes=ret_income_objs,
        portfolio=Portfolio(assets=assets_objs, liabilities=liab_objs),
        cpf_accounts=CPFAccounts(oa=cpf_oa, sa=cpf_sa, ma=cpf_ma),
        cpf_retirement_sum=cpf_rs,
        cpflife_payout_plan=cpflife_plan,
        cpflife_payout_age=cpflife_age,
        withdrawal_strategy=w_strategy,
        withdrawal_percent=w_pct,
        guardrail_upper=g_upper,
        guardrail_lower=g_lower,
        simulation_mode=sim_mode,
        legacy_target=legacy_target,
        n_simulations=n_sims,
        use_fat_tails=use_fat_tails,
    )


# ─── RUN BUTTON ──────────────────────────────────────────────────────────────
st.divider()
run_col1, run_col2, run_col3 = st.columns([2, 2, 3])
with run_col1:
    run_button = st.button("🚀 Run Simulation", use_container_width=True, type="primary")
with run_col2:
    if st.session_state.result is not None:
        profile_for_export = build_profile_from_ui()
        csv_data = export_csv(st.session_state.result, profile_for_export)
        st.download_button("📥 Export CSV", csv_data, "retirement_simulation.csv", "text/csv", use_container_width=True)

if run_button:
    profile = build_profile_from_ui()
    if not st.session_state.get("assets_list"):
        st.warning("⚠️ No assets added. Add assets in the **Assets & CPF** tab for meaningful results.")
    if pre_ret_expenses == 0 and post_ret_expenses == 0:
        st.warning("⚠️ Expenses are zero. Please enter your expected living expenses.")

    with st.spinner(f"Running {n_sims:,} Monte Carlo simulations..."):
        progress_bar = st.progress(0)
        def update_progress(frac):
            progress_bar.progress(min(frac, 1.0))
        result = run_monte_carlo(profile, progress_callback=update_progress)
        st.session_state.result = result
        st.session_state.sim_profile = profile
        progress_bar.progress(1.0)
    st.success("✅ Simulation complete! View results in the **Results** and **Analysis** tabs.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6: RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_results:
    result = st.session_state.get("result")
    sim_profile = st.session_state.get("sim_profile")

    if result is None:
        st.info("👈 Fill in your profile and click **Run Simulation** to see results.")
    else:
        # ── Key Metrics Row ───────────────────────────────────────────────────
        sr = result.success_rate
        color_class = "success-green" if sr >= 80 else ("warning-yellow" if sr >= 60 else "danger-red")
        icon = "🟢" if sr >= 80 else ("🟡" if sr >= 60 else "🔴")

        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Success Rate</div>
                <div class="metric-value {color_class}">{sr:.1f}%</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            bkr = result.bankruptcy_rate
            bk_col = "danger-red" if bkr > 20 else ("warning-yellow" if bkr > 10 else "success-green")
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Bankruptcy Risk</div>
                <div class="metric-value {bk_col}">{bkr:.1f}%</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Median Final Net Worth</div>
                <div class="metric-value">SGD {result.median_final_nw/1e6:.2f}M</div>
            </div>""", unsafe_allow_html=True)
        with m4:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Best Case (P90)</div>
                <div class="metric-value success-green">SGD {result.p90_final_nw/1e6:.2f}M</div>
            </div>""", unsafe_allow_html=True)
        with m5:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Worst Case (P10)</div>
                <div class="metric-value danger-red">SGD {result.p10_final_nw/1e6:.2f}M</div>
            </div>""", unsafe_allow_html=True)

        # ── Key Insights ──────────────────────────────────────────────────────
        st.markdown("")
        if sr >= 90:
            st.markdown(f"""<div class="insight-box">
                🟢 <strong>Strong Plan:</strong> You can retire at age {sim_profile.retirement_age} with <strong>{sr:.0f}% confidence</strong>. Your plan is resilient to market downturns and inflation.
            </div>""", unsafe_allow_html=True)
        elif sr >= 70:
            st.markdown(f"""<div class="insight-box">
                🟡 <strong>Moderate Risk:</strong> Retirement at {sim_profile.retirement_age} has a <strong>{sr:.0f}% success rate</strong>. Consider increasing savings, delaying retirement by 2–3 years, or reducing post-retirement expenses.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="insight-box" style="border-left-color: #ef4444;">
                🔴 <strong>High Risk:</strong> Only <strong>{sr:.0f}% success probability</strong>. You risk running out of money before age {sim_profile.lifespan}. Significant adjustments needed.
            </div>""", unsafe_allow_html=True)

        if result.median_bankruptcy_age:
            st.markdown(f"""<div class="insight-box" style="border-left-color: #f59e0b;">
                ⚠️ In worst-case scenarios, funds may run out around <strong>age {result.median_bankruptcy_age:.0f}</strong>.
            </div>""", unsafe_allow_html=True)

        # ── Net Worth Fan Chart ───────────────────────────────────────────────
        st.markdown('<div class="section-header">Net Worth Projection</div>', unsafe_allow_html=True)
        ages_arr = result.ages[:result.n_years]

        fig = go.Figure()

        # Shaded bands
        fig.add_trace(go.Scatter(
            x=ages_arr + ages_arr[::-1],
            y=(result.p90_path / 1e6).tolist() + (result.p10_path / 1e6).tolist()[::-1],
            fill='toself', fillcolor='rgba(79,110,247,0.12)',
            line=dict(color='rgba(255,255,255,0)'),
            name='P10–P90 Range', showlegend=True,
        ))
        fig.add_trace(go.Scatter(
            x=ages_arr + ages_arr[::-1],
            y=(result.p75_path / 1e6).tolist() + (result.p25_path / 1e6).tolist()[::-1],
            fill='toself', fillcolor='rgba(79,110,247,0.22)',
            line=dict(color='rgba(255,255,255,0)'),
            name='P25–P75 Range', showlegend=True,
        ))

        # Lines
        fig.add_trace(go.Scatter(x=ages_arr, y=result.median_path / 1e6, name='Median',
                                  line=dict(color='#4f6ef7', width=3)))
        fig.add_trace(go.Scatter(x=ages_arr, y=result.p90_path / 1e6, name='P90 (Best 10%)',
                                  line=dict(color='#10d48e', width=1.5, dash='dot')))
        fig.add_trace(go.Scatter(x=ages_arr, y=result.p10_path / 1e6, name='P10 (Worst 10%)',
                                  line=dict(color='#ef4444', width=1.5, dash='dot')))
        fig.add_trace(go.Scatter(x=ages_arr, y=result.best_path / 1e6, name='Best Simulation',
                                  line=dict(color='#10d48e', width=1, dash='dash'), opacity=0.5))
        fig.add_trace(go.Scatter(x=ages_arr, y=result.worst_path / 1e6, name='Worst Simulation',
                                  line=dict(color='#ef4444', width=1, dash='dash'), opacity=0.5))

        # Retirement age marker
        fig.add_vline(x=sim_profile.retirement_age, line_color='#f59e0b', line_dash='dash', line_width=1.5,
                      annotation_text=f"Retire {sim_profile.retirement_age}", annotation_position="top right",
                      annotation_font_color='#f59e0b')

        # Zero line
        fig.add_hline(y=0, line_color='#ef4444', line_dash='dash', line_width=1, opacity=0.5)

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(20,22,35,0.6)',
            font=dict(color='#a0a8cc', family='IBM Plex Mono'),
            xaxis=dict(title='Age', gridcolor='#1e2235', showgrid=True),
            yaxis=dict(title='Net Worth (SGD Millions)', gridcolor='#1e2235'),
            legend=dict(bgcolor='rgba(20,22,35,0.8)', bordercolor='#2e3248', borderwidth=1),
            height=480, margin=dict(t=20, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── CPF Chart ─────────────────────────────────────────────────────────
        st.markdown('<div class="section-header">CPF Account Balances Over Time</div>', unsafe_allow_html=True)

        cpf_traj = simulate_cpf_trajectory(
            current_age=sim_profile.current_age,
            retirement_age=sim_profile.retirement_age,
            lifespan=sim_profile.lifespan,
            annual_income_series=sim_profile.annual_income_series(result.n_years + 1),
            cpf_initial=sim_profile.cpf_accounts,
            retirement_sum=sim_profile.cpf_retirement_sum,
            payout_plan=sim_profile.cpflife_payout_plan,
            payout_age=sim_profile.cpflife_payout_age,
        )
        cpf_ages = [d["age"] for d in cpf_traj["yearly"]]
        cpf_oa_vals = [d["oa"] / 1e3 for d in cpf_traj["yearly"]]
        cpf_sa_vals = [d["sa"] / 1e3 for d in cpf_traj["yearly"]]
        cpf_ma_vals = [d["ma"] / 1e3 for d in cpf_traj["yearly"]]
        cpf_ra_vals = [d["ra"] / 1e3 for d in cpf_traj["yearly"]]
        cpf_life_vals = [d["cpflife_monthly"] for d in cpf_traj["yearly"]]

        fig_cpf = make_subplots(specs=[[{"secondary_y": True}]])
        fig_cpf.add_trace(go.Bar(x=cpf_ages, y=cpf_oa_vals, name='OA', marker_color='#4f6ef7'), secondary_y=False)
        fig_cpf.add_trace(go.Bar(x=cpf_ages, y=cpf_sa_vals, name='SA', marker_color='#10d48e'), secondary_y=False)
        fig_cpf.add_trace(go.Bar(x=cpf_ages, y=cpf_ma_vals, name='MA', marker_color='#f59e0b'), secondary_y=False)
        fig_cpf.add_trace(go.Bar(x=cpf_ages, y=cpf_ra_vals, name='RA', marker_color='#a78bfa'), secondary_y=False)
        fig_cpf.add_trace(go.Scatter(x=cpf_ages, y=cpf_life_vals, name='CPF LIFE Monthly (SGD)',
                                      line=dict(color='#fb7185', width=2), mode='lines+markers',
                                      marker=dict(size=4)), secondary_y=True)
        fig_cpf.update_layout(
            barmode='stack',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(20,22,35,0.6)',
            font=dict(color='#a0a8cc', family='IBM Plex Mono'),
            xaxis=dict(title='Age', gridcolor='#1e2235'),
            yaxis=dict(title='Balance (SGD Thousands)', gridcolor='#1e2235'),
            legend=dict(bgcolor='rgba(20,22,35,0.8)', bordercolor='#2e3248', borderwidth=1),
            height=380, margin=dict(t=20, b=40),
        )
        fig_cpf.update_yaxes(title_text="CPF LIFE Monthly Payout (SGD)", secondary_y=True)
        st.plotly_chart(fig_cpf, use_container_width=True)

        # ── Summary Table ─────────────────────────────────────────────────────
        st.markdown('<div class="section-header">Year-by-Year Summary</div>', unsafe_allow_html=True)
        df = pd.DataFrame({
            "Age": ages_arr,
            "Median NW (SGD)": (result.median_path).astype(int),
            "P25 NW (SGD)": (result.p25_path).astype(int),
            "P75 NW (SGD)": (result.p75_path).astype(int),
            "P10 NW (SGD)": (result.p10_path).astype(int),
            "P90 NW (SGD)": (result.p90_path).astype(int),
        })
        df["Median NW (SGD)"] = df["Median NW (SGD)"].apply(lambda x: f"SGD {x:,.0f}")
        df["P25 NW (SGD)"] = df["P25 NW (SGD)"].apply(lambda x: f"SGD {x:,.0f}")
        df["P75 NW (SGD)"] = df["P75 NW (SGD)"].apply(lambda x: f"SGD {x:,.0f}")
        df["P10 NW (SGD)"] = df["P10 NW (SGD)"].apply(lambda x: f"SGD {x:,.0f}")
        df["P90 NW (SGD)"] = df["P90 NW (SGD)"].apply(lambda x: f"SGD {x:,.0f}")
        st.dataframe(df, use_container_width=True, height=300)

        # ── Text Report ───────────────────────────────────────────────────────
        report_text = export_summary_text(result, sim_profile)
        st.download_button("📄 Download Text Report", report_text.encode(), "retirement_report.txt", "text/plain")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7: SENSITIVITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_analysis:
    result = st.session_state.get("result")
    sim_profile = st.session_state.get("sim_profile")

    if result is None:
        st.info("Run a simulation first to see sensitivity analysis.")
    else:
        st.markdown('<div class="section-header">Scenario Comparison</div>', unsafe_allow_html=True)
        st.caption("Compare base case vs optimistic vs pessimistic scenarios.")

        if st.button("🔬 Run Scenario Analysis (3x simulations)", use_container_width=False):
            scenarios = {
                "Pessimistic": {"infl": 4.0, "ret_adj": -0.02, "ret_age_adj": 0},
                "Base": {"infl": sim_profile.inflation_rate, "ret_adj": 0, "ret_age_adj": 0},
                "Optimistic": {"infl": 1.5, "ret_adj": 0.015, "ret_age_adj": 0},
            }
            scenario_results = {}
            import copy
            for sc_name, params in scenarios.items():
                sc_profile = copy.deepcopy(sim_profile)
                sc_profile.inflation_rate = params["infl"]
                sc_profile.n_simulations = 2000
                for asset in sc_profile.portfolio.assets:
                    asset.expected_return = max(0, asset.expected_return + params["ret_adj"] * 100)
                with st.spinner(f"Running {sc_name} scenario..."):
                    scenario_results[sc_name] = run_monte_carlo(sc_profile)

            # Plot comparison
            fig_sc = go.Figure()
            colors = {"Pessimistic": "#ef4444", "Base": "#4f6ef7", "Optimistic": "#10d48e"}
            for sc_name, sc_result in scenario_results.items():
                sc_ages = sc_result.ages[:sc_result.n_years]
                fig_sc.add_trace(go.Scatter(
                    x=sc_ages, y=sc_result.median_path / 1e6,
                    name=f"{sc_name} (Success: {sc_result.success_rate:.0f}%)",
                    line=dict(color=colors[sc_name], width=2.5),
                ))
            fig_sc.add_hline(y=0, line_color='white', line_dash='dash', line_width=1, opacity=0.3)
            fig_sc.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(20,22,35,0.6)',
                font=dict(color='#a0a8cc', family='IBM Plex Mono'),
                xaxis=dict(title='Age', gridcolor='#1e2235'),
                yaxis=dict(title='Net Worth (SGD Millions)', gridcolor='#1e2235'),
                height=400, margin=dict(t=20, b=40),
            )
            st.plotly_chart(fig_sc, use_container_width=True)

            sc_data = {sc: {"Success Rate": f"{r.success_rate:.1f}%",
                             "Median Final NW": f"SGD {r.median_final_nw/1e6:.2f}M",
                             "P10 Final NW": f"SGD {r.p10_final_nw/1e6:.2f}M"}
                       for sc, r in scenario_results.items()}
            st.table(pd.DataFrame(sc_data).T)

        st.markdown('<div class="section-header">Sensitivity: Retirement Age</div>', unsafe_allow_html=True)
        if st.button("📊 Retirement Age Sensitivity", use_container_width=False):
            ret_ages = list(range(max(sim_profile.current_age + 5, 50), min(sim_profile.lifespan - 5, 75), 2))
            success_rates = []
            import copy
            for ra in ret_ages:
                tp = copy.deepcopy(sim_profile)
                tp.retirement_age = ra
                tp.n_simulations = 1000
                r = run_monte_carlo(tp)
                success_rates.append(r.success_rate)

            fig_ra = go.Figure()
            fig_ra.add_trace(go.Scatter(x=ret_ages, y=success_rates, mode='lines+markers',
                                         line=dict(color='#4f6ef7', width=2.5),
                                         marker=dict(size=8, color='#4f6ef7')))
            fig_ra.add_hline(y=80, line_color='#10d48e', line_dash='dash', annotation_text="80% threshold")
            fig_ra.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(20,22,35,0.6)',
                font=dict(color='#a0a8cc', family='IBM Plex Mono'),
                xaxis=dict(title='Retirement Age', gridcolor='#1e2235'),
                yaxis=dict(title='Success Rate (%)', gridcolor='#1e2235', range=[0, 105]),
                height=350, margin=dict(t=20, b=40),
            )
            st.plotly_chart(fig_ra, use_container_width=True)

        st.markdown('<div class="section-header">Probability Distribution at Death</div>', unsafe_allow_html=True)
        final_nw = result.net_worth_paths[:, -1] / 1e6
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=final_nw, nbinsx=60,
            marker_color='#4f6ef7', opacity=0.75,
            name='Final Net Worth Distribution',
        ))
        fig_dist.add_vline(x=float(np.median(final_nw)), line_color='#10d48e', line_dash='dash',
                           annotation_text=f"Median: SGD {np.median(final_nw):.2f}M", annotation_position="top right")
        fig_dist.add_vline(x=0, line_color='#ef4444', line_dash='dash', annotation_text="Zero", annotation_position="top left")
        fig_dist.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(20,22,35,0.6)',
            font=dict(color='#a0a8cc', family='IBM Plex Mono'),
            xaxis=dict(title='Net Worth at Death (SGD Millions)', gridcolor='#1e2235'),
            yaxis=dict(title='Number of Simulations', gridcolor='#1e2235'),
            height=350, margin=dict(t=20, b=40),
        )
        st.plotly_chart(fig_dist, use_container_width=True)
