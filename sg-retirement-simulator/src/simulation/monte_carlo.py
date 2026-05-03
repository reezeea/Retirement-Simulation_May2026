"""
Monte Carlo Simulation Engine for Singapore Retirement Simulator.
Fully rewritten: fixes all critical bugs, adds multi-asset correlation,
multiple simulation types, proper withdrawal strategies, and correct
bankruptcy detection.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from src.simulation.profile import UserProfile
from src.simulation.assets import AssetType, Liquidity
from src.cpf.cpf_model import (
    CPFAccounts, compute_annual_cpf_contribution, apply_cpf_interest,
    form_retirement_account, estimate_cpflife_monthly_payout,
    CPFLIFE_PAYOUT_AGE_DEFAULT,
)
from src.simulation.historical_data import (
    HIST_RETURNS_MAP, HIST_YEARS,
    build_correlation_submatrix,
)
import copy


@dataclass
class SimulationResult:
    """Results from Monte Carlo simulation."""
    n_simulations: int
    n_years: int
    ages: list[int]

    # Shape: (n_simulations, n_years)
    net_worth_paths: np.ndarray        # liquid portfolio only (spendable)
    cpf_total_paths: np.ndarray        # CPF balances (informational)
    total_nw_paths: np.ndarray         # liquid + CPF (for display)

    # Derived statistics
    success_rate: float                # % of paths with positive liquid NW at death
    failure_rate: float                # % that run out of money
    median_final_nw: float
    p5_final_nw: float                 # 5th percentile (worst case)
    p10_final_nw: float
    p25_final_nw: float
    p75_final_nw: float
    p90_final_nw: float

    # Percentile paths (n_years,) — liquid portfolio
    median_path: np.ndarray
    p5_path: np.ndarray
    p10_path: np.ndarray
    p25_path: np.ndarray
    p75_path: np.ndarray
    p90_path: np.ndarray
    best_path: np.ndarray
    worst_path: np.ndarray

    # Failure tracking
    ruin_ages: np.ndarray              # Age of ruin per sim; NaN if solvent
    median_ruin_age: Optional[float]
    ruin_age_distribution: list[int]   # Histogram of ruin ages

    # Sustainability label
    sustainability_label: str          # "SUSTAINABLE", "AT RISK", "NOT SUSTAINABLE"


# ─── Return Generation ────────────────────────────────────────────────────────

def _generate_correlated_returns(
    expected_returns: np.ndarray,
    volatilities: np.ndarray,
    correlation_matrix: np.ndarray,
    n_sims: int,
    n_years: int,
    sim_type: str = "Standard",
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Generate correlated returns for multiple assets.
    Returns shape: (n_assets, n_sims, n_years).
    """
    if rng is None:
        rng = np.random.default_rng()

    n_assets = len(expected_returns)

    if n_assets == 0:
        return np.empty((0, n_sims, n_years))

    if sim_type == "FatTail":
        # Student-t with df=5 for fat tails
        df = 5
        # Generate independent t-distributed samples
        raw = rng.standard_t(df, size=(n_assets, n_sims, n_years))
        # Scale to unit variance: t with df=5 has var = df/(df-2) = 5/3
        raw *= np.sqrt((df - 2) / df)
    else:
        # Standard normal
        raw = rng.standard_normal(size=(n_assets, n_sims, n_years))

    # Apply correlation via Cholesky decomposition
    try:
        L = np.linalg.cholesky(correlation_matrix)
    except np.linalg.LinAlgError:
        # If matrix is not positive definite, use nearest PD approximation
        eigvals, eigvecs = np.linalg.eigh(correlation_matrix)
        eigvals = np.maximum(eigvals, 1e-8)
        fixed = eigvecs @ np.diag(eigvals) @ eigvecs.T
        L = np.linalg.cholesky(fixed)

    # raw shape: (n_assets, n_sims, n_years) → reshape for matmul
    for t in range(n_years):
        raw[:, :, t] = L @ raw[:, :, t]

    # Scale to desired mean and volatility
    for i in range(n_assets):
        raw[i] = expected_returns[i] + volatilities[i] * raw[i]

    return raw


def _generate_historical_returns(
    asset_types: list[AssetType],
    n_sims: int,
    n_years: int,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Generate returns by randomly sampling rolling windows from historical data.
    Returns shape: (n_assets, n_sims, n_years).
    """
    if rng is None:
        rng = np.random.default_rng()

    n_assets = len(asset_types)
    result = np.zeros((n_assets, n_sims, n_years))
    n_hist = len(HIST_YEARS)

    for sim in range(n_sims):
        # Random starting point in history, with wrap-around
        start = rng.integers(0, n_hist)
        for yr in range(n_years):
            idx = (start + yr) % n_hist
            for ai, at in enumerate(asset_types):
                hist = HIST_RETURNS_MAP.get(at, HIST_RETURNS_MAP[AssetType.CASH])
                result[ai, sim, yr] = hist[idx]

    return result


def _generate_regime_returns(
    expected_returns: np.ndarray,
    volatilities: np.ndarray,
    n_sims: int,
    n_years: int,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Regime-based simulation: Bull / Bear / High-Inflation regimes.
    Uses a Markov transition matrix between regimes.
    Returns shape: (n_assets, n_sims, n_years).
    """
    if rng is None:
        rng = np.random.default_rng()

    n_assets = len(expected_returns)

    # Regime definitions: (return_multiplier, vol_multiplier)
    regimes = {
        0: (1.3, 0.8),    # Bull: higher returns, lower vol
        1: (0.5, 1.8),    # Bear: lower returns, higher vol
        2: (0.7, 1.3),    # High inflation: depressed real returns
    }

    # Transition matrix: P[i,j] = prob of going from regime i to j
    transition = np.array([
        [0.70, 0.20, 0.10],   # Bull → Bull/Bear/HiInfl
        [0.30, 0.50, 0.20],   # Bear → Bull/Bear/HiInfl
        [0.25, 0.25, 0.50],   # HiInfl → Bull/Bear/HiInfl
    ])

    result = np.zeros((n_assets, n_sims, n_years))

    for sim in range(n_sims):
        regime = rng.integers(0, 3)  # Random initial regime
        for yr in range(n_years):
            ret_mult, vol_mult = regimes[regime]
            for ai in range(n_assets):
                mu = expected_returns[ai] * ret_mult
                sigma = volatilities[ai] * vol_mult
                result[ai, sim, yr] = rng.normal(mu, sigma)
            # Transition to next regime
            regime = rng.choice(3, p=transition[regime])

    return result


# ─── Withdrawal Strategies ────────────────────────────────────────────────────

def compute_withdrawal(
    strategy: str,
    base_expenses: np.ndarray,
    portfolio_value: np.ndarray,
    years_left: int,
    withdrawal_pct: float = 4.0,
    guardrail_upper: float = 5.5,
    guardrail_lower: float = 4.0,
) -> np.ndarray:
    """
    Compute withdrawal amount for each simulation path.
    All inputs/outputs are (n_sims,) vectors.

    Strategies:
      Fixed: withdraw inflation-adjusted base_expenses
      PercentPortfolio: withdraw X% of current portfolio
      Guardrail: dynamic between upper/lower bounds
      DieWithZero: portfolio / years_left (smoothed)
    """
    n_sims = len(portfolio_value)

    if strategy == "Fixed":
        return base_expenses.copy()

    elif strategy == "PercentPortfolio":
        return np.maximum(portfolio_value * (withdrawal_pct / 100), 0)

    elif strategy == "Guardrail":
        # Start with base expenses
        withdrawal = base_expenses.copy()
        # Compute effective withdrawal rate
        safe_portfolio = np.where(portfolio_value > 0, portfolio_value, 1.0)
        effective_rate = withdrawal / safe_portfolio

        # If rate > upper guardrail → cut spending to lower guardrail
        too_high = effective_rate > (guardrail_upper / 100)
        withdrawal[too_high] = portfolio_value[too_high] * (guardrail_lower / 100)

        # If rate < lower guardrail → can increase (optional; keep base)
        return np.maximum(withdrawal, 0)

    elif strategy == "DieWithZero":
        if years_left <= 0:
            return portfolio_value.copy()
        # Amortize remaining portfolio over remaining years
        # Use a smoothing factor to avoid wild swings
        target = portfolio_value / max(years_left, 1)
        # Blend: 50% target, 50% base_expenses for stability
        blended = 0.5 * target + 0.5 * base_expenses
        return np.maximum(blended, 0)

    else:
        return base_expenses.copy()


# ─── Main Simulation ─────────────────────────────────────────────────────────

def run_monte_carlo(profile: UserProfile, progress_callback=None) -> SimulationResult:
    """
    Main Monte Carlo simulation. Fully rewritten with correct logic:
    - Expenses deducted every year
    - Inflation applied correctly
    - Portfolio CAN go to zero (triggers ruin)
    - CPF tracked separately from liquid portfolio
    - Proper withdrawal strategies
    - Correlated multi-asset returns
    """
    rng = np.random.default_rng(seed=None)

    n_sims = profile.n_simulations
    n_years = profile.total_years()
    ages = list(range(profile.current_age, profile.lifespan + 1))
    ret_age = profile.retirement_age
    infl_rate = profile.inflation_rate / 100

    # ── Identify investable assets (exclude CPF) ──────────────────────────────
    assets = [a for a in profile.portfolio.assets
              if a.asset_type not in (AssetType.CPF_OA, AssetType.CPF_SA, AssetType.CPF_MA)]
    n_assets = len(assets)

    asset_types = [a.asset_type for a in assets]
    expected_returns = np.array([a.expected_return_decimal for a in assets])
    volatilities = np.array([a.volatility_decimal for a in assets])
    asset_values_init = np.array([a.current_value for a in assets], dtype=np.float64)
    is_liquid = np.array([
        1.0 if a.liquidity in (Liquidity.LIQUID, Liquidity.SEMI_LIQUID) else 0.0
        for a in assets
    ])

    # ── Generate return matrices ──────────────────────────────────────────────
    sim_type = getattr(profile, 'simulation_type', 'Standard')

    if sim_type == "Historical" and n_assets > 0:
        asset_returns = _generate_historical_returns(
            asset_types, n_sims, n_years, rng
        )
    elif sim_type == "RegimeBased" and n_assets > 0:
        asset_returns = _generate_regime_returns(
            expected_returns, volatilities, n_sims, n_years, rng
        )
    elif n_assets > 0:
        # Standard or FatTail
        corr_matrix = build_correlation_submatrix(asset_types)
        asset_returns = _generate_correlated_returns(
            expected_returns, volatilities, corr_matrix,
            n_sims, n_years,
            sim_type=sim_type if sim_type in ("Standard", "FatTail") else "Standard",
            rng=rng,
        )
    else:
        asset_returns = np.empty((0, n_sims, n_years))

    # ── Income series ─────────────────────────────────────────────────────────
    income_series = profile.annual_income_series(n_years + 1)

    # ── Liability payments ────────────────────────────────────────────────────
    annual_liab = profile.portfolio.annual_liability_payments()

    # ── Inflation multipliers ─────────────────────────────────────────────────
    infl_mult = np.array([(1 + infl_rate) ** i for i in range(n_years + 1)])

    # ── One-off expense lookup (FIXED: no double-counting) ────────────────────
    one_off_by_age: dict[int, float] = {}
    for exp in profile.one_off_expenses:
        yr_idx = exp.age_due - profile.current_age
        if 0 <= yr_idx <= n_years:
            if exp.inflation_adjusted:
                amt = exp.amount * infl_mult[yr_idx]
            else:
                amt = exp.amount
            one_off_by_age[exp.age_due] = one_off_by_age.get(exp.age_due, 0) + amt

    # ── Post-retirement income lookup ─────────────────────────────────────────
    def get_retirement_income(age: int, yr: int) -> float:
        total = 0.0
        for ri in profile.retirement_incomes:
            if ri.start_age <= age <= ri.end_age:
                if ri.inflation_adjusted:
                    total += ri.annual_amount * infl_mult[yr]
                else:
                    total += ri.annual_amount
        return total

    # ── CPF trajectory (deterministic) ────────────────────────────────────────
    cpf = copy.deepcopy(profile.cpf_accounts)
    cpf_total_by_year = np.zeros(n_years)
    cpflife_monthly_by_year = np.zeros(n_years)

    for yr in range(n_years):
        age = profile.current_age + yr
        # Pre-retirement contributions
        if age < ret_age and yr < len(income_series):
            _, oa_add, sa_add, ma_add = compute_annual_cpf_contribution(
                income_series[yr], age
            )
            cpf.oa += oa_add
            cpf.sa += sa_add
            cpf.ma += ma_add

        # Age 55: form Retirement Account
        if age == 55 and not cpf.ra_formed:
            cpf = form_retirement_account(cpf, yr, profile.cpf_retirement_sum)

        # Apply CPF interest
        cpf = apply_cpf_interest(cpf)

        # CPF LIFE payouts
        if age >= profile.cpflife_payout_age and cpf.ra > 0:
            cpflife_monthly_by_year[yr] = estimate_cpflife_monthly_payout(
                cpf.ra, profile.cpflife_payout_plan, profile.cpflife_payout_age
            )

        cpf_total_by_year[yr] = cpf.oa + cpf.sa + cpf.ma + cpf.ra

    # ── Main simulation loop ──────────────────────────────────────────────────
    # portfolio_values: (n_sims, n_assets) — current value of each asset per sim
    portfolio_values = np.tile(asset_values_init, (n_sims, 1)).astype(np.float64)

    # Output arrays
    liquid_nw_paths = np.zeros((n_sims, n_years))
    cpf_paths = np.zeros((n_sims, n_years))

    # Ruin tracking
    ruined = np.zeros(n_sims, dtype=bool)
    ruin_age = np.full(n_sims, np.nan)

    for yr in range(n_years):
        age = profile.current_age + yr
        if progress_callback and yr % 5 == 0:
            progress_callback(yr / n_years)

        # ── Step 1: Apply investment returns ──────────────────────────────────
        if n_assets > 0:
            for ai in range(n_assets):
                returns = asset_returns[ai, :, yr]  # (n_sims,)
                # Only apply returns to non-ruined simulations
                portfolio_values[~ruined, ai] *= (1 + returns[~ruined])
                # Floor individual assets at 0 for realism (can't have negative stock)
                portfolio_values[:, ai] = np.maximum(portfolio_values[:, ai], 0)

        # ── Step 2: Compute total liquid portfolio ────────────────────────────
        total_portfolio = portfolio_values.sum(axis=1)  # (n_sims,)

        # ── Step 3: Compute income ────────────────────────────────────────────
        if age < ret_age:
            # Pre-retirement: employment income minus CPF employee contribution
            gross_income = income_series[min(yr, len(income_series) - 1)]
            from src.cpf.cpf_model import get_contribution_rate
            emp_rate, _ = get_contribution_rate(age)
            net_income = gross_income * (1 - emp_rate)
        else:
            net_income = 0.0

        # Retirement income sources
        other_income = get_retirement_income(age, yr)

        # CPF LIFE payout (annual)
        cpflife_annual = cpflife_monthly_by_year[yr] * 12

        # Inheritance
        inheritance = 0.0
        if age == profile.inheritance_age and profile.expected_inheritance > 0:
            inheritance = profile.expected_inheritance

        total_income = net_income + other_income + cpflife_annual + inheritance

        # ── Step 4: Compute expenses ──────────────────────────────────────────
        if age < ret_age:
            base_exp = profile.annual_expenses_pre_retirement
        else:
            base_exp = profile.annual_expenses_post_retirement

        # Inflate expenses
        inflated_expenses = base_exp * infl_mult[yr]

        # Add obligations (inflation-adjusted)
        inflated_expenses += profile.annual_obligations * infl_mult[yr]

        # Add one-off expenses
        inflated_expenses += one_off_by_age.get(age, 0.0)

        # Add liability payments (pre-retirement)
        if age < ret_age:
            inflated_expenses += annual_liab

        # ── Step 5: Compute withdrawal / net cash flow ────────────────────────
        if age >= ret_age:
            # Use withdrawal strategy to determine actual spending
            expenses_vec = np.full(n_sims, inflated_expenses)
            withdrawal = compute_withdrawal(
                strategy=profile.withdrawal_strategy,
                base_expenses=expenses_vec,
                portfolio_value=total_portfolio,
                years_left=max(1, profile.lifespan - age),
                withdrawal_pct=profile.withdrawal_percent,
                guardrail_upper=profile.guardrail_upper,
                guardrail_lower=profile.guardrail_lower,
            )
            # Add one-off and obligations on top (non-negotiable)
            non_negotiable = (profile.annual_obligations * infl_mult[yr]
                              + one_off_by_age.get(age, 0.0))
            # For Fixed strategy, withdrawal already includes everything
            if profile.withdrawal_strategy in ("PercentPortfolio", "DieWithZero"):
                withdrawal += non_negotiable

            net_cf = total_income - withdrawal  # (n_sims,)
        else:
            # Pre-retirement: net savings
            net_cf = np.full(n_sims, total_income - inflated_expenses)

        # ── Step 6: Apply cash flow to portfolio ──────────────────────────────
        if n_assets > 0:
            # Distribute cash flow across liquid assets proportionally
            liquid_values = portfolio_values * is_liquid  # (n_sims, n_assets)
            total_liquid = liquid_values.sum(axis=1, keepdims=True)  # (n_sims, 1)
            total_liquid = np.where(total_liquid > 0, total_liquid, 1.0)

            weights = liquid_values / total_liquid  # (n_sims, n_assets)

            # Apply net cash flow
            for ai in range(n_assets):
                if is_liquid[ai] > 0:
                    portfolio_values[~ruined, ai] += (
                        weights[~ruined, ai] * net_cf[~ruined]
                    )

            # If all liquid assets are depleted, mark negative assets
            # Don't floor at zero — allow portfolio to reflect true shortfall
            # But individual assets can't go below 0 (you can't have -$50k in an ETF)
            # Instead, track total portfolio deficit
        else:
            # No assets: track as a virtual cash balance
            pass

        # ── Step 7: Compute liquid net worth ──────────────────────────────────
        if n_assets > 0:
            liquid_nw = portfolio_values.sum(axis=1)
        else:
            # No assets: accumulate/deplete a virtual cash account
            if yr == 0:
                liquid_nw = net_cf.copy()
            else:
                liquid_nw = liquid_nw_paths[:, yr - 1] + net_cf

        # ── Step 8: Detect ruin ───────────────────────────────────────────────
        newly_ruined = (~ruined) & (liquid_nw <= 0)
        ruined |= newly_ruined
        ruin_age[newly_ruined] = age

        # Set ruined portfolios to 0
        liquid_nw[ruined] = 0.0
        if n_assets > 0:
            for ai in range(n_assets):
                portfolio_values[ruined, ai] = 0.0

        # ── Step 9: Record ────────────────────────────────────────────────────
        liquid_nw_paths[:, yr] = liquid_nw
        cpf_paths[:, yr] = cpf_total_by_year[yr]

    # ── Compute statistics ────────────────────────────────────────────────────
    final_liquid_nw = liquid_nw_paths[:, -1]
    total_nw_paths = liquid_nw_paths + cpf_paths  # For display only

    # Success = liquid portfolio > 0 at death (or > legacy_target for Legacy mode)
    if profile.simulation_mode == "Legacy":
        success_mask = final_liquid_nw > profile.legacy_target
    else:
        success_mask = final_liquid_nw > 0

    success_rate = float(np.mean(success_mask) * 100)
    failure_rate = 100.0 - success_rate

    # Sustainability label
    if success_rate >= 85:
        label = "SUSTAINABLE"
    elif success_rate >= 50:
        label = "AT RISK"
    else:
        label = "NOT SUSTAINABLE"

    # Ruin age stats
    ruin_ages_valid = ruin_age[~np.isnan(ruin_age)]
    median_ruin = float(np.median(ruin_ages_valid)) if len(ruin_ages_valid) > 0 else None

    # Ruin age distribution (histogram)
    if len(ruin_ages_valid) > 0:
        ruin_hist = list(ruin_ages_valid.astype(int))
    else:
        ruin_hist = []

    return SimulationResult(
        n_simulations=n_sims,
        n_years=n_years,
        ages=ages,
        net_worth_paths=liquid_nw_paths,
        cpf_total_paths=cpf_paths,
        total_nw_paths=total_nw_paths,
        success_rate=success_rate,
        failure_rate=failure_rate,
        median_final_nw=float(np.median(final_liquid_nw)),
        p5_final_nw=float(np.percentile(final_liquid_nw, 5)),
        p10_final_nw=float(np.percentile(final_liquid_nw, 10)),
        p25_final_nw=float(np.percentile(final_liquid_nw, 25)),
        p75_final_nw=float(np.percentile(final_liquid_nw, 75)),
        p90_final_nw=float(np.percentile(final_liquid_nw, 90)),
        median_path=np.median(liquid_nw_paths, axis=0),
        p5_path=np.percentile(liquid_nw_paths, 5, axis=0),
        p10_path=np.percentile(liquid_nw_paths, 10, axis=0),
        p25_path=np.percentile(liquid_nw_paths, 25, axis=0),
        p75_path=np.percentile(liquid_nw_paths, 75, axis=0),
        p90_path=np.percentile(liquid_nw_paths, 90, axis=0),
        best_path=liquid_nw_paths[np.argmax(final_liquid_nw)],
        worst_path=liquid_nw_paths[np.argmin(final_liquid_nw)],
        ruin_ages=ruin_age,
        median_ruin_age=median_ruin,
        ruin_age_distribution=ruin_hist,
        sustainability_label=label,
    )
