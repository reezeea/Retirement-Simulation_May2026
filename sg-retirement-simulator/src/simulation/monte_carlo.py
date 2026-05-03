"""
Monte Carlo Simulation Engine for Singapore Retirement Simulator.
Runs 5,000–10,000 simulation paths using per-asset return modeling.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from src.simulation.profile import UserProfile
from src.simulation.assets import AssetType, Liquidity
from src.cpf.cpf_model import (
    CPFAccounts, compute_annual_cpf_contribution, apply_cpf_interest,
    form_retirement_account, estimate_cpflife_monthly_payout,
    CPFLIFE_PAYOUT_AGE_DEFAULT,
)
import copy


@dataclass
class SimulationResult:
    """Results from Monte Carlo simulation."""
    n_simulations: int
    n_years: int
    ages: list[int]

    # Shape: (n_simulations, n_years)
    net_worth_paths: np.ndarray
    cash_flow_paths: np.ndarray
    cpf_total_paths: np.ndarray

    # Derived statistics
    success_rate: float          # % of paths with positive NW at death
    bankruptcy_rate: float       # % that go broke before death
    median_final_nw: float
    p10_final_nw: float          # 10th percentile
    p25_final_nw: float
    p75_final_nw: float
    p90_final_nw: float

    # Percentile paths (n_years,)
    median_path: np.ndarray
    p10_path: np.ndarray
    p25_path: np.ndarray
    p75_path: np.ndarray
    p90_path: np.ndarray
    best_path: np.ndarray
    worst_path: np.ndarray

    bankruptcy_ages: list[Optional[int]]   # Age of bankruptcy per sim, None if solvent
    median_bankruptcy_age: Optional[float]


def _generate_returns(
    expected_return: float,
    volatility: float,
    n_sims: int,
    n_years: int,
    use_fat_tails: bool = True,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Generate a (n_sims, n_years) matrix of annual returns.
    Uses normal distribution or fat-tailed (Student-t with df=5) distribution.
    """
    if rng is None:
        rng = np.random.default_rng()

    mu = expected_return
    sigma = volatility

    if use_fat_tails and sigma > 0.05:
        # Student-t with df=5 for fat tails
        df = 5
        t_samples = rng.standard_t(df, size=(n_sims, n_years))
        # Scale to match desired mean and std
        returns = mu + sigma * t_samples * np.sqrt((df - 2) / df)
    else:
        returns = rng.normal(mu, sigma, size=(n_sims, n_years))

    return returns


def run_monte_carlo(profile: UserProfile, progress_callback=None) -> SimulationResult:
    """
    Main Monte Carlo simulation.
    Runs `profile.n_simulations` paths over the user's lifetime.
    """
    rng = np.random.default_rng(seed=None)  # Fresh seed each run

    n_sims = profile.n_simulations
    n_years = profile.total_years()
    ages = list(range(profile.current_age, profile.lifespan + 1))
    ret_age = profile.retirement_age
    infl_rate = profile.inflation_rate / 100

    # ── Pre-generate per-asset return matrices ────────────────────────────────
    assets = profile.portfolio.assets
    # Shape: (n_assets, n_sims, n_years)
    asset_returns = []
    asset_values_init = []
    for asset in assets:
        # CPF assets handled separately
        if asset.asset_type in (AssetType.CPF_OA, AssetType.CPF_SA, AssetType.CPF_MA):
            continue
        r = _generate_returns(
            expected_return=asset.expected_return_decimal,
            volatility=asset.volatility_decimal,
            n_sims=n_sims,
            n_years=n_years,
            use_fat_tails=profile.use_fat_tails,
            rng=rng,
        )
        asset_returns.append(r)
        asset_values_init.append(asset.current_value)

    n_investable_assets = len(asset_returns)
    asset_values_init = np.array(asset_values_init)  # (n_assets,)

    # ── Income series ────────────────────────────────────────────────────────
    income_series = profile.annual_income_series(n_years + 1)

    # ── Pre-compute liability payments ───────────────────────────────────────
    annual_liab = profile.portfolio.annual_liability_payments()

    # ── Inflation multipliers ─────────────────────────────────────────────────
    infl_mult = np.array([(1 + infl_rate) ** i for i in range(n_years + 1)])

    # ── One-off expense lookup ────────────────────────────────────────────────
    one_off_by_age: dict[int, float] = {}
    for exp in profile.one_off_expenses:
        amt = exp.amount * (infl_mult[exp.age_due - profile.current_age] if exp.inflation_adjusted and exp.age_due >= profile.current_age else exp.amount)
        one_off_by_age[exp.age_due] = one_off_by_age.get(exp.age_due, 0) + amt

    # ── Post-retirement income lookup ─────────────────────────────────────────
    retirement_income_by_age: dict[int, float] = {}
    for yr in range(n_years + 1):
        age = profile.current_age + yr
        for ri in profile.retirement_incomes:
            if ri.start_age <= age <= ri.end_age:
                amt = ri.annual_amount * (infl_mult[yr] if ri.inflation_adjusted else 1.0)
                retirement_income_by_age[age] = retirement_income_by_age.get(age, 0) + amt

    # ── CPF trajectory (deterministic, computed once) ─────────────────────────
    cpf = copy.deepcopy(profile.cpf_accounts)
    cpf_oa_arr = np.zeros(n_years)
    cpf_sa_arr = np.zeros(n_years)
    cpf_ma_arr = np.zeros(n_years)
    cpf_ra_arr = np.zeros(n_years)
    cpflife_monthly_arr = np.zeros(n_years)

    for yr in range(n_years):
        age = profile.current_age + yr
        if age < ret_age and yr < len(income_series):
            _, oa_add, sa_add, ma_add = compute_annual_cpf_contribution(income_series[yr], age)
            cpf.oa += oa_add
            cpf.sa += sa_add
            cpf.ma += ma_add
        if age == 55 and not cpf.ra_formed:
            cpf = form_retirement_account(cpf, yr, profile.cpf_retirement_sum)
        cpf = apply_cpf_interest(cpf)
        if age >= profile.cpflife_payout_age and cpf.ra > 0:
            cpflife_monthly_arr[yr] = estimate_cpflife_monthly_payout(
                cpf.ra, profile.cpflife_payout_plan, profile.cpflife_payout_age
            )
        cpf_oa_arr[yr] = cpf.oa
        cpf_sa_arr[yr] = cpf.sa
        cpf_ma_arr[yr] = cpf.ma
        cpf_ra_arr[yr] = cpf.ra

    cpf_total_arr = cpf_oa_arr + cpf_sa_arr + cpf_ma_arr + cpf_ra_arr  # (n_years,)

    # ── Simulation loop ───────────────────────────────────────────────────────
    # net_worth_paths: (n_sims, n_years)
    net_worth_paths = np.zeros((n_sims, n_years))
    cash_flow_paths = np.zeros((n_sims, n_years))

    # Initial investable portfolio values: (n_sims, n_assets) – must be float64
    asset_values_init = asset_values_init.astype(np.float64)
    portfolio_values = np.tile(asset_values_init, (n_sims, 1)).astype(np.float64)

    bankruptcy_ages = [None] * n_sims

    for yr in range(n_years):
        age = profile.current_age + yr
        if progress_callback and yr % 5 == 0:
            progress_callback(yr / n_years)

        # ── Grow portfolio for this year ──────────────────────────────────────
        for ai in range(n_investable_assets):
            r_this_year = asset_returns[ai][:, yr]  # (n_sims,)
            portfolio_values[:, ai] *= (1 + r_this_year)

        total_portfolio = portfolio_values.sum(axis=1)  # (n_sims,)

        # ── Income ──────────────────────────────────────────────────────────
        if age < ret_age:
            earned_income = income_series[min(yr, len(income_series) - 1)]
            # Subtract CPF employee contribution (already goes to CPF, not investable)
            from src.cpf.cpf_model import get_contribution_rate
            emp_rate, _ = get_contribution_rate(age)
            net_income = earned_income * (1 - emp_rate)
        else:
            net_income = 0.0

        # Other income (rental, part-time, retirement income)
        other_income = retirement_income_by_age.get(age, 0.0)

        # CPF LIFE payout
        cpflife_annual = cpflife_monthly_arr[yr] * 12

        # Inheritance
        inheritance_income = 0.0
        if age == profile.inheritance_age and profile.expected_inheritance > 0:
            inheritance_income = profile.expected_inheritance

        total_income = net_income + other_income + cpflife_annual + inheritance_income

        # ── Expenses ─────────────────────────────────────────────────────────
        if age < ret_age:
            base_expenses = profile.annual_expenses_pre_retirement
        else:
            base_expenses = profile.annual_expenses_post_retirement

        # Inflation-adjusted
        expenses = base_expenses * infl_mult[yr]
        expenses += profile.annual_obligations * infl_mult[yr]
        expenses += one_off_by_age.get(age, 0.0)

        # ── Dynamic withdrawal adjustment ─────────────────────────────────────
        if age >= ret_age:
            if profile.withdrawal_strategy == "PercentPortfolio":
                target_withdrawal = total_portfolio * (profile.withdrawal_percent / 100)
                expenses = max(expenses, target_withdrawal)
            elif profile.withdrawal_strategy == "Guardrail":
                withdrawal_rate = expenses / np.where(total_portfolio > 0, total_portfolio, 1e-9)
                # Vectorized guardrail
                expenses_vec = np.where(
                    withdrawal_rate > profile.guardrail_upper / 100,
                    total_portfolio * (profile.guardrail_lower / 100),
                    expenses
                )
                expenses = expenses_vec  # now a vector (n_sims,)

        # Liability payments (pre-retirement mostly)
        if age < ret_age:
            expenses = np.asarray(expenses) + annual_liab

        # ── Cash flow ─────────────────────────────────────────────────────────
        net_cf = total_income - np.asarray(expenses)  # scalar or (n_sims,)

        # Inject into portfolio (add savings / subtract drawdowns)
        if n_investable_assets > 0:
            # Distribute cash flow proportionally across liquid assets
            liquid_mask = np.array([
                1.0 if assets[ai].liquidity == Liquidity.LIQUID else 0.0
                for ai in range(n_investable_assets)
            ])
            total_liquid = (portfolio_values * liquid_mask).sum(axis=1)  # (n_sims,)
            total_liquid = np.where(total_liquid > 0, total_liquid, 1e-9)

            for ai in range(n_investable_assets):
                if liquid_mask[ai] > 0:
                    weight = portfolio_values[:, ai] / total_liquid
                    portfolio_values[:, ai] += weight * np.asarray(net_cf)
                    # Floor at 0 (no negative asset values in simple model)
                    portfolio_values[:, ai] = np.maximum(portfolio_values[:, ai], 0)

        total_portfolio_updated = portfolio_values.sum(axis=1)

        # ── Die with Zero / Legacy adjustments ────────────────────────────────
        if profile.simulation_mode == "DieWithZero" and age >= ret_age:
            years_left = max(1, profile.lifespan - age)
            # Encourage higher spending to reach ~0
            extra_spend = total_portfolio_updated / years_left
            # Apply as additional withdrawal from liquid assets
            extra_per_sim = extra_spend * 0.1  # gradual adjustment
            for ai in range(n_investable_assets):
                if liquid_mask[ai] > 0:
                    weight = portfolio_values[:, ai] / np.where(total_liquid > 0, total_liquid, 1e-9)
                    portfolio_values[:, ai] -= weight * extra_per_sim
                    portfolio_values[:, ai] = np.maximum(portfolio_values[:, ai], 0)

        # ── Record ────────────────────────────────────────────────────────────
        final_total = portfolio_values.sum(axis=1) + cpf_total_arr[yr]
        net_worth_paths[:, yr] = final_total

        # CPF not modeled stochastically (guaranteed rates)
        cash_flow_paths[:, yr] = total_income - np.mean(np.asarray(expenses))

        # Track bankruptcy
        for sim_i in range(n_sims):
            if bankruptcy_ages[sim_i] is None and final_total[sim_i] <= 0:
                bankruptcy_ages[sim_i] = age

    # ── Statistics ───────────────────────────────────────────────────────────
    final_nw = net_worth_paths[:, -1]
    success_rate = np.mean(final_nw > profile.legacy_target) * 100
    bankruptcy_rate = 100 - success_rate

    bk_ages_numeric = [a for a in bankruptcy_ages if a is not None]
    median_bk_age = float(np.median(bk_ages_numeric)) if bk_ages_numeric else None

    return SimulationResult(
        n_simulations=n_sims,
        n_years=n_years,
        ages=ages,
        net_worth_paths=net_worth_paths,
        cash_flow_paths=cash_flow_paths,
        cpf_total_paths=np.tile(cpf_total_arr, (n_sims, 1)),
        success_rate=success_rate,
        bankruptcy_rate=bankruptcy_rate,
        median_final_nw=float(np.median(final_nw)),
        p10_final_nw=float(np.percentile(final_nw, 10)),
        p25_final_nw=float(np.percentile(final_nw, 25)),
        p75_final_nw=float(np.percentile(final_nw, 75)),
        p90_final_nw=float(np.percentile(final_nw, 90)),
        median_path=np.median(net_worth_paths, axis=0),
        p10_path=np.percentile(net_worth_paths, 10, axis=0),
        p25_path=np.percentile(net_worth_paths, 25, axis=0),
        p75_path=np.percentile(net_worth_paths, 75, axis=0),
        p90_path=np.percentile(net_worth_paths, 90, axis=0),
        best_path=net_worth_paths[np.argmax(final_nw)],
        worst_path=net_worth_paths[np.argmin(final_nw)],
        bankruptcy_ages=bankruptcy_ages,
        median_bankruptcy_age=median_bk_age,
    )
