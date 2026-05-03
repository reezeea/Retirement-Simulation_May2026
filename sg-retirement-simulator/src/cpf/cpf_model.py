"""
CPF (Central Provident Fund) Model for Singapore Retirement Simulator
Accurately models CPF contribution rates, interest, and withdrawal rules.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# ── CPF Contribution Rates by Age ────────────────────────────────────────────
# Source: CPF Board (as of 2024)
CPF_CONTRIBUTION_RATES = {
    # age_ceiling: (employee_pct, employer_pct)
    35: (0.20, 0.17),
    45: (0.20, 0.17),
    50: (0.20, 0.17),
    55: (0.15, 0.13),
    60: (0.09, 0.09),
    65: (0.075, 0.075),
    70: (0.05, 0.05),
    999: (0.05, 0.05),  # 70 and above
}

# Allocation of contributions by age (fraction going to each account)
# (OA_fraction, SA_fraction, MA_fraction)
CPF_ALLOCATION = {
    35:  (0.6217, 0.1621, 0.2162),
    45:  (0.5677, 0.1891, 0.2432),
    50:  (0.5136, 0.2162, 0.2702),
    55:  (0.4055, 0.3108, 0.2837),
    60:  (0.3108, 0.0675, 0.6217),
    65:  (0.0800, 0.0000, 0.9200),
    999: (0.0800, 0.0000, 0.9200),
}

# CPF Interest Rates (annual)
CPF_INTEREST = {
    "OA": 0.025,        # 2.5% p.a.
    "SA": 0.04,         # 4.0% p.a.
    "MA": 0.04,         # 4.0% p.a.
    "RA": 0.04,         # 4.0% p.a. (Retirement Account)
    "extra_first_60k": 0.01,   # Extra 1% on first $60k combined (OA capped at $20k)
}

# CPF Annual Salary Ceiling (2024+)
CPF_SALARY_CEILING = 102_000  # Annual Ordinary Wage ceiling
CPF_MONTHLY_OW_CEILING = 6_800  # Monthly OW ceiling

# CPF Retirement Sums (2024, indexed annually ~3%)
CPF_BRS_2024 = 102_900    # Basic Retirement Sum
CPF_FRS_2024 = 205_800    # Full Retirement Sum
CPF_ERS_2024 = 308_700    # Enhanced Retirement Sum
CPF_RS_GROWTH = 0.035     # ~3.5% annual increase

# CPF LIFE Payout Plans (multiplier of FRS to monthly payout, approximate)
# These are rough approximations; actual depends on cohort and age
CPFLIFE_PAYOUT_FACTOR = {
    "Basic":     0.0026,   # ~$535/mo per $205k FRS
    "Standard":  0.0029,   # ~$600/mo per $205k FRS
    "Escalating": 0.0024,  # lower start, increases 2% p.a.
}

CPFLIFE_PAYOUT_AGE_DEFAULT = 65


@dataclass
class CPFAccounts:
    """Tracks CPF account balances."""
    oa: float = 0.0
    sa: float = 0.0
    ma: float = 0.0
    ra: float = 0.0  # Retirement Account (formed at 55)
    ra_formed: bool = False

    @property
    def total(self) -> float:
        return self.oa + self.sa + self.ma + self.ra


def get_contribution_rate(age: int) -> tuple[float, float]:
    """Return (employee_pct, employer_pct) for given age."""
    for ceiling, rates in CPF_CONTRIBUTION_RATES.items():
        if age <= ceiling:
            return rates
    return (0.05, 0.05)


def get_allocation(age: int) -> tuple[float, float, float]:
    """Return (OA_frac, SA_frac, MA_frac) for given age."""
    for ceiling, alloc in CPF_ALLOCATION.items():
        if age <= ceiling:
            return alloc
    return (0.08, 0.00, 0.92)


def compute_annual_cpf_contribution(
    annual_income: float,
    age: int,
) -> tuple[float, float, float, float]:
    """
    Compute annual CPF contributions split by account.
    Returns: (total_contribution, oa_add, sa_add, ma_add)
    """
    emp_rate, empr_rate = get_contribution_rate(age)
    # Cap at salary ceiling
    capped_income = min(annual_income, CPF_SALARY_CEILING)
    total_contrib = capped_income * (emp_rate + empr_rate)

    oa_frac, sa_frac, ma_frac = get_allocation(age)
    oa_add = total_contrib * oa_frac
    sa_add = total_contrib * sa_frac
    ma_add = total_contrib * ma_frac

    return total_contrib, oa_add, sa_add, ma_add


def apply_cpf_interest(cpf: CPFAccounts) -> CPFAccounts:
    """Apply annual CPF interest to all accounts."""
    # Base interest
    cpf.oa += cpf.oa * CPF_INTEREST["OA"]
    cpf.sa += cpf.sa * CPF_INTEREST["SA"]
    cpf.ma += cpf.ma * CPF_INTEREST["MA"]
    cpf.ra += cpf.ra * CPF_INTEREST["RA"]

    # Extra 1% on first $60k combined (OA portion capped at $20k)
    oa_eligible = min(cpf.oa, 20_000)
    remaining = max(0, 60_000 - oa_eligible)
    sa_eligible = min(cpf.sa, remaining)
    remaining -= sa_eligible
    ma_eligible = min(cpf.ma, remaining)
    remaining -= ma_eligible
    ra_eligible = min(cpf.ra, remaining)

    extra = (oa_eligible + sa_eligible + ma_eligible + ra_eligible) * CPF_INTEREST["extra_first_60k"]
    # Distribute extra interest proportionally (simplified: to OA/SA/MA/RA in order)
    cpf.oa += oa_eligible * CPF_INTEREST["extra_first_60k"]
    cpf.sa += sa_eligible * CPF_INTEREST["extra_first_60k"]
    cpf.ma += ma_eligible * CPF_INTEREST["extra_first_60k"]
    cpf.ra += ra_eligible * CPF_INTEREST["extra_first_60k"]

    return cpf


def form_retirement_account(cpf: CPFAccounts, year_index: int, retirement_sum: str = "FRS") -> CPFAccounts:
    """
    At age 55: SA + OA funds transferred to RA up to chosen retirement sum.
    """
    if cpf.ra_formed:
        return cpf

    target = {
        "BRS": CPF_BRS_2024,
        "FRS": CPF_FRS_2024,
        "ERS": CPF_ERS_2024,
    }.get(retirement_sum, CPF_FRS_2024)

    # Index target to future year
    target *= (1 + CPF_RS_GROWTH) ** year_index

    # SA first, then OA
    from_sa = min(cpf.sa, target)
    cpf.ra += from_sa
    cpf.sa -= from_sa
    needed = max(0, target - cpf.ra)
    from_oa = min(cpf.oa, needed)
    cpf.ra += from_oa
    cpf.oa -= from_oa
    cpf.ra_formed = True

    return cpf


def estimate_cpflife_monthly_payout(
    ra_balance: float,
    payout_plan: str = "Standard",
    payout_age: int = 65,
) -> float:
    """
    Estimate monthly CPF LIFE payout.
    Uses simplified multiplier approach.
    """
    factor = CPFLIFE_PAYOUT_FACTOR.get(payout_plan, CPFLIFE_PAYOUT_FACTOR["Standard"])
    monthly = ra_balance * factor
    return max(monthly, 0)


def simulate_cpf_trajectory(
    current_age: int,
    retirement_age: int,
    lifespan: int,
    annual_income_series: list[float],
    cpf_initial: CPFAccounts,
    retirement_sum: str = "FRS",
    payout_plan: str = "Standard",
    payout_age: int = 65,
    oa_for_housing: float = 0.0,  # One-off OA withdrawal for housing
) -> dict:
    """
    Simulate CPF accounts year by year from current_age to lifespan.
    Returns dict with yearly CPF data.
    """
    import copy
    cpf = copy.deepcopy(cpf_initial)
    results = []

    for i, age in enumerate(range(current_age, lifespan + 1)):
        year_data = {"age": age, "oa": cpf.oa, "sa": cpf.sa, "ma": cpf.ma, "ra": cpf.ra, "cpflife_monthly": 0.0}

        # Pre-retirement: contributions
        if age < retirement_age and i < len(annual_income_series):
            income = annual_income_series[i]
            _, oa_add, sa_add, ma_add = compute_annual_cpf_contribution(income, age)
            cpf.oa += oa_add
            cpf.sa += sa_add
            cpf.ma += ma_add

        # Housing withdrawal (one-off)
        if i == 0 and oa_for_housing > 0:
            cpf.oa = max(0, cpf.oa - oa_for_housing)

        # Age 55: form RA
        if age == 55 and not cpf.ra_formed:
            cpf = form_retirement_account(cpf, i, retirement_sum)

        # Apply interest
        cpf = apply_cpf_interest(cpf)

        # CPF LIFE payouts
        if age >= payout_age and cpf.ra > 0:
            monthly_payout = estimate_cpflife_monthly_payout(cpf.ra, payout_plan, payout_age)
            year_data["cpflife_monthly"] = monthly_payout
            # RA balance decreases (annuity; simplified: balance depletes over expected lifetime)
            # CPF LIFE is an annuity – RA doesn't actually go to zero; we model payout only
            # Keep RA notional for display; payouts come from CPF board pool

        year_data.update({"oa": cpf.oa, "sa": cpf.sa, "ma": cpf.ma, "ra": cpf.ra})
        results.append(year_data)

    return {"yearly": results, "cpf": cpf}
