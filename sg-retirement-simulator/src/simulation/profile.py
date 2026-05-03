"""
User profile data model for the Singapore Retirement Simulator.
"""

from dataclasses import dataclass, field
from typing import Optional
from src.simulation.assets import Portfolio
from src.cpf.cpf_model import CPFAccounts


@dataclass
class OneOffExpense:
    """A one-time or irregular future expense."""
    name: str
    amount: float       # SGD
    age_due: int        # Age at which this occurs
    inflation_adjusted: bool = True


@dataclass
class RetirementIncome:
    """Post-retirement income source."""
    name: str
    annual_amount: float    # SGD
    start_age: int
    end_age: int            # Use 999 for lifetime
    inflation_adjusted: bool = True


@dataclass
class UserProfile:
    """Complete user profile for simulation."""

    # ── Personal ─────────────────────────────────────────────────────────────
    current_age: int = 35
    retirement_age: int = 65
    lifespan: int = 90
    gender: str = "Male"              # Male / Female
    marital_status: str = "Single"    # Single / Married / Divorced
    num_dependents: int = 0
    expected_inheritance: float = 0.0
    inheritance_age: int = 60         # Age when inheritance arrives
    annual_obligations: float = 0.0   # Supporting parents etc. SGD/yr

    # ── Income ───────────────────────────────────────────────────────────────
    current_annual_income: float = 0.0
    income_growth_rate: float = 3.0     # % p.a.
    spouse_annual_income: float = 0.0
    spouse_income_growth_rate: float = 3.0

    # ── Expenses ─────────────────────────────────────────────────────────────
    annual_expenses_pre_retirement: float = 0.0
    annual_expenses_post_retirement: float = 0.0
    inflation_rate: float = 2.5         # % p.a.

    # ── One-off / Irregular Expenses ─────────────────────────────────────────
    one_off_expenses: list[OneOffExpense] = field(default_factory=list)

    # ── Post-retirement Income ────────────────────────────────────────────────
    retirement_incomes: list[RetirementIncome] = field(default_factory=list)

    # ── Portfolio ─────────────────────────────────────────────────────────────
    portfolio: Portfolio = field(default_factory=Portfolio)

    # ── CPF ──────────────────────────────────────────────────────────────────
    cpf_accounts: CPFAccounts = field(default_factory=CPFAccounts)
    cpf_retirement_sum: str = "FRS"          # BRS / FRS / ERS
    cpflife_payout_plan: str = "Standard"    # Basic / Standard / Escalating
    cpflife_payout_age: int = 65

    # ── Strategy ─────────────────────────────────────────────────────────────
    withdrawal_strategy: str = "Fixed"       # Fixed / PercentPortfolio / Guardrail
    withdrawal_percent: float = 4.0          # For % strategy
    guardrail_upper: float = 5.5             # % – increase spending
    guardrail_lower: float = 4.0             # % – reduce spending
    simulation_mode: str = "Standard"       # Standard / DieWithZero / Legacy
    legacy_target: float = 0.0              # For Legacy mode

    # ── Simulation Settings ──────────────────────────────────────────────────
    n_simulations: int = 5000
    use_fat_tails: bool = True

    def years_to_retirement(self) -> int:
        return max(0, self.retirement_age - self.current_age)

    def retirement_duration(self) -> int:
        return max(0, self.lifespan - self.retirement_age)

    def total_years(self) -> int:
        return max(1, self.lifespan - self.current_age)

    def annual_income_series(self, years: int) -> list[float]:
        """Project income series pre-retirement."""
        series = []
        for i in range(years):
            age = self.current_age + i
            if age < self.retirement_age:
                income = self.current_annual_income * ((1 + self.income_growth_rate / 100) ** i)
                if self.marital_status == "Married":
                    spouse = self.spouse_annual_income * ((1 + self.spouse_income_growth_rate / 100) ** i)
                    income += spouse
            else:
                income = 0.0
            series.append(income)
        return series
