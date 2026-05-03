"""
Unit tests for Singapore Retirement Simulator.
Run with: python -m pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from src.cpf.cpf_model import (
    get_contribution_rate, get_allocation, compute_annual_cpf_contribution,
    apply_cpf_interest, form_retirement_account, estimate_cpflife_monthly_payout,
    CPFAccounts, CPF_FRS_2024,
)
from src.simulation.assets import Asset, AssetType, Liquidity, Portfolio, ASSET_DEFAULTS
from src.simulation.profile import UserProfile
from src.simulation.monte_carlo import run_monte_carlo
from src.utils.presets import preset_average_singaporean, preset_fire_early_retirement


class TestCPFContributions:
    def test_age_35_rates(self):
        emp, empr = get_contribution_rate(35)
        assert emp == pytest.approx(0.20)
        assert empr == pytest.approx(0.17)

    def test_age_55_rates(self):
        emp, empr = get_contribution_rate(55)
        assert emp == pytest.approx(0.15)
        assert empr == pytest.approx(0.13)

    def test_age_70_rates(self):
        emp, empr = get_contribution_rate(70)
        assert emp == pytest.approx(0.05)
        assert empr == pytest.approx(0.05)

    def test_income_cap(self):
        # Income above ceiling should be capped
        total, oa, sa, ma = compute_annual_cpf_contribution(500_000, 35)
        total_capped, oa_capped, sa_capped, ma_capped = compute_annual_cpf_contribution(102_000, 35)
        assert total == pytest.approx(total_capped, rel=0.01)

    def test_allocation_sums_to_one(self):
        for age in [25, 35, 45, 55, 60, 65, 70]:
            oa, sa, ma = get_allocation(age)
            assert abs(oa + sa + ma - 1.0) < 1e-6, f"Allocation doesn't sum to 1 at age {age}"

    def test_contribution_splits(self):
        total, oa, sa, ma = compute_annual_cpf_contribution(72_000, 35)
        assert abs(oa + sa + ma - total) < 1.0  # rounding tolerance


class TestCPFInterest:
    def test_oa_interest(self):
        cpf = CPFAccounts(oa=100_000, sa=0, ma=0)
        cpf = apply_cpf_interest(cpf)
        # OA: 2.5% base + 1% extra on first $20k OA
        expected_min = 100_000 * 1.025
        assert cpf.oa >= expected_min

    def test_sa_interest(self):
        cpf = CPFAccounts(oa=0, sa=50_000, ma=0)
        cpf = apply_cpf_interest(cpf)
        assert cpf.sa >= 50_000 * 1.04

    def test_retirement_account_formation(self):
        cpf = CPFAccounts(sa=300_000, oa=50_000)
        cpf = form_retirement_account(cpf, year_index=0, retirement_sum="FRS")
        assert cpf.ra_formed
        assert cpf.ra >= CPF_FRS_2024 * 0.99  # at least FRS
        assert cpf.ra <= CPF_FRS_2024 * 1.01


class TestCPFLifePayout:
    def test_payout_positive(self):
        payout = estimate_cpflife_monthly_payout(205_800, "Standard")
        assert payout > 0

    def test_escalating_lower_than_standard_initially(self):
        standard = estimate_cpflife_monthly_payout(205_800, "Standard")
        escalating = estimate_cpflife_monthly_payout(205_800, "Escalating")
        assert escalating < standard


class TestPortfolio:
    def test_net_worth(self):
        from src.simulation.assets import Liability
        p = Portfolio(
            assets=[Asset("Cash", AssetType.CASH, 100_000, 2.5, 0.5, Liquidity.LIQUID)],
            liabilities=[Liability("Loan", 50_000, 3.0, 1_000, 5)],
        )
        assert p.net_worth() == 50_000

    def test_weighted_return(self):
        assets = [
            Asset("A", AssetType.CASH, 50_000, 2.0, 0.0, Liquidity.LIQUID),
            Asset("B", AssetType.EQUITIES_US, 50_000, 8.0, 16.0, Liquidity.LIQUID),
        ]
        p = Portfolio(assets=assets)
        assert p.weighted_return() == pytest.approx(0.05)  # (2+8)/2 / 100


class TestMonteCarloSmoke:
    """Smoke tests to ensure simulation runs without errors."""

    def test_average_sg_preset(self):
        profile = preset_average_singaporean()
        profile.n_simulations = 200  # Fast for testing
        result = run_monte_carlo(profile)
        assert result.n_simulations == 200
        assert 0 <= result.success_rate <= 100
        assert result.net_worth_paths.shape == (200, profile.total_years())

    def test_fire_preset(self):
        profile = preset_fire_early_retirement()
        profile.n_simulations = 200
        result = run_monte_carlo(profile)
        assert result.success_rate >= 0
        assert result.net_worth_paths.shape[0] == 200

    def test_empty_portfolio(self):
        profile = UserProfile(
            current_age=40, retirement_age=65, lifespan=85,
            annual_expenses_pre_retirement=60_000,
            annual_expenses_post_retirement=48_000,
            n_simulations=100,
        )
        result = run_monte_carlo(profile)
        # With no assets, should show 0 or negative NW
        assert result.median_final_nw <= 0

    def test_percentile_ordering(self):
        profile = preset_average_singaporean()
        profile.n_simulations = 500
        result = run_monte_carlo(profile)
        # P10 <= P25 <= Median <= P75 <= P90
        assert result.p10_final_nw <= result.p25_final_nw
        assert result.p25_final_nw <= result.median_final_nw
        assert result.median_final_nw <= result.p75_final_nw
        assert result.p75_final_nw <= result.p90_final_nw


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
