"""
Preset user profiles for Singapore retirement scenarios.
"""

from src.simulation.profile import UserProfile, OneOffExpense, RetirementIncome
from src.simulation.assets import Asset, AssetType, Liquidity, Portfolio, ASSET_DEFAULTS
from src.cpf.cpf_model import CPFAccounts


def make_asset(asset_type: AssetType, value: float, name: str = None) -> Asset:
    defaults = ASSET_DEFAULTS[asset_type]
    return Asset(
        name=name or asset_type.value,
        asset_type=asset_type,
        current_value=value,
        expected_return=defaults["return"],
        volatility=defaults["volatility"],
        liquidity=defaults["liquidity"],
    )


def preset_average_singaporean() -> UserProfile:
    """Average Singaporean HDB owner, mid-career."""
    assets = [
        make_asset(AssetType.CASH, 30_000, "Emergency Fund"),
        make_asset(AssetType.EQUITIES_SG, 15_000, "STI ETF"),
        make_asset(AssetType.REAL_ESTATE_PRIMARY, 450_000, "HDB Flat"),
        make_asset(AssetType.BONDS_SG, 10_000, "Singapore Savings Bonds"),
    ]
    liabilities = []
    cpf = CPFAccounts(oa=85_000, sa=35_000, ma=20_000)
    return UserProfile(
        current_age=35,
        retirement_age=65,
        lifespan=85,
        gender="Male",
        marital_status="Married",
        num_dependents=2,
        current_annual_income=72_000,
        spouse_annual_income=54_000,
        income_growth_rate=3.0,
        annual_expenses_pre_retirement=72_000,
        annual_expenses_post_retirement=48_000,
        inflation_rate=2.5,
        portfolio=Portfolio(assets=assets, liabilities=liabilities),
        cpf_accounts=cpf,
        cpf_retirement_sum="FRS",
        cpflife_payout_plan="Standard",
        one_off_expenses=[
            OneOffExpense("Child 1 University", 80_000, age_due=57),
            OneOffExpense("Child 2 University", 80_000, age_due=60),
            OneOffExpense("Home Renovation", 40_000, age_due=45),
        ],
        retirement_incomes=[
            RetirementIncome("Part-time Work", 12_000, start_age=65, end_age=72),
        ],
        n_simulations=10000,
        simulation_type="Standard",
    )


def preset_high_income_professional() -> UserProfile:
    """High-income professional, private property owner."""
    assets = [
        make_asset(AssetType.CASH, 150_000, "Cash / Money Market"),
        make_asset(AssetType.EQUITIES_US, 200_000, "S&P 500 ETF"),
        make_asset(AssetType.EQUITIES_SG, 80_000, "Singapore Equities"),
        make_asset(AssetType.EQUITIES_EM, 50_000, "Emerging Markets ETF"),
        make_asset(AssetType.BONDS_SG, 100_000, "SGS Bonds"),
        make_asset(AssetType.REAL_ESTATE_PRIMARY, 2_000_000, "Private Condo"),
        make_asset(AssetType.REAL_ESTATE_INVESTMENT, 800_000, "Investment Property"),
    ]
    cpf = CPFAccounts(oa=120_000, sa=80_000, ma=60_000)
    return UserProfile(
        current_age=40,
        retirement_age=60,
        lifespan=90,
        gender="Female",
        marital_status="Married",
        num_dependents=2,
        current_annual_income=240_000,
        spouse_annual_income=180_000,
        income_growth_rate=4.0,
        annual_expenses_pre_retirement=180_000,
        annual_expenses_post_retirement=120_000,
        inflation_rate=2.5,
        portfolio=Portfolio(assets=assets),
        cpf_accounts=cpf,
        cpf_retirement_sum="ERS",
        cpflife_payout_plan="Escalating",
        one_off_expenses=[
            OneOffExpense("Child 1 International School", 200_000, age_due=50),
            OneOffExpense("Child 2 International School", 200_000, age_due=53),
            OneOffExpense("Overseas Education (x2)", 300_000, age_due=56),
            OneOffExpense("Home Upgrade", 200_000, age_due=48),
        ],
        retirement_incomes=[
            RetirementIncome("Investment Property Rental", 36_000, start_age=60, end_age=85),
        ],
        legacy_target=1_000_000,
        simulation_mode="Legacy",
        n_simulations=10000,
        simulation_type="Standard",
    )


def preset_fire_early_retirement() -> UserProfile:
    """FIRE – Financial Independence, Retire Early. Age 45 target."""
    assets = [
        make_asset(AssetType.CASH, 80_000, "Cash Buffer"),
        make_asset(AssetType.EQUITIES_US, 400_000, "Global ETF (VWRA)"),
        make_asset(AssetType.EQUITIES_SG, 100_000, "STI / Dividend Stocks"),
        make_asset(AssetType.BONDS_SG, 50_000, "SSB / T-Bills"),
        make_asset(AssetType.REAL_ESTATE_PRIMARY, 600_000, "HDB / Condo"),
        make_asset(AssetType.CRYPTO, 20_000, "BTC / ETH"),
    ]
    cpf = CPFAccounts(oa=60_000, sa=90_000, ma=40_000)
    return UserProfile(
        current_age=32,
        retirement_age=45,
        lifespan=90,
        gender="Male",
        marital_status="Single",
        num_dependents=0,
        current_annual_income=120_000,
        income_growth_rate=5.0,
        annual_expenses_pre_retirement=60_000,
        annual_expenses_post_retirement=48_000,
        inflation_rate=2.5,
        portfolio=Portfolio(assets=assets),
        cpf_accounts=cpf,
        cpf_retirement_sum="FRS",
        cpflife_payout_plan="Standard",
        withdrawal_strategy="Guardrail",
        withdrawal_percent=3.5,
        guardrail_upper=5.0,
        guardrail_lower=3.0,
        simulation_mode="DieWithZero",
        n_simulations=10000,
        simulation_type="FatTail",
    )


PRESETS = {
    "Average Singaporean": preset_average_singaporean,
    "High-Income Professional": preset_high_income_professional,
    "FIRE – Early Retirement": preset_fire_early_retirement,
}
