"""
Asset class definitions, default return/volatility assumptions,
and portfolio modeling for Singapore Retirement Simulator.
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class AssetType(str, Enum):
    CASH = "Cash"
    EQUITIES_SG = "Singapore Equities"
    EQUITIES_US = "US Equities"
    EQUITIES_EM = "Emerging Markets Equities"
    EQUITIES_GLOBAL = "Global Equities (ex-SG)"
    BONDS_SG = "Singapore Bonds"
    BONDS_GLOBAL = "Global Bonds"
    ETFS = "ETFs / Unit Trusts"
    REAL_ESTATE_PRIMARY = "Primary Residence"
    REAL_ESTATE_INVESTMENT = "Investment Property"
    CPF_OA = "CPF – Ordinary Account"
    CPF_SA = "CPF – Special Account"
    CPF_MA = "CPF – MediSave Account"
    ANNUITY = "Private Annuity / Pension"
    BUSINESS = "Business Ownership"
    CRYPTO = "Cryptocurrency"
    OTHER = "Other"


class Liquidity(str, Enum):
    LIQUID = "Liquid"
    SEMI_LIQUID = "Semi-Liquid"
    ILLIQUID = "Illiquid"


# ── Default Assumptions ───────────────────────────────────────────────────────
# (expected_return_pct, volatility_pct, liquidity)
ASSET_DEFAULTS: dict[AssetType, dict] = {
    AssetType.CASH: {
        "return": 2.5, "volatility": 0.5, "liquidity": Liquidity.LIQUID,
        "note": "Singapore savings rate / T-bills ~2–3%",
    },
    AssetType.EQUITIES_SG: {
        "return": 6.0, "volatility": 15.0, "liquidity": Liquidity.LIQUID,
        "note": "STI historical ~6–7% nominal",
    },
    AssetType.EQUITIES_US: {
        "return": 8.0, "volatility": 16.0, "liquidity": Liquidity.LIQUID,
        "note": "S&P 500 historical ~8–10% nominal USD; SGD-adjusted ~7–9%",
    },
    AssetType.EQUITIES_EM: {
        "return": 7.0, "volatility": 22.0, "liquidity": Liquidity.LIQUID,
        "note": "MSCI EM historical ~7% with high vol",
    },
    AssetType.EQUITIES_GLOBAL: {
        "return": 7.5, "volatility": 15.0, "liquidity": Liquidity.LIQUID,
        "note": "MSCI World ex-SG",
    },
    AssetType.BONDS_SG: {
        "return": 3.5, "volatility": 5.0, "liquidity": Liquidity.LIQUID,
        "note": "SGS bonds / SSB ~3–4%",
    },
    AssetType.BONDS_GLOBAL: {
        "return": 3.0, "volatility": 7.0, "liquidity": Liquidity.LIQUID,
        "note": "Global bond aggregate",
    },
    AssetType.ETFS: {
        "return": 6.5, "volatility": 14.0, "liquidity": Liquidity.LIQUID,
        "note": "Depends on underlying; default blended",
    },
    AssetType.REAL_ESTATE_PRIMARY: {
        "return": 3.5, "volatility": 8.0, "liquidity": Liquidity.ILLIQUID,
        "note": "SG residential appreciation ~3–4% p.a.; no rental income (primary)",
    },
    AssetType.REAL_ESTATE_INVESTMENT: {
        "return": 5.0, "volatility": 8.0, "liquidity": Liquidity.ILLIQUID,
        "note": "~3% rental yield + ~2% appreciation; net of expenses",
    },
    AssetType.CPF_OA: {
        "return": 2.5, "volatility": 0.0, "liquidity": Liquidity.SEMI_LIQUID,
        "note": "CPF OA guaranteed 2.5% p.a.",
    },
    AssetType.CPF_SA: {
        "return": 4.0, "volatility": 0.0, "liquidity": Liquidity.ILLIQUID,
        "note": "CPF SA guaranteed 4.0% p.a.",
    },
    AssetType.CPF_MA: {
        "return": 4.0, "volatility": 0.0, "liquidity": Liquidity.ILLIQUID,
        "note": "CPF MA guaranteed 4.0% p.a.",
    },
    AssetType.ANNUITY: {
        "return": 3.5, "volatility": 0.0, "liquidity": Liquidity.ILLIQUID,
        "note": "Private pension / endowment",
    },
    AssetType.BUSINESS: {
        "return": 10.0, "volatility": 30.0, "liquidity": Liquidity.ILLIQUID,
        "note": "High risk, high return; illiquid",
    },
    AssetType.CRYPTO: {
        "return": 15.0, "volatility": 70.0, "liquidity": Liquidity.LIQUID,
        "note": "Very high risk. Fat-tailed distribution strongly recommended.",
    },
    AssetType.OTHER: {
        "return": 4.0, "volatility": 10.0, "liquidity": Liquidity.SEMI_LIQUID,
        "note": "User-defined asset",
    },
}


@dataclass
class Asset:
    """Represents a single asset in the portfolio."""
    name: str
    asset_type: AssetType
    current_value: float                    # SGD
    expected_return: float                  # % p.a.
    volatility: float                       # % p.a. (std dev)
    liquidity: Liquidity
    currency: str = "SGD"
    region: str = "Singapore"
    rental_yield: float = 0.0              # % for investment property
    notes: str = ""

    @property
    def expected_return_decimal(self) -> float:
        return self.expected_return / 100

    @property
    def volatility_decimal(self) -> float:
        return self.volatility / 100


@dataclass
class Liability:
    """Represents a liability."""
    name: str
    balance: float              # Outstanding balance SGD
    interest_rate: float        # % p.a.
    monthly_payment: float      # SGD
    tenure_years: int           # Remaining years
    liability_type: str = "Mortgage"


@dataclass
class Portfolio:
    """Full portfolio of assets and liabilities."""
    assets: list[Asset] = field(default_factory=list)
    liabilities: list[Liability] = field(default_factory=list)

    def total_assets(self) -> float:
        return sum(a.current_value for a in self.assets)

    def total_liabilities(self) -> float:
        return sum(l.balance for l in self.liabilities)

    def net_worth(self) -> float:
        return self.total_assets() - self.total_liabilities()

    def liquid_assets(self) -> float:
        return sum(
            a.current_value for a in self.assets
            if a.liquidity == Liquidity.LIQUID
        )

    def weighted_return(self) -> float:
        """Weighted average expected return across all assets."""
        total = self.total_assets()
        if total == 0:
            return 0.0
        return sum(a.current_value * a.expected_return_decimal for a in self.assets) / total

    def weighted_volatility(self) -> float:
        """Weighted average volatility (simplified, no correlation)."""
        total = self.total_assets()
        if total == 0:
            return 0.0
        return sum(a.current_value * a.volatility_decimal for a in self.assets) / total

    def annual_liability_payments(self) -> float:
        return sum(l.monthly_payment * 12 for l in self.liabilities)
