"""
Historical annual returns data (nominal, in local/SGD terms where applicable).
Used for Historical Simulation mode.
Sources: STI, S&P 500, MSCI EM, Bloomberg Barclays indices, MAS data.
All values are decimal (e.g., 0.10 = 10%).
"""

import numpy as np

# Years covered
HIST_YEARS = list(range(1990, 2025))

# Singapore Equities (STI Total Return approximation)
HIST_SG_EQUITIES = [
    -0.12, 0.22, 0.03, 0.48, -0.03, 0.05, -0.02, -0.26, -0.08, 0.55,
    -0.17, -0.14, -0.17, 0.33, 0.19, 0.14, 0.27, 0.18, -0.49, 0.64,
    0.10, -0.17, 0.20, -0.01, 0.06, -0.14, -0.01, 0.18, -0.10, 0.05,
    -0.09, 0.10, -0.04, -0.01, 0.04,
]

# US Equities (S&P 500 Total Return, SGD-adjusted approx)
HIST_US_EQUITIES = [
    -0.03, 0.30, 0.07, 0.10, 0.01, 0.37, 0.23, 0.33, 0.28, 0.21,
    -0.09, -0.12, -0.22, 0.28, 0.11, 0.05, 0.16, 0.05, -0.37, 0.26,
    0.15, 0.02, 0.16, 0.32, 0.14, 0.01, 0.12, 0.22, -0.04, 0.31,
    0.18, 0.29, -0.18, 0.26, 0.25,
]

# Emerging Markets Equities (MSCI EM, SGD-adjusted approx)
HIST_EM_EQUITIES = [
    0.10, 0.60, 0.11, 0.75, -0.07, -0.05, 0.06, -0.12, -0.25, 0.66,
    -0.31, -0.02, -0.06, 0.56, 0.26, 0.34, 0.33, 0.39, -0.53, 0.79,
    0.19, -0.18, 0.18, -0.02, -0.02, -0.15, 0.12, 0.37, -0.15, 0.18,
    0.18, -0.03, -0.20, 0.10, 0.08,
]

# Singapore Bonds (SGS/SSB proxy)
HIST_SG_BONDS = [
    0.04, 0.05, 0.06, 0.06, 0.04, 0.05, 0.05, 0.04, 0.04, 0.03,
    0.04, 0.04, 0.04, 0.03, 0.03, 0.03, 0.03, 0.02, 0.05, 0.03,
    0.02, 0.02, 0.02, 0.02, 0.03, 0.03, 0.03, 0.02, 0.03, 0.02,
    0.01, -0.02, -0.05, 0.04, 0.04,
]

# Global Bonds (Bloomberg Global Agg, SGD-adjusted approx)
HIST_GLOBAL_BONDS = [
    0.03, 0.06, 0.05, 0.08, 0.02, 0.07, 0.04, 0.04, 0.05, -0.01,
    0.03, 0.05, 0.06, 0.02, 0.04, 0.01, 0.04, 0.02, 0.06, 0.05,
    0.03, 0.05, 0.04, -0.02, 0.01, -0.03, 0.03, 0.02, -0.01, 0.06,
    0.03, -0.02, -0.13, 0.05, 0.03,
]

# Cash / T-Bills (SGD)
HIST_CASH = [
    0.04, 0.04, 0.03, 0.03, 0.03, 0.03, 0.03, 0.02, 0.02, 0.02,
    0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
    0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.03, 0.03,
    0.03, 0.02, 0.03, 0.04, 0.03,
]

# Singapore CPI Inflation (annual)
HIST_INFLATION = [
    0.034, 0.034, 0.023, 0.023, 0.031, 0.017, 0.014, 0.020, -0.003, 0.000,
    0.013, 0.010, -0.004, 0.005, 0.017, 0.005, 0.010, 0.021, 0.065, 0.006,
    0.028, 0.052, 0.045, 0.024, 0.010, -0.005, -0.005, 0.006, 0.004, 0.006,
    -0.002, 0.023, 0.061, 0.049, 0.024,
]

# Map asset types to historical series
from src.simulation.assets import AssetType

HIST_RETURNS_MAP = {
    AssetType.CASH: HIST_CASH,
    AssetType.EQUITIES_SG: HIST_SG_EQUITIES,
    AssetType.EQUITIES_US: HIST_US_EQUITIES,
    AssetType.EQUITIES_EM: HIST_EM_EQUITIES,
    AssetType.EQUITIES_GLOBAL: HIST_US_EQUITIES,  # proxy
    AssetType.BONDS_SG: HIST_SG_BONDS,
    AssetType.BONDS_GLOBAL: HIST_GLOBAL_BONDS,
    AssetType.ETFS: HIST_SG_EQUITIES,  # proxy (blended)
    AssetType.REAL_ESTATE_PRIMARY: HIST_SG_BONDS,  # low-vol proxy
    AssetType.REAL_ESTATE_INVESTMENT: HIST_SG_BONDS,  # low-vol proxy
    AssetType.ANNUITY: HIST_SG_BONDS,
    AssetType.BUSINESS: HIST_SG_EQUITIES,  # high-vol proxy
    AssetType.CRYPTO: HIST_US_EQUITIES,  # no long history; proxy
    AssetType.OTHER: HIST_SG_BONDS,
}

# Default correlation matrix for main asset classes
# Order: SG_EQ, US_EQ, EM_EQ, GLOBAL_EQ, SG_BOND, GLOBAL_BOND, CASH, RE, CRYPTO
CORRELATION_LABELS = [
    AssetType.EQUITIES_SG,
    AssetType.EQUITIES_US,
    AssetType.EQUITIES_EM,
    AssetType.EQUITIES_GLOBAL,
    AssetType.BONDS_SG,
    AssetType.BONDS_GLOBAL,
    AssetType.CASH,
    AssetType.REAL_ESTATE_PRIMARY,
    AssetType.REAL_ESTATE_INVESTMENT,
    AssetType.CRYPTO,
]

DEFAULT_CORRELATION = np.array([
    # SG_EQ  US_EQ  EM_EQ  GL_EQ  SG_BD  GL_BD  CASH   RE_P   RE_I   CRYPTO
    [ 1.00,  0.65,  0.70,  0.65,  0.10, -0.10,  0.05,  0.30,  0.35,  0.25],  # SG_EQ
    [ 0.65,  1.00,  0.60,  0.95,  0.05, -0.15,  0.05,  0.20,  0.25,  0.30],  # US_EQ
    [ 0.70,  0.60,  1.00,  0.65,  0.05, -0.10,  0.05,  0.20,  0.25,  0.25],  # EM_EQ
    [ 0.65,  0.95,  0.65,  1.00,  0.05, -0.15,  0.05,  0.20,  0.25,  0.30],  # GL_EQ
    [ 0.10,  0.05,  0.05,  0.05,  1.00,  0.60,  0.30, -0.05, -0.05, -0.10],  # SG_BD
    [-0.10, -0.15, -0.10, -0.15,  0.60,  1.00,  0.20, -0.10, -0.10, -0.15],  # GL_BD
    [ 0.05,  0.05,  0.05,  0.05,  0.30,  0.20,  1.00,  0.00,  0.00,  0.00],  # CASH
    [ 0.30,  0.20,  0.20,  0.20, -0.05, -0.10,  0.00,  1.00,  0.85,  0.10],  # RE_P
    [ 0.35,  0.25,  0.25,  0.25, -0.05, -0.10,  0.00,  0.85,  1.00,  0.10],  # RE_I
    [ 0.25,  0.30,  0.25,  0.30, -0.10, -0.15,  0.00,  0.10,  0.10,  1.00],  # CRYPTO
])


def get_correlation_index(asset_type: AssetType) -> int:
    """Map any asset type to a correlation matrix index."""
    if asset_type in CORRELATION_LABELS:
        return CORRELATION_LABELS.index(asset_type)
    # Map unlisted types to closest proxy
    mapping = {
        AssetType.ETFS: AssetType.EQUITIES_GLOBAL,
        AssetType.ANNUITY: AssetType.BONDS_SG,
        AssetType.BUSINESS: AssetType.EQUITIES_SG,
        AssetType.OTHER: AssetType.BONDS_SG,
    }
    proxy = mapping.get(asset_type, AssetType.EQUITIES_GLOBAL)
    return CORRELATION_LABELS.index(proxy)


def build_correlation_submatrix(asset_types: list[AssetType]) -> np.ndarray:
    """Build a correlation sub-matrix for a list of asset types."""
    n = len(asset_types)
    corr = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            idx_i = get_correlation_index(asset_types[i])
            idx_j = get_correlation_index(asset_types[j])
            c = DEFAULT_CORRELATION[idx_i, idx_j]
            corr[i, j] = c
            corr[j, i] = c
    return corr
