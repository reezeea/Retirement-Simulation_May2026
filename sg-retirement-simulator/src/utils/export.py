"""
Export utilities: CSV and text report generation.
"""

import io
import csv
import pandas as pd
from datetime import datetime
from src.simulation.monte_carlo import SimulationResult
from src.simulation.profile import UserProfile


def results_to_dataframe(result: SimulationResult) -> pd.DataFrame:
    """Convert simulation results to a pandas DataFrame."""
    df = pd.DataFrame({
        "Age": result.ages[:result.n_years],
        "Net Worth - Median (SGD)": result.median_path,
        "Net Worth - P5 (SGD)": result.p5_path,
        "Net Worth - P10 (SGD)": result.p10_path,
        "Net Worth - P25 (SGD)": result.p25_path,
        "Net Worth - P75 (SGD)": result.p75_path,
        "Net Worth - P90 (SGD)": result.p90_path,
        "Net Worth - Best Case (SGD)": result.best_path,
        "Net Worth - Worst Case (SGD)": result.worst_path,
    })
    return df


def export_csv(result: SimulationResult, profile: UserProfile) -> bytes:
    """Export simulation results as CSV bytes."""
    df = results_to_dataframe(result)
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def export_summary_text(result: SimulationResult, profile: UserProfile) -> str:
    """Generate a plain text summary of results."""
    lines = [
        "=" * 60,
        "SINGAPORE RETIREMENT SIMULATION REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 60,
        "",
        "── PERSONAL PROFILE ─────────────────────────────────────",
        f"  Current Age:        {profile.current_age}",
        f"  Retirement Age:     {profile.retirement_age}",
        f"  Expected Lifespan:  {profile.lifespan}",
        f"  Marital Status:     {profile.marital_status}",
        f"  Dependents:         {profile.num_dependents}",
        "",
        "── SIMULATION SETTINGS ──────────────────────────────────",
        f"  Simulations Run:    {result.n_simulations:,}",
        f"  Years Modelled:     {result.n_years}",
        f"  Inflation Rate:     {profile.inflation_rate}%",
        f"  Withdrawal Strategy:{profile.withdrawal_strategy}",
        f"  Simulation Type:    {profile.simulation_type}",
        "",
        "── RESULTS ──────────────────────────────────────────────",
        f"  ✅ Retirement Success Rate:  {result.success_rate:.1f}%",
        f"  ❌ Failure Rate:             {result.failure_rate:.1f}%",
        f"  📊 Sustainability:           {result.sustainability_label}",
        "",
        f"  Net Worth at Death (Median): SGD {result.median_final_nw:,.0f}",
        f"  Net Worth at Death (P5):     SGD {result.p5_final_nw:,.0f}",
        f"  Net Worth at Death (P10):    SGD {result.p10_final_nw:,.0f}",
        f"  Net Worth at Death (P25):    SGD {result.p25_final_nw:,.0f}",
        f"  Net Worth at Death (P75):    SGD {result.p75_final_nw:,.0f}",
        f"  Net Worth at Death (P90):    SGD {result.p90_final_nw:,.0f}",
        "",
    ]
    if result.median_ruin_age:
        lines.append(f"  ⚠️  Median Ruin Age: {result.median_ruin_age:.0f}")
    lines += [
        "",
        "── KEY INSIGHTS ─────────────────────────────────────────",
    ]

    if result.success_rate >= 85:
        lines.append(f"  🟢 SUSTAINABLE: You can retire at {profile.retirement_age} with {result.success_rate:.0f}% confidence.")
    elif result.success_rate >= 50:
        lines.append(f"  🟡 AT RISK: Retirement at {profile.retirement_age} has {result.success_rate:.0f}% success probability.")
        lines.append(f"     Consider: increase savings, delay retirement, or reduce expenses.")
    else:
        lines.append(f"  🔴 NOT SUSTAINABLE: Only {result.success_rate:.0f}% success rate.")
        lines.append(f"     DANGER: Significant adjustments needed to avoid running out of money.")

    if result.failure_rate > 15:
        lines.append("")
        lines.append(f"  ⚠️  WARNING: {result.failure_rate:.0f}% chance of running out of money before death.")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)
