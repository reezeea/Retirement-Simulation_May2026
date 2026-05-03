# 🇸🇬 Singapore Retirement Simulator

A production-grade Monte Carlo retirement planning tool specifically built for Singapore citizens, modelling CPF, local inflation, cost of living, and typical SG financial scenarios.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## ✨ Features

### Core Simulation Engine
- **Monte Carlo** with 1,000–10,000 simulation paths
- **Fat-tail returns** (Student-t distribution) for realistic crash modelling
- **Per-asset return/volatility modelling** with inflation adjustment
- **Sequence-of-returns risk** automatically modelled

### Singapore-Specific
- ✅ Full CPF model: OA / SA / MA / RA with accurate contribution rates by age
- ✅ CPF allocation rates by age (35 / 45 / 50 / 55 / 60 / 65+)
- ✅ CPF LIFE payout estimation (Basic / Standard / Escalating)
- ✅ Retirement Account formation at age 55 (BRS / FRS / ERS)
- ✅ Extra interest modelling (1% on first $60k combined)
- ✅ Singapore inflation default (2.5%)

### Asset Classes Supported
| Asset | Default Return | Default Volatility |
|---|---|---|
| Cash | 2.5% | 0.5% |
| Singapore Equities (STI) | 6.0% | 15% |
| US Equities (S&P 500) | 8.0% | 16% |
| Emerging Markets | 7.0% | 22% |
| Singapore Bonds / SSB | 3.5% | 5% |
| Investment Property | 5.0% (incl. yield) | 8% |
| CPF OA | 2.5% (guaranteed) | 0% |
| CPF SA | 4.0% (guaranteed) | 0% |
| Cryptocurrency | 15% | 70% |

All defaults are fully editable.

### Simulation Modes
- **Standard** – Maximize probability of not running out of money
- **Die With Zero** – Optimize withdrawals to spend down to ~$0 at death
- **Legacy** – Maintain a target inheritance amount

### Withdrawal Strategies
- **Fixed** – Constant real spending
- **% of Portfolio** – Dynamic (e.g. 4% rule)
- **Guardrail** – Reduce spending in downturns, increase in bull markets

### Outputs
- 📊 Net worth fan chart (P10/P25/Median/P75/P90 bands)
- 📈 CPF account trajectory with CPF LIFE payout overlay
- 🎯 Retirement success probability (%)
- 📅 Year-by-year table
- 🔬 Scenario comparison (Pessimistic / Base / Optimistic)
- 📉 Retirement age sensitivity analysis
- 📊 Final net worth distribution histogram
- 📥 CSV + text report export

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/sg-retirement-simulator.git
cd sg-retirement-simulator
```

### 2. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

The app will open at **http://localhost:8501**

---

## 🗂️ Project Structure

```
sg-retirement-simulator/
├── app.py                          # Main Streamlit application
├── requirements.txt
├── README.md
├── src/
│   ├── simulation/
│   │   ├── assets.py               # Asset classes & default assumptions
│   │   ├── profile.py              # User profile data model
│   │   └── monte_carlo.py          # Monte Carlo simulation engine
│   ├── cpf/
│   │   └── cpf_model.py            # Singapore CPF system model
│   └── utils/
│       ├── export.py               # CSV / text report export
│       └── presets.py              # Preset Singapore profiles
└── tests/
    └── test_simulation.py          # Unit & smoke tests
```

---

## 🧪 Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

---

## 📋 Preset Profiles

Load these from the UI to get started quickly:

| Profile | Description |
|---|---|
| **Average Singaporean** | HDB owner, age 35, married with 2 kids, CPF OA/SA/MA pre-filled |
| **High-Income Professional** | Private property, dual income, legacy planning |
| **FIRE – Early Retirement** | Age 32, target retirement 45, guardrail strategy, Die With Zero mode |

---

## ⚙️ CPF Data Sources

- Contribution rates: [CPF Board](https://www.cpf.gov.sg/employer/employer-obligations/how-much-cpf-contributions-to-pay)
- Interest rates: [CPF Interest Rates](https://www.cpf.gov.sg/member/growing-your-savings/earning-higher-returns/earning-attractive-interest)
- CPF LIFE: [CPF LIFE Estimator](https://www.cpf.gov.sg/member/retirement-income/monthly-payouts/cpf-life)
- Retirement sums: [CPF Retirement Sums](https://www.cpf.gov.sg/member/retirement-income/retirement-sum-scheme)

---

## ⚠️ Disclaimer

This tool is for **educational and planning purposes only**. It does not constitute financial advice. CPF rules, returns, and regulations may change. Consult a licensed financial adviser for personalised advice.

---

## 📄 License

MIT License. See [LICENSE](LICENSE).
