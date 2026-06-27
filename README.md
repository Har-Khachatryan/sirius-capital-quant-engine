# Sirius Capital — Quant-Grade ML Risk Pipeline (v3.1)

**Author:** Harutyun Arami Khachatryan  
**Stack:** Python, XGBoost, Scikit-Learn (KMeans), SciPy (SLSQP), yfinance, Pydantic v2

An enterprise-grade financial engineering pipeline that solves the customer churn problem for retail/HNW trading clients and dynamically restructures asset allocation portfolios to maximize retention utility.

---

## 🚀 Key Features

* **Investor Segmentation:** Multi-feature `KMeans` clustering mapping clients to dynamic risk profiles (*Aggressive*, *Balanced*, *Conservative*).
* **Churn Predictive Engine:** `XGBoost` classifier predicting client attrition probability using behavioral and balance velocity features.
* **Dynamic Markowitz Optimization:** Custom asset allocation engine based on Modern Portfolio Theory (MPT) using `SciPy.optimize.minimize` (SLSQP).
* **Feasibility Repair Layer (v3.1):** Advanced headroom-proportional math layer ensuring portfolio box constraints and asset ceiling boundaries (`WEIGHT_MAX = 40%`) never conflict with budget constraints ($\sum w_i = 1$).
* **Robust Covariance:** Implements `Ledoit-Wolf` shrinkage for stable annualized covariance matrices against noisy financial data.

---

## 🛠️ Architecture Blueprint

1. **Module 0:** Input Contract via strict `Pydantic v2` models.
2. **Module 1 & 2:** Synthetic financial environment generator & Dynamic profile resolver.
3. **Module 3:** Offline training pipeline targeting Churn AUC-ROC maximization.
4. **Module 4 & 5:** Quant portfolio optimization engine backed by thread-safe `yfinance` TTL caching & production API gateway simulator.

---

## 📦 Installation & Setup

1. Clone the repository:
   ```bash
   git clone [https://github.com/Har-Khachatryan/sirius-capital-quant-engine.git](https://github.com/YOUR_USERNAME/sirius-capital-quant-engine.git)
   cd sirius-capital-quant-engine

   
## 📊 Production Execution Output (v3.1)

```text
╔══════════════════════════════════════════════════════════════════╗
║         SIRIUS CAPITAL — QUANT GRADE ENGINE  v3.1                ║
╚══════════════════════════════════════════════════════════════════╝
🚀  Executing Production Batch (Full Cycle Context)...

══════════════════════════════════════════════════════════════════════════════════════════
🔹  Client #9001  [Crypto Enthusiast — Aggressive (58% Churn)]
    Profile : AGGRESSIVE      |  Churn Risk : 🟡 58.73%
    Strategy: Dynamic Retention — AGGRESSIVE (Risk Scale: 0.59)
      • AAPL  ██                    5.00%  →  $    2,100.00
      • MSFT  ██                    5.00%  →  $    2,100.00
      ...
🔹  Client #9005  [Conservative Senior — Sudden Outflow]
    Profile : CONSERVATIVE    |  Churn Risk : 🔴 87.97%
    Strategy: Dynamic Retention — CONSERVATIVE (Risk Scale: 0.88)
      • AAPL  █████                12.84%  →  $   14,128.44
      • MSFT  █████                12.84%  →  $   14,128.44
      • KO    █████████████████    52.00%  →  $   57,200.00  ✓ [Feasibility Repaired]
      ...