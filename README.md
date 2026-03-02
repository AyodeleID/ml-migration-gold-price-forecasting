<div align="center">

# 🤖 ML in Food Economics & Agribusiness

### Applied Machine Learning — Exam Project

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![University](https://img.shields.io/badge/Uni-Göttingen-003366?style=for-the-badge&logo=academia&logoColor=white)](https://uni-goettingen.de)

<br/>

> **Two applied ML problems — one notebook.**
> Migration behaviour classification with 5 algorithms, and gold price forecasting with deep learning.
> Both evaluated through the lens of the **No-Free-Lunch Theorem**.

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Datasets](#-datasets)
- [Section 1 — Migration Prediction](#-section-1--migration-behaviour-prediction)
- [Section 2 — Gold Price Forecasting](#-section-2--gold-price-forecasting)
- [Tech Stack](#-tech-stack)
- [Key Results Summary](#-key-results-summary)
- [Project Structure](#-project-structure)
- [Author](#-author)

---

## 🔍 Overview

This project was completed as a live coding exam for the course **Machine Learning in Food Economics and Agribusiness** at the **University of Göttingen**. It covers two distinct applied ML problems:

| # | Problem | Domain | Methods |
|---|---------|--------|---------|
| 1 | **Migration Decision** | Development Economics | Logistic Regression, Decision Tree, Random Forest, XGBoost, Neural Network, SHAP |
| 2 | **Gold Price Forecasting** | Commodity Markets | CNN, RNN, LSTM, PSI Analysis |

Both sections apply the **No-Free-Lunch (NFL) Theorem** — the principle that no algorithm is universally best, and model selection must match the structure of the problem.

---

## 📦 Datasets

### Dataset 1 — Migration Household Study (Bangladesh)

> Used in: **Section 1 — Migration Behaviour Prediction**

| Field | Details |
|---|---|
| **Source** | Rana, M.S., Faye, A., & Qaim, M. (2025). *Temporary Migration Decisions and Effects on Household Income and Diets in Rural Bangladesh*. Agricultural Economics, Vol. 56, Issue 5, pp. 769–781 |
| **DOI** | [10.1111/agec.70030](https://doi.org/10.1111/agec.70030) |
| **Paper** | [Wiley Online Library](https://onlinelibrary.wiley.com/doi/10.1111/agec.70030) |
| **Data Download** | [Supporting Information — Dataset.zip](https://onlinelibrary.wiley.com/action/downloadSupplement?doi=10.1111%2Fagec.70030&file=agec70030-sup-0002-Dataset.zip) |
| **File** | `Migration_HH_Study.dta` (Stata format) |
| **Observations** | 851 households |
| **Coverage** | Northern Bangladesh — agricultural lean season panel |
| **Access** | Open Access (CC BY) |

**Key variables used:**

```
Target:   migration          →  1 = migrant household, 0 = non-migrant
Features: mig_network        →  Migrant network size (instrumental variable)
          distrust_neighbor  →  Social trust measure
          season_employ_fluc →  Seasonal employment fluctuations
          nuclear            →  Nuclear/small household dummy
          cattle_rearing     →  Livestock farming dummy
          HHage, HHedu       →  Head of household demographics
          Wealthindex        →  Asset-based wealth index
          ... + 15 more controls (see notebook)
```

---

### Dataset 2 — World Bank Commodity Price Data (Monthly)

> Used in: **Section 2 — Gold Price Forecasting**

| Field | Details |
|---|---|
| **Source** | World Bank Commodity Markets ("Pink Sheet") |
| **Page** | [worldbank.org/en/research/commodity-markets](https://www.worldbank.org/en/research/commodity-markets) |
| **Direct Download** | [CMO-Historical-Data-Monthly.xlsx](https://thedocs.worldbank.org/en/doc/74e8be41ceb20fa0da750cda2f6b9e4e-0050012026/related/CMO-Historical-Data-Monthly.xlsx) |
| **File** | `CMO-Historical-Data-Monthly.xlsx` |
| **Frequency** | Monthly |
| **Coverage** | 1960 – present (updated regularly) |
| **Target variable** | Gold price (USD/troy oz) |
| **Access** | Publicly available — World Bank Open Data |

**Context:** Global gold prices recently hit all-time highs (2024–2025), driven by geopolitical uncertainty, central bank demand, and inflation hedging — making this an economically relevant forecasting target.

```
Train period:  ~2001 – 2018  |  Gold: ~$300 – $1,300 / troy oz
Test period:   ~2018 – 2026  |  Gold: ~$1,500 – $2,700+ / troy oz
PSI > 7.2  →  Major structural break confirmed between periods
```

---

## 🧭 Section 1 — Migration Behaviour Prediction

### Problem Statement

Using the Bangladesh household survey, predict whether a household makes a migration decision. This replicates and extends the logit model from **Equation 4 / Table 3 Column 2** of Rana et al. (2025) using five ML classifiers.

**Model Specification:**

$$M_i = \omega_0 + \omega_z Z_i + \omega_{iv} IV_j + \vartheta_i$$

Where $Z_i$ = household control variables and $IV_j$ = village-level migration network (instrumental variable).

### ⚙️ Models & Performance

| Model | Train Accuracy | Test Accuracy | Overfitting? |
|---|---|---|---|
| ✅ **Logistic Regression** | 75.63% | **76.95%** | None |
| 🌳 Decision Tree | 77.65% | 74.22% | Slight |
| 🌲 Random Forest | 81.01% | 74.22% | Moderate |
| ⚡ XGBoost (GradientBoosting) | 100% | 69.53% | **Severe** |
| 🧠 Neural Network (MLP) | 100% | 68.36% | **Severe** |

> 💡 **Winner: Logistic Regression** — confirms the NFL theorem. With only 851 observations, simpler models generalise far better than complex ones.

### 🔬 SHAP Feature Importance

Top predictors consistent across **all five** algorithms:

```
1. 🔴 distrust_neighbor    →  High social distrust strongly REDUCES migration  (−34 pp)
2. 🟢 mig_network          →  Migrant network size strongly INCREASES migration (+26 pp)
3. 🔴 hh_biz_num           →  Owning a business reduces migration              (−16 pp)
4. 🟢 season_employ_fluc   →  Seasonal employment volatility increases it       (+6 pp)
5. 🔴 cattle_rearing       →  Livestock farming reduces migration               (−13 pp)
```

### 🌳 Tree Insights

- **Both** the Classification Tree and the first tree of the Random Forest root-split on `distrust_neighbor ≤ 0.5`
- Key decision path: *Low distrust → High network → Migrant* 🟠
- `hh_biz_num`, `HHage`, and `Wealthindex` appear as secondary splits

---

## 📈 Section 2 — Gold Price Forecasting

### Problem Statement

Forecast monthly gold prices ($/troy oz) using **12 lagged observations** as input features, trained on World Bank CMO historical data.

**Train/Test Split:** ~70/30 (approx. 2001–2018 train | 2018–2026 test)

### ⚙️ Deep Learning Models & Performance

| Model | Test RMSE | Test R² | Notes |
|---|---|---|---|
| 🟢 **RNN** | **161.42** | **0.949** | Best — sequential processing fits monthly dependencies |
| 🔵 CNN | 182.70 | 0.935 | Good — detects local lag patterns |
| 🔴 LSTM | 504.68 | 0.504 | Worst — too many parameters for ~202 training samples |

> 💡 **Winner: RNN** — LSTM's theoretical advantage for long-term dependencies did not materialise due to small dataset size. NFL theorem confirmed again.

### 📊 Distribution Shift (PSI Analysis)

All PSI values exceeded **7.2** (threshold = 0.25), confirming a major structural break between the training and test periods — consistent with the historic gold price regime shift of 2018–2025.

---

## 🛠 Tech Stack

```python
# Core Data & ML
pandas | numpy | scikit-learn | statsmodels

# Deep Learning
tensorflow / keras

# Explainability
shap

# Visualisation
matplotlib | seaborn
```

---

## 📊 Key Results Summary

| Section | Winner | Why |
|---|---|---|
| Migration Classification | Logistic Regression (76.95%) | Small sample (851 obs) favours simplicity |
| Gold Price Forecasting | RNN (R² = 0.949) | Sequential structure > LSTM complexity at this scale |

**NFL Theorem validated in both sections:** the winning model in each case was *not* the most complex one — it was the one whose assumptions best matched the data structure and sample size.

---

## 📁 Project Structure

```
📦 ml-migration-gold-price-forecasting
 ┣ 📓 Ayodele_Idowu_11053741_ML_Exam.ipynb      ← Main notebook
 ┣ 📊 Migration_HH_Study.dta                    ← Dataset 1 (Rana et al., 2025)
 ┣ 📊 CMO-Historical-Data-Monthly.xlsx          ← Dataset 2 (World Bank, 2025)
 ┗ 📄 README.md
```

> ⚠️ **Note on data access:**
> - `Migration_HH_Study.dta` — download from [Wiley Supporting Information](https://onlinelibrary.wiley.com/action/downloadSupplement?doi=10.1111%2Fagec.70030&file=agec70030-sup-0002-Dataset.zip) (Open Access)
> - `CMO-Historical-Data-Monthly.xlsx` — download from [World Bank Commodity Markets](https://thedocs.worldbank.org/en/doc/74e8be41ceb20fa0da750cda2f6b9e4e-0050012026/related/CMO-Historical-Data-Monthly.xlsx) (Public)

---

## 👤 Author

<div align="center">

**Ayodele Isaiah Idowu**

MSc Applied Economics · Development & Agricultural Economics
University of Göttingen · DAAD LfA Scholar

[![Website](https://img.shields.io/badge/Portfolio-ayodeleid.com-0A66C2?style=for-the-badge&logo=google-chrome&logoColor=white)](https://ayodeleid.com)
[![GitHub](https://img.shields.io/badge/GitHub-AyodeleID-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/AyodeleID)

</div>
