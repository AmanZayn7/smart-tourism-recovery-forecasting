## Smart Traveler Insights System 
A comparative machine learning forecasting and decision-support system designed to analyze and predict post-pandemic recovery in international tourist arrivals across major Asia-Pacific tourism hubs.
The project evaluates Ridge Regression, Random Forest, and XGBoost against a transparent Seasonal-Naïve baseline under structural shock conditions introduced by COVID-19, using a feature-engineered, rolling-origin validation framework.

Forecasts and diagnostics are operationalized through a reproducible Streamlit dashboard.

## Problem Context

COVID-19 introduced structural breaks, regime shifts, and broken seasonality in international tourism data (2020–2021).

Traditional univariate time-series models assume stable seasonal patterns. Under post-shock conditions, those assumptions degrade.

This project evaluates whether machine learning models, benchmarked transparently against a Seasonal-Naïve baseline, can produce defensible short-horizon forecasts in structurally unstable recovery regimes.

## Data Scope

Timeframe: January 2017 – December 2024 (monthly)

Cities:
- Singapore (STB data)
- Hong Kong (HKTB data)
- Bangkok / Thailand proxy (Bank of Thailand)

Variables:
- Monthly international visitor arrivals
- Google Trends indicators
- Hotel Occupancy Rates

Kuala Lumpur was evaluated during scoping but excluded due to data consistency limitations.

## Feature Engineering

- Lag features: 1, 3, 6, 12 months  
- Rolling means: 3-month, 6-month  
- Month dummy variables  
- COVID structural-break dummy  
- Standardization pipeline  

Validation methodology:
- 2024 hold-out year
- Rolling / expanding-origin evaluation design
- No leakage between training and test sets

## Models Evaluated

1. Seasonal-Naïve (Baseline comparator)  
2. Ridge Regression  
3. Random Forest  
4. XGBoost  

Evaluation Metrics:
- MAPE (primary)
- MAE
- RMSE
- R² (secondary diagnostic)

Variance-based metrics were interpreted cautiously in plateaued recovery regimes where test-set variance was low.

## Results (2024 Hold-Out)

| City        | Best Model        | Key Performance |
|-------------|------------------|----------------|
| Bangkok     | Ridge Regression | R² ≈ 0.859, MAPE ≈ 3.09% |
| Singapore   | Random Forest    | MAPE ≈ 5.04% |
| Hong Kong   | Seasonal-Naïve   | MAPE ≈ 10.10% |
All models were trained on 2017–2023 data and evaluated strictly on an unseen 2024 hold-out period.


## Interpretation
- Ridge Regression demonstrated strong robustness under volatile recovery conditions.
- Random Forest generalized effectively for Singapore despite low variance in the test period.
- XGBoost showed signs of overfitting and weaker out-of-sample generalization.
- In plateaued recovery regimes, Seasonal-Naïve remained competitive.
- Model selection was therefore context-sensitive rather than complexity-driven.

## System Architecture

Backend:
- Modular data ingestion
- Feature engineering pipeline
- Model training & evaluation
- Artifact generation
- Configuration-based model selection per city

Frontend:
- Streamlit-based dashboard
- Forecast visualizations
- Model comparison outputs
- Diagnostic metrics
- Downloadable results

The system translates research-grade forecasting into a reproducible, interactive decision-support tool.

## Project Structure
app/ - Streamlit frontend
stis/ - Core forecasting pipeline
data/ - Processed datasets
raw/ - Source datasets
requirements.txt - Python dependencies

Virtual environments and large artifacts are intentionally excluded.

## Why This Matters

Post-pandemic tourism data violates classical stationarity assumptions.
This system demonstrates that:

- Transparent baseline benchmarking is critical.
- Model complexity does not guarantee superior performance.
- Regularized linear models can outperform advanced ensembles under structural instability.
- Metric interpretation must adapt to low-variance recovery regimes.
- Methodological discipline outweighs algorithm hype.

## Quick Start
```bash
git clone https://github.com/AmanZayn7/smart-tourism-recovery-forecasting.git
cd smart-tourism-recovery-forecasting
pip install -r requirements.txt
python -m stis.build_artifacts
streamlit run app/app.py
```

