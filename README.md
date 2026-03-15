# Data Analysis App

![Language](https://img.shields.io/badge/Language-Python-3776AB?style=flat-square) ![Stars](https://img.shields.io/github/stars/Devanik21/Data-Analysis-App?style=flat-square&color=yellow) ![Forks](https://img.shields.io/github/forks/Devanik21/Data-Analysis-App?style=flat-square&color=blue) ![Author](https://img.shields.io/badge/Author-Devanik21-black?style=flat-square&logo=github) ![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

> Interactive data analysis for everyone — upload any dataset and receive automated EDA, statistical insights, and ML-ready feature analysis in minutes.

---

**Topics:** `machine-learning` · `data-science` · `automated-eda` · `data-analysis` · `pandas` · `plotly` · `python` · `statistical-profiling` · `streamlit` · `visualization`

## Overview

Data Analysis App is a zero-code exploratory data analysis (EDA) platform built on Streamlit that
accepts any tabular dataset in CSV, Excel, or JSON format and automatically produces a comprehensive
analytical report without requiring the user to write a single line of code. It is designed for
data scientists who want to quickly orient themselves to a new dataset, for domain experts without
programming backgrounds who need data insight, and for students learning what good EDA looks like.

The automated analysis pipeline follows a principled structure. The first pass covers dataset anatomy:
shape, data types, missing value distribution, duplicate detection, and cardinality analysis for
categorical columns. The second pass covers univariate statistics: distribution plots and summary
statistics for numerical columns, frequency charts and mode analysis for categoricals, and temporal
distribution for datetime columns. The third pass covers bivariate relationships: correlation matrix
with significance testing, scatter plot matrices for numerical columns, and chi-squared tests for
categorical associations.

The fourth pass is the most analytically valuable: automated anomaly and insight flagging. The system
identifies highly skewed distributions and suggests transformations, flags low-variance features that
may be uninformative, identifies features with high missing-value rates and suggests imputation strategies,
detects potential data leakage candidates (features perfectly correlated with the target), and highlights
class imbalance in classification targets with corrective strategy suggestions.

---

## Motivation

EDA is the most important step in any data science project — more important than model selection, more
important than hyperparameter tuning. Yet it is tedious, time-consuming, and often done superficially
because of the mechanical effort required. This application automates the mechanical EDA work so that
analysts can spend their time on interpretation and hypothesis generation rather than on plotting
histograms.

---

## Architecture

```
Dataset Upload (CSV / Excel / JSON)
        │
  pandas: load, type inference, missing value map
        │
  Pass 1: Dataset Anatomy
  (shape, types, nulls, duplicates, cardinality)
        │
  Pass 2: Univariate Statistics
  (distributions, summary stats, frequency charts)
        │
  Pass 3: Bivariate Analysis
  (correlation, scatter matrices, chi-squared tests)
        │
  Pass 4: Automated Insight Flagging
  (skew, low variance, missing%, leakage risk, imbalance)
        │
  HTML/PDF Report Export
```

---

## Features

### Automatic Dataset Anatomy
Instant report on dataset structure: row/column count, data type per column, missing value percentage and pattern (MCAR/MAR heuristics), duplicate detection, and unique value cardinality for categoricals.

### Univariate Distribution Analysis
Distribution plots for all numerical columns (histogram + KDE + box plot) and frequency bar charts for categoricals — with skewness, kurtosis, and outlier count statistics.

### Correlation Analysis Suite
Pearson, Spearman, and Kendall correlation matrices with significance p-values; colour-coded heatmap; and automatic identification of highly correlated feature pairs (potential multicollinearity).

### Automated Insight Flags
System-generated textual insights: skewed distributions with suggested transformations, high-missing columns with imputation recommendations, constant and quasi-constant features, and potential target leakage detection.

### Missing Value Visualisation
Missingno-style matrix plot showing missing value patterns across rows and columns — revealing whether missingness is random, systematic, or correlated between columns.

### Target Variable Analysis
Designate any column as the target variable and receive target-focused analysis: distribution, class balance (classification) or skewness (regression), and feature-target correlation ranking.

### Feature Engineering Suggestions
Automatic suggestions for feature engineering: log-transform for right-skewed numericals, one-hot encoding for low-cardinality categoricals, ordinal encoding candidates, and date feature extraction.

### One-Click Report Export
Export the full EDA analysis as a self-contained HTML report (with embedded Plotly charts) or PDF summary for sharing with stakeholders who don't have access to the application.

---

## Tech Stack

| Library / Tool | Role | Why This Choice |
|---|---|---|
| **Streamlit** | Application framework | File upload, interactive plots, report download button |
| **pandas** | Data manipulation | Type inference, missing values, statistics, groupby |
| **Plotly** | Interactive visualisation | Distribution plots, heatmaps, scatter matrices |
| **SciPy** | Statistical tests | Spearman, Kendall, chi-squared, Shapiro-Wilk normality |
| **NumPy** | Numerical operations | Skewness, kurtosis, percentile computation |
| **missingno (optional)** | Missing value plots | Matrix, bar, heatmap for missing patterns |
| **WeasyPrint / pdfkit** | PDF export | Convert HTML report to downloadable PDF |

---

## Getting Started

### Prerequisites

- Python 3.9+ (or Node.js 18+ for TypeScript/JavaScript projects)
- A virtual environment manager (`venv`, `conda`, or equivalent)
- API keys as listed in the Configuration section

### Installation

```bash
git clone https://github.com/Devanik21/Data-Analysis-App.git
cd Data-Analysis-App
python -m venv venv && source venv/bin/activate
pip install streamlit pandas plotly scipy numpy missingno
streamlit run app.py
```

---

## Usage

```bash
# Launch the app
streamlit run app.py

# Automated EDA from CLI (no UI)
python auto_eda.py --data titanic.csv --target Survived --output eda_report.html

# Statistical test suite
python stats_tests.py --data data.csv --target target --alpha 0.05

# Feature correlation ranking
python correlations.py --data data.csv --target target --method spearman
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `MAX_UPLOAD_MB` | `200` | Maximum uploaded file size in megabytes |
| `HIGH_CARDINALITY_THRESHOLD` | `50` | Unique values above which a column is 'high cardinality' |
| `MISSING_FLAG_THRESHOLD` | `0.2` | Missing value fraction above which a column is flagged |
| `CORR_FLAG_THRESHOLD` | `0.9` | Correlation coefficient above which a pair is flagged |
| `OUTLIER_IQR_MULTIPLIER` | `1.5` | IQR multiplier for outlier detection (1.5 = standard, 3.0 = extreme only) |

> Copy `.env.example` to `.env` and populate required values before running.

---

## Project Structure

```
Data-Analysis-App/
├── README.md
├── requirements.txt
├── app.py
└── ...
```

---

## Roadmap

- [ ] AutoML integration: from EDA to trained baseline model with one click (AutoSklearn or PyCaret)
- [ ] Time-series EDA mode: stationarity tests, decomposition, autocorrelation, seasonality detection
- [ ] Comparative EDA: compare two datasets (e.g., train vs test) and flag distribution shift
- [ ] Natural language EDA queries: 'which column has the most outliers?' answered conversationally
- [ ] Custom insight rules: user-definable flag conditions (e.g., flag any column with >30% zeros)

---

## Contributing

Contributions, issues, and suggestions are welcome.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-idea`
3. Commit your changes: `git commit -m 'feat: add your idea'`
4. Push to your branch: `git push origin feature/your-idea`
5. Open a Pull Request with a clear description

Please follow conventional commit messages and add documentation for new features.

---

## Notes

EDA results are automatically computed — interpretation requires domain knowledge that the tool cannot provide. Automated insight flags are heuristic suggestions, not conclusions. Always review flagged issues in the context of the dataset's domain and collection process before acting on them.

---

## Author

**Devanik Debnath**  
B.Tech, Electronics & Communication Engineering  
National Institute of Technology Agartala

[![GitHub](https://img.shields.io/badge/GitHub-Devanik21-black?style=flat-square&logo=github)](https://github.com/Devanik21)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-devanik-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/devanik/)

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

*Built with curiosity, depth, and care — because good projects deserve good documentation.*
