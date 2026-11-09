# Customer Segmentation & Response Modelling — UCI Bank Marketing

Customer segmentation (K-Means on RFM-style features) plus a
subscription-propensity classifier (Logistic Regression + Random Forest),
built on the real UCI Bank Marketing dataset — 45,211 customers from a
Portuguese bank's phone-marketing campaigns (May 2008 – Nov 2010).

**All numbers in `RESULTS.md` come from running this code on the real
dataset. No metric is invented or estimated.**

## Honest note on the target

The Bank Marketing target `y` is **term-deposit subscription** (did the
client say yes to a term deposit?). It is a marketing-response /
propensity model, **not a churn model** — this dataset has no churn label.
The classifier in this project therefore predicts subscription propensity.
If a true churn AUC is needed, the same pipeline should be run on a
dataset with a churn label (e.g. the Telco Customer Churn dataset).

## What it does

1. **EDA** — dataset size, subscription rate, distributions, and
   subscription rates by previous outcome, contact type, month, and job.
2. **RFM-style segmentation (K-Means)** — segments built from
   - Recency: `pdays` (days since last contact; 999 = never contacted)
   - Frequency: `previous` (contacts before this campaign)
   - Monetary: `balance` (average yearly balance, EUR; clipped at the
     1st/99th percentile so a few huge balances don't dominate)
   
   The number of segments is chosen by silhouette score (k = 2–7 tested).
   Each segment is profiled by size, mean recency/frequency/balance, and
   subscription rate.
3. **Subscription-propensity classifier** — Logistic Regression and Random
   Forest (80/20 stratified split), reporting ROC-AUC, precision, recall,
   F1, confusion matrix, and 3-fold CV AUC. `duration` (last call length)
   is **deliberately excluded**: it is only known after the call ends, so
   including it leaks the outcome and inflates AUC. A recall-oriented
   operating point (threshold tuned on train to reach recall ≥ 0.80) is
   also reported for the Random Forest.

## Dataset

- **Source:** UCI Machine Learning Repository, Bank Marketing (dataset 222)
- **Download (no auth needed):**
  `https://archive.ics.uci.edu/static/public/222/bank+marketing.zip`
- **File used:** `data/bank-full.csv` — 45,211 rows, 16 features + target
- **Citation:** Moro, S., Laureano, R., Cortez, P. (2011). *Using Data
  Mining for Bank Direct Marketing*. ESM'2011, pp. 117–121.
- The original download archives are kept in `data/`; `bank-names.txt`
  is the dataset documentation from UCI.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Runtime is ~1–3 minutes on a normal laptop (45k rows, no sampling needed).

## Outputs

- `results.json` — every number the script computes (EDA, segments,
  classifier metrics, feature importances)
- `figures/segment_subscription_rates.png` — subscription rate per segment
- `RESULTS.md` — the results written up with business recommendations

## Project structure

```
Customer-Segmentation-Churn/
├── main.py            # the whole pipeline: EDA → K-Means → classifiers
├── requirements.txt
├── README.md
├── RESULTS.md         # real results + business recommendations
├── results.json       # machine-readable results
├── figures/
│   └── segment_subscription_rates.png
└── data/
    ├── bank-full.csv  # 45,211 rows (the dataset actually used)
    ├── bank-names.txt # UCI documentation
    ├── bank.zip       # original UCI download (classic dataset)
    └── bank-additional.zip  # original UCI download (additional dataset)
```

## Skills demonstrated

Python, pandas, scikit-learn (K-Means, Logistic Regression, Random
Forest, train/test split, cross-validation, precision-recall threshold
tuning), matplotlib, EDA, feature engineering (RFM proxies), avoiding
target leakage (`duration`), and turning model output into business
recommendations.
