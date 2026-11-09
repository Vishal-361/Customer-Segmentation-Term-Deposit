"""
Customer Segmentation & Response Modelling — UCI Bank Marketing
================================================================
Dataset : UCI Bank Marketing, bank-full.csv (45,211 rows, May 2008 – Nov 2010)
          Direct-marketing phone campaigns of a Portuguese bank.
          Target `y` = client subscribed to a term deposit (yes/no).

What this script does
---------------------
1. EDA — size, target rate, distributions, key rates by group.
2. RFM-style segmentation with K-Means (k chosen by silhouette score,
   computed on a random 10,000-row sample for speed):
      Recency   = pdays    (days since last contact; -1 in the file means
                            never contacted -> recoded to 999, least recent)
      Frequency = previous (contacts before this campaign)
      Monetary  = balance  (average yearly balance, EUR)
3. Subscription-propensity classifier (Logistic Regression + Random Forest).
   `duration` (last call length) is EXCLUDED: it is only known after the
   call ends, so using it leaks the outcome and inflates AUC.
4. Writes every number to results.json and a segment chart to figures/.

Honest labelling: the Bank Marketing target is term-deposit SUBSCRIPTION,
not churn. There is no churn label in this dataset. The classifier below
is a subscription-propensity model; a true churn model needs a churn
label (e.g. the Telco Customer Churn dataset).

Run:  python main.py        (from this project folder, venv active)
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (auc, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import silhouette_score

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "bank-full.csv"
OUT = BASE / "results.json"
FIG = BASE / "figures" / "segment_subscription_rates.png"
RNG = 42

# ---------------------------------------------------------------- load + EDA
df = pd.read_csv(DATA, sep=";")
df["y_bin"] = (df["y"] == "yes").astype(int)

eda = {
    "rows": int(len(df)),
    "columns": int(df.shape[1] - 1),  # excluding derived y_bin
    "target": "y (subscribed to term deposit)",
    "subscription_rate": round(float(df["y_bin"].mean()), 4),
    "subscribers": int(df["y_bin"].sum()),
    "non_subscribers": int((1 - df["y_bin"]).sum()),
    "missing_values_total": int(df.isna().sum().sum()),
    "date_range": "May 2008 – Nov 2010 (per UCI docs)",
    "numeric_summary": {
        c: {k: round(float(v), 2) for k, v in df[c].describe().to_dict().items()}
        for c in ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
    },
    "never_contacted_before_share_pdays_minus1": round(float((df["pdays"] == -1).mean()), 4),
    "negative_balance_share": round(float((df["balance"] < 0).mean()), 4),
    "subscription_rate_by_poutcome": {
        k: round(float(v), 4) for k, v in df.groupby("poutcome")["y_bin"].mean().to_dict().items()
    },
    "subscription_rate_by_contact": {
        k: round(float(v), 4) for k, v in df.groupby("contact")["y_bin"].mean().to_dict().items()
    },
    "subscription_rate_by_month": {
        k: round(float(v), 4) for k, v in df.groupby("month")["y_bin"].mean().to_dict().items()
    },
    "subscription_rate_by_job_top": {
        k: round(float(v), 4)
        for k, v in df.groupby("job")["y_bin"].mean().sort_values(ascending=False).head(5).to_dict().items()
    },
    "job_counts": {k: int(v) for k, v in df["job"].value_counts().to_dict().items()},
    "median_balance": round(float(df["balance"].median()), 2),
    "mean_balance": round(float(df["balance"].mean()), 2),
    "median_age": round(float(df["age"].median()), 2),
    "mean_age": round(float(df["age"].mean()), 2),
    "median_previous_contacts": round(float(df["previous"].median()), 2),
}

print(f"Rows: {eda['rows']:,} | subscription rate: {eda['subscription_rate']:.2%} "
      f"({eda['subscribers']:,} subscribers)")
print(f"Never contacted before (pdays=-1): {eda['never_contacted_before_share_pdays_minus1']:.2%}")
print(f"Negative balance: {eda['negative_balance_share']:.2%}")

# ------------------------------------------------- RFM-style K-Means segments
# Recency = pdays. NOTE: in this file -1 means "never contacted before"
# (the UCI docs say 999, but the CSV uses -1 — 81.7% of rows). Recode -1
# to 999 so never-contacted counts as LEAST recent; using -1 raw would
# invert recency for 82% of customers.
# Frequency = previous contacts; Monetary = balance (clipped at 1st/99th pct
# to stop a few huge balances dominating the distance metric).
recency = df["pdays"].astype(float).replace(-1, 999)
rfm = pd.DataFrame({
    "recency_pdays": recency,
    "frequency_previous": df["previous"].astype(float),
    "monetary_balance": df["balance"].astype(float).clip(
        lower=df["balance"].quantile(0.01), upper=df["balance"].quantile(0.99)),
})
rfm["frequency_previous"] = rfm["frequency_previous"].clip(
    upper=rfm["frequency_previous"].quantile(0.99))
X_rfm = StandardScaler().fit_transform(rfm)

# Silhouette on a random 10,000-row sample (full 45k pairwise distances are
# needlessly slow); K-Means itself is fitted on all rows.
rng = np.random.default_rng(RNG)
sample_idx = rng.choice(len(X_rfm), size=min(10_000, len(X_rfm)), replace=False)
X_sample = X_rfm[sample_idx]
silhouettes = {}
for k in range(2, 8):
    km = KMeans(n_clusters=k, n_init=10, random_state=RNG).fit(X_rfm)
    silhouettes[k] = round(float(silhouette_score(X_sample, km.predict(X_sample))), 4)
best_k = max(silhouettes, key=silhouettes.get)
print("Silhouette by k:", silhouettes, "-> best k =", best_k)

km = KMeans(n_clusters=best_k, n_init=10, random_state=RNG).fit(X_rfm)
df["segment"] = km.labels_

segments = []
for s in range(best_k):
    sub = df[df["segment"] == s]
    segments.append({
        "segment": int(s),
        "size": int(len(sub)),
        "share": round(float(len(sub) / len(df)), 4),
        "mean_recency_pdays": round(float(recency[df["segment"] == s].mean()), 1),
        "median_recency_pdays": round(float(recency[df["segment"] == s].median()), 1),
        "share_never_contacted": round(float((sub["pdays"] == -1).mean()), 4),
        "mean_frequency_previous": round(float(sub["previous"].mean()), 2),
        "mean_balance_eur": round(float(sub["balance"].mean()), 2),
        "median_balance_eur": round(float(sub["balance"].median()), 2),
        "mean_age": round(float(sub["age"].mean()), 1),
        "subscription_rate": round(float(sub["y_bin"].mean()), 4),
        "subscribers": int(sub["y_bin"].sum()),
        "top_job": str(sub["job"].mode().iloc[0]),
    })
segments.sort(key=lambda d: d["subscription_rate"], reverse=True)
for s in segments:
    print(f"Segment {s['segment']}: n={s['size']:,} ({s['share']:.1%}) | "
          f"recency {s['mean_recency_pdays']:.0f}d | freq {s['mean_frequency_previous']:.2f} | "
          f"balance €{s['mean_balance_eur']:,.0f} | subscribe {s['subscription_rate']:.2%}")

# chart: subscription rate per segment
FIG.parent.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 4.5))
labels = [f"Seg {s['segment']}\n(n={s['size']:,})" for s in segments]
rates = [s["subscription_rate"] * 100 for s in segments]
bars = ax.bar(labels, rates, color="#2c6fbb")
ax.set_ylabel("Subscription rate (%)")
ax.set_title("Term-deposit subscription rate by K-Means segment (UCI Bank Marketing)")
for b, r in zip(bars, rates):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3, f"{r:.1f}%", ha="center")
ax.set_ylim(0, max(rates) * 1.2)
fig.tight_layout()
fig.savefig(FIG, dpi=150)
plt.close(fig)
print("Chart saved:", FIG)

# ------------------------------------------- subscription-propensity classifier
# duration excluded (leakage: known only after the call). Everything else in.
FEATURES = [c for c in df.columns if c not in ("y", "y_bin", "segment", "duration")]
NUM = ["age", "balance", "day", "campaign", "pdays", "previous"]
CAT = [c for c in FEATURES if c not in NUM]

X = df[FEATURES]
y = df["y_bin"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RNG)
print(f"Train: {len(X_train):,} | Test: {len(X_test):,} | "
      f"test positives: {int(y_test.sum()):,} ({y_test.mean():.2%})")

pre = ColumnTransformer([
    ("num", StandardScaler(), NUM),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
])
models = {
    "logistic_regression": Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RNG)),
    ]),
    "random_forest": Pipeline([
        ("pre", pre),
        ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample",
                                       n_jobs=-1, random_state=RNG)),
    ]),
}

clf_results = {}
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    cm = confusion_matrix(y_test, pred).tolist()
    cv_auc = cross_val_score(pipe, X_train, y_train, cv=3, scoring="roc_auc")
    clf_results[name] = {
        "test_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "cv_auc_mean_3fold_train": round(float(cv_auc.mean()), 4),
        "cv_auc_std_3fold_train": round(float(cv_auc.std()), 4),
        "precision_at_0.5": round(float(precision_score(y_test, pred)), 4),
        "recall_at_0.5": round(float(recall_score(y_test, pred)), 4),
        "f1_at_0.5": round(float(f1_score(y_test, pred)), 4),
        "confusion_matrix_test": {"tn": cm[0][0], "fp": cm[0][1], "fn": cm[1][0], "tp": cm[1][1]},
    }
    print(f"{name}: AUC={clf_results[name]['test_auc']:.4f} | "
          f"P={clf_results[name]['precision_at_0.5']:.3f} "
          f"R={clf_results[name]['recall_at_0.5']:.3f} "
          f"F1={clf_results[name]['f1_at_0.5']:.3f} | CV AUC={cv_auc.mean():.4f}")

# Recall-oriented operating point for the best model (Random Forest).
# The threshold must be tuned on OUT-OF-FOLD probabilities: a Random Forest
# memorises its training data, so in-sample probabilities make any threshold
# look far better than it is on new customers (first attempt tuned in-sample
# and collapsed from 0.80 train recall to 0.12 test recall). Here the
# threshold is chosen on 3-fold out-of-fold train probabilities, then the
# final model (fit on all train rows) is evaluated on the held-out test set.
best = models["random_forest"]
oof_proba = cross_val_predict(best, X_train, y_train, cv=3, method="predict_proba")[:, 1]
thr80 = 0.5
for t in sorted(np.unique(np.round(np.quantile(oof_proba, np.linspace(0, 1, 401)), 4)), reverse=True):
    if recall_score(y_train, (oof_proba >= t).astype(int)) >= 0.80:
        thr80 = float(t)
        break
oof_recall_at_thr = round(float(recall_score(y_train, (oof_proba >= thr80).astype(int))), 4)
test_proba = best.predict_proba(X_test)[:, 1]
pred80 = (test_proba >= thr80).astype(int)
clf_results["random_forest_recall80_operating_point"] = {
    "threshold_chosen_on_oof_train_probs": round(thr80, 4),
    "oof_train_recall_at_threshold": oof_recall_at_thr,
    "test_precision": round(float(precision_score(y_test, pred80)), 4),
    "test_recall": round(float(recall_score(y_test, pred80)), 4),
    "test_f1": round(float(f1_score(y_test, pred80)), 4),
    "test_flagged_share": round(float(pred80.mean()), 4),
}
print("RF @ recall>=0.80 threshold", round(thr80, 4), "-> test P=",
      clf_results["random_forest_recall80_operating_point"]["test_precision"],
      "R=", clf_results["random_forest_recall80_operating_point"]["test_recall"])

# Random Forest feature importance (top 10, from the fitted pipeline)
rf = models["random_forest"].named_steps["clf"]
feat_names = models["random_forest"].named_steps["pre"].get_feature_names_out()
imp = sorted(zip(feat_names, rf.feature_importances_), key=lambda t: -t[1])[:10]
clf_results["random_forest_top_features"] = [
    {"feature": str(f), "importance": round(float(v), 4)} for f, v in imp
]
print("Top features:", clf_results["random_forest_top_features"])

results = {
    "dataset": "UCI Bank Marketing — bank-full.csv (45,211 rows)",
    "source_url": "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip",
    "citation": "Moro et al., 2011, 'Using Data Mining for Bank Direct Marketing'",
    "target_note": ("Target y = subscribed to a term deposit. This is a SUBSCRIPTION/ "
                    "response model, NOT a churn model — Bank Marketing has no churn label."),
    "leakage_note": ("'duration' (last call length) excluded from the classifier: it is "
                     "only known after the call ends."),
    "eda": eda,
    "segmentation": {
        "method": "K-Means on standardised RFM-style features "
                  "(recency=pdays with -1 never-contacted recoded to 999, "
                  "frequency=previous, monetary=balance clipped p1/p99); "
                  "silhouette scores on a random 10,000-row sample",
        "silhouette_by_k": silhouettes,
        "best_k": best_k,
        "segments": segments,
    },
    "classifier": clf_results,
}
OUT.write_text(json.dumps(results, indent=2))
print("Results saved:", OUT)
print("DONE.")
