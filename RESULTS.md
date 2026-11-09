# Results — Customer Segmentation & Response Modelling (UCI Bank Marketing)

**Every number below was computed by running `main.py` on the real dataset.
Nothing is estimated or invented.** Machine-readable copy: `results.json`.
Chart: `figures/segment_subscription_rates.png`.

## Dataset

- **UCI Bank Marketing, `bank-full.csv` — 45,211 customers**, 16 features +
  target. Portuguese bank phone-marketing campaigns, May 2008 – Nov 2010.
- Source (direct public download, no auth):
  `https://archive.ics.uci.edu/static/public/222/bank+marketing.zip`
- Citation: Moro et al., 2011, *Using Data Mining for Bank Direct Marketing*.
- **Target `y` = subscribed to a term deposit (yes/no). This is a
  subscription/response model, NOT a churn model — Bank Marketing has no
  churn label.** Any resume bullet using these numbers must say
  "subscription propensity", not "churn", unless a churn-labelled dataset
  (e.g. Telco Customer Churn) is run through the same pipeline.
- **0 missing values** in the file.
- `duration` (last call length) was **excluded from the classifier**: it is
  only known after the call ends, so including it leaks the outcome.

## 1. EDA — what the data looks like

| Fact | Value |
|---|---|
| Customers | 45,211 |
| Subscribed to term deposit | 5,289 (**11.70%**); 39,922 did not |
| Age | mean 40.9, median 39, range 18–95 |
| Balance (EUR) | mean €1,362, median €448, range −€8,019 to €102,127 |
| Negative balance | 8.33% of customers |
| Never contacted before (`pdays = −1`) | **81.74%** (36,954 customers) |
| Previous contacts | median 0, mean 0.58, max 275 |
| Contacts in this campaign | mean 2.76, max 63 |

Subscription rate by **previous campaign outcome** (the strongest single signal):

| Previous outcome | n | Subscription rate |
|---|---|---|
| success | 1,511 | **64.73%** |
| other | 1,840 | 16.68% |
| failure | 4,901 | 12.61% |
| unknown / never contacted | 36,959 | 9.16% |

Subscription rate by **contact type**: cellular 14.92% (n=29,285),
telephone 13.42% (n=2,906), unknown 4.07% (n=13,020).

Subscription rate by **job** (top 5): student 28.68% (n=938), retired
22.79% (n=2,264), unemployed 15.50% (n=1,303), management 13.76%
(n=9,458), admin. 12.20% (n=5,171). Largest groups: blue-collar 9,732,
management 9,458, technician 7,597.

Subscription rate by **month** varies wildly (Mar 51.99%, Dec 46.73%,
Sep 46.46%, Oct 43.77% vs May 6.72%) — but the high-rate months are
small (Mar n=477, Dec n=214) while May is the bulk campaign month
(n=13,766). Read this as campaign-wave/seasonality confounded with volume,
not as "call in March".

## 2. Segmentation — K-Means on RFM-style features

Features (standardised): **Recency** = `pdays` (−1 = never contacted,
recoded to 999 so never-contacted = least recent), **Frequency** =
`previous` contacts, **Monetary** = `balance` clipped at the 1st/99th
percentile. Silhouette scores (10,000-row sample): k=2: 0.7191, k=3:
0.7450, **k=4: 0.7526**, k=5: 0.6931, k=6: 0.6789, k=7: 0.6837 →
**4 segments**.

| Segment | Size | Share | Mean recency | Mean prev. contacts | Mean balance | Median balance | Subscription rate | Subscribers |
|---|---|---|---|---|---|---|---|---|
| **2 — Frequent engagers** | 1,515 | 3.4% | 208 days | 8.54 | €1,455 | €640 | **26.20%** | 397 |
| **1 — Warm repeat contacts** | 6,438 | 14.2% | 227 days | 1.98 | €1,107 | €546 | **21.95%** | 1,413 |
| **3 — High-balance dormant** | 2,967 | 6.6% | 925 days | 0.18 | **€9,390** | €7,203 | 13.92% | 413 |
| **0 — Never-contacted mass** | 34,291 | 75.8% | 999 days | 0.00 | €711 | €345 | 8.94% | 3,066 |

Reading the segments:

- **Segment 2 (frequent engagers)** is tiny but converts at 26.20% —
  2.2× the 11.70% base rate. Every member was contacted before (0%
  never-contacted) and averages 8.5 previous contacts. Top job: management.
- **Segment 1 (warm repeat contacts)** converts at 21.95%, also 0%
  never-contacted. Together segments 1 + 2 are 7,953 customers (17.6% of
  the base) but deliver **1,810 of 5,289 subscribers (34.2%)**.
- **Segment 3 (high-balance dormant)** is the richest group by far
  (mean balance €9,390 vs €711 in the mass segment, median €7,203) yet
  90.5% were never contacted. Even so it subscribes at 13.92% — above the
  base rate. Highest value per call in the whole base.
- **Segment 0 (never-contacted mass)** is 75.8% of customers, lowest
  balance (median €345), lowest rate (8.94%) — but because it is so large
  it still holds **3,066 subscribers, 58.0% of all subscribers**. It
  cannot be ignored; it must be worked cheaply and selectively.

## 3. Subscription-propensity classifier

80/20 stratified split (train 36,168 / test 9,043; test positives 1,058,
11.70%). `duration` excluded (leakage — see above).

| Model | Test ROC-AUC | 3-fold CV AUC (train) | Precision @0.5 | Recall @0.5 | F1 @0.5 |
|---|---|---|---|---|---|
| Logistic Regression | **0.7722** | 0.7634 ± 0.0039 | 0.2662 | 0.6238 | 0.3732 |
| Random Forest (200 trees) | **0.7945** | 0.7739 ± 0.0063 | 0.6446 | 0.2212 | 0.3293 |

Confusion matrices (test set, threshold 0.5):

- Logistic Regression: TP 660, FP 1,819, FN 398, TN 6,166
- Random Forest: TP 234, FP 129, FN 824, TN 7,856

**Recall-oriented operating point (Random Forest):** the 0.5 threshold is
too strict for a campaign — it catches only 22% of subscribers. Tuning the
threshold on **out-of-fold** train probabilities to reach recall ≥ 0.80
gives threshold **0.06**, which on the held-out test set delivers
**precision 0.1966, recall 0.8318, F1 0.3180** while flagging **49.5%** of
customers (4,476 of 9,043). That catches **880 of 1,058 subscribers** and
skips 4,567 low-probability customers who hold only 178 subscribers.
Precision 19.66% is a **1.68× lift** over the 11.70% base rate.
(Deliberate detail: the threshold was tuned on out-of-fold probabilities,
not in-sample ones — tuning it in-sample collapsed test recall to 0.12
because the forest memorises its training data.)

Top Random Forest features: balance 0.160, age 0.136, day-of-month 0.120,
campaign contacts 0.068, pdays 0.040, previous-outcome=success 0.034,
contact=unknown 0.025, previous 0.022, contact=cellular 0.019,
housing=yes 0.018. Caveat: `day` is mostly a proxy for campaign
wave/month, not a customer trait — don't over-interpret it.

## 4. Business recommendations (each follows from the numbers above)

1. **Call the warm segments first.** Segments 1 + 2 (7,953 customers,
   17.6% of the base) subscribe at 21.95–26.20% — about 2× the base rate —
   and already deliver 34.2% of all subscribers. Opening each campaign
   with them roughly doubles conversions per call versus dialling at
   random. They are, by definition, people the bank has spoken to before.
2. **Win back the high-balance dormant segment before buying new leads.**
   Segment 3 is only 2,967 customers but holds mean balance €9,390
   (13× the mass segment) and 90.5% have never been contacted — yet they
   still subscribe at 13.92%. A targeted premium-offer call to this small
   group is the highest value-per-call play in the base: lifting them to
   segment 1's 21.95% rate would add **~238 subscriptions from under
   3,000 calls**.
3. **Let the model cut the call list in half.** At the recall-oriented
   threshold (0.06), the Random Forest flags 49.5% of customers and
   catches 83.18% of subscribers at 19.66% precision (1.68× base rate).
   Calling only the flagged half reaches 880 of 1,058 subscribers while
   skipping ~4,500 low-probability calls. Do **not** use the default 0.5
   threshold for this campaign — it finds only 22% of subscribers.
4. **Re-contact previous successes first, and fix the contact channel.**
   Clients whose last campaign ended in success re-subscribe at **64.73%**
   (978 of 1,511) versus 9.16% for never-contacted clients — always
   re-call prior successes before cold lists. And 13,020 customers have
   an "unknown" contact type converting at just 4.07% (vs 14.92% cellular):
   verify a cellular number before dialling them, or work them through a
   cheaper channel. Relatedly, students (28.68%) and retired customers
   (22.79%) convert at 2–2.5× the base rate — pitch them differently
   (first-savings vs retirement income) instead of one blanket script.
   For the never-contacted mass (segment 0, 75.8% of the base, 8.94%),
   use low-cost outreach first and reserve agent calls for the
   model-flagged subset — no blanket discounts for the loyal low-risk
   majority.

## 5. Limitations — read before quoting these numbers

- **Subscription ≠ churn.** The target is term-deposit subscription.
  There is no churn label here; do not present the 0.79 AUC as a
  "churn AUC" in an interview or on a resume.
- **RFM is proxied, not literal.** Real RFM needs transaction recency /
  frequency / monetary value. Here recency = days since last marketing
  contact, frequency = prior campaign contacts, monetary = account
  balance. The segments are behaviourally sensible, but say "RFM-style".
- **2008–2010 Portuguese retail data.** Behaviour, rates, and balances
  are from that market and era; the methods transfer, the exact rates
  don't.
- **Month effects are confounded** with campaign size (see §1) — the
  March/December rates rest on a few hundred contacts each.
- **`day`-of-month feature importance** largely reflects campaign
  timing, not customer quality.
