# Bank Customer Segmentation: From Raw Transactions to Predictable Segments

## Executive Summary

The bank's transaction data separates into three customer segments defined by balance and spending size. Random Forest predicts which segment a new customer belongs to with 99.1% accuracy on data the model never saw during training. A saved pipeline scores a new customer in milliseconds, no clustering re-run required.

An earlier version of this analysis reported ten segments named after cities, with a silhouette score of 0.71. That result came from a bug: an unscaled location column with 1,595 distinct city values dominated every distance calculation in K-Means, so the model was grouping customers by an arbitrary alphabetical ID assigned to their city, not by their transaction behavior. Once every feature going into the clustering step is scaled the same way, three segments emerge, none of them tied to geography. The sections below walk through what changed, why, and what it means for marketing and risk.

## Background

The bank processes millions of transactions daily and treats most customers the same way. A customer holding a large balance with steady, recurring transactions needs a different pitch than a customer making small, infrequent ones. Marketing sends the same offers to both. Risk struggles to know which accounts deserve a closer look.

Raw transaction data carries no segment label. Nothing in the columns says "premium" or "high-risk." This project builds that label from scratch: cluster the historical transactions into groups, then train a classifier that can assign a new customer to a group the moment they sign up, without waiting to re-run the clustering.

## The Four Business Questions

| # | Question | Short Answer |
|---|----------|--------------|
| BQ1 | Does the data contain real customer groupings? | Yes — silhouette score 0.37, a moderate but genuine structure |
| BQ2 | What separates one group from another? | Account balance and transaction size. Not location, not gender |
| BQ3 | Can we predict a new customer's segment without re-clustering? | Yes — a saved pipeline scores one customer in milliseconds |
| BQ4 | Which model should the business trust? | Random Forest — 99.1% accuracy, checked once on held-out test data |

Each answer is unpacked below.

---

## BQ1: Does the Data Contain Meaningful Groupings?

**Yes, and here's the number that proves it.** Running K-Means across cluster counts from 2 to 10 and scoring each with the silhouette metric, the best split lands at **k = 3, silhouette = 0.3665**. A silhouette score above zero means clusters separate better than random assignment would; 0.37 sits in the range analysts typically call moderate, real structure, well short of the crisp 0.5+ that would suggest customers fall into obviously distinct camps.

That "moderate but real" framing matters because an earlier pass at this same dataset reported silhouette 0.71 and ten clean clusters. Digging into why revealed the cause: the clustering code label-encoded `CustLocation` (1,595 unique city strings) into raw integers from 0 to 1,594, then scaled only the three original numeric columns and left that integer column untouched. A single unscaled column with a range of 1,600 overwhelms three columns hovering between -3 and +3. K-Means was effectively sorting customers by the alphabetical rank of their city name. Every "segment" mapped cleanly to a dominant city because the model was clustering on city, full stop — the transaction behavior barely moved the needle.

The fix: log-transform the two heavily skewed monetary columns (account balance ranges from near zero to 82 million, while the median sits at 17,000), scale every feature that enters K-Means on the same footing, and drop the 1,595-category location column from the clustering step entirely. What's left is honest: three segments built from behavior, not from an artifact of how a city name got encoded.

## BQ2: What Separates One Group From Another?

**Balance and transaction size. Nothing else moves the needle.** The three segments, in the bank's own numbers:

| Segment | Avg. Balance (INR) | Avg. Transaction (INR) | Customers | Share |
|---|---:|---:|---:|---:|
| High-Value – Big Spenders | 217,839 | 3,338 | 7,014 | 44.6% |
| Mid-Tier – Light Spenders | 54,016 | 189 | 6,615 | 42.1% |
| Growth – Moderate Spenders | 315 | 871 | 2,100 | 13.4% |

Two things stand out. First, the *Growth* segment holds almost nothing in their accounts (an average of 315 rupees) yet spends more per transaction than the *Mid-Tier* segment (871 vs. 189 rupees). These look like customers running their balance close to zero between transactions rather than customers who simply have less money — a pattern risk teams may want to watch, and marketing may want to convert into savers rather than write off as low-value.

Second, and this contradicts what the earlier version of this analysis claimed: **location does not separate these segments.** Checking the most common city within each cluster, the top city never accounts for more than 11.6% of any segment's customers, and the same city (Mumbai) tops all three. If geography drove the segmentation, each cluster would concentrate heavily in one region — it doesn't. Gender shows the same story: every segment skews male at roughly the same rate the overall dataset does, so it isn't a differentiator either.

**What this means for the business:** campaigns and risk rules should key off balance and spending size, not the customer's city. A "premium banking in Gurgaon" campaign is targeting the wrong signal — a premium customer in Gurgaon looks financially identical to one in Chennai. The behavior is what's different, not the address.

## BQ3: Can We Predict a Segment Without Re-Clustering?

**Yes.** A saved pipeline — the fitted encoders, imputer, scaler, and the tuned Random Forest model — takes one new customer record and returns a segment in milliseconds. No historical dataset, no re-running K-Means.

Two live examples from the pipeline:

```
New customer predicted segment: High-Value - Big Spenders (Cluster 2)
New customer predicted segment: Growth - Moderate Spenders (Cluster 1)
```

The second example used a city that never appeared in the training data, and the pipeline still returned a segment instead of failing — unseen locations fall back to a neutral value rather than crashing the scoring function. That matters for a bank onboarding customers from towns that weren't in last year's transaction sample.

This is what turns a one-off analysis into something a loan-onboarding flow, mobile banking sign-up, or CRM lookup can call directly, the moment a new customer's first transaction comes in.

## BQ4: Which Model Should the Business Trust?

**Random Forest, at 99.1% accuracy on data it never touched during training or tuning.**

Both models were compared on a held-out evaluation set first, kept separate from the final test set:

| Model | Eval Accuracy | Eval Macro F1 |
|---|---:|---:|
| Random Forest | 99.49% | 99.35% |
| XGBoost | 99.28% | 99.15% |

Random Forest came out ahead on both metrics and went on to hyperparameter tuning (300 trees, no depth limit, minimum leaf size of 1). Its tuned score on eval barely moved (99.45% accuracy), which is a good sign — the model wasn't overfitting to a lucky split. The number that actually counts is the one checked exactly once, on the test set, after every modeling decision had already been made:

**Final Random Forest, test set, first and only look: 99.11% accuracy, 99.02% macro F1.**

Checking the test set only once matters more than it sounds. An earlier version of the classification notebook created a three-way train/eval/test split but only ever used two of the three — the eval set sat unused while the test set doubled as both the tuning ground and the final report card. That risks a subtly optimistic number: the "final" score creeps up because the model gets tuned toward it. Separating those two roles here means the 99.1% above wasn't shaped by the data it's being scored on.

**Why accuracy is this high:** the segments were built directly from balance and transaction size, and the classifier has access to those same two numbers. It's a nearly linear problem once you see it that way — feature importance confirms it, with balance (58%) and transaction amount (39%) accounting for 98% of what drives every prediction, versus a negligible role for location (0.8%) and gender (0.1%). This is the honest version of a story the earlier notebook told for the wrong reason: back when location leaked into the clustering step, it also dominated the classifier's feature importance at 91%, because the model was just recovering the same bug rather than learning a real relationship.

---

## What Changed From the Previous Version, and Why It Matters

| Issue | Before | After | Why It Matters |
|---|---|---|---|
| Location encoding | Unscaled integer 0–1594, dwarfing every other feature | Excluded from clustering, kept only for profiling | Segments now reflect behavior, not an artifact of encoding order |
| Monetary outliers | Raw values up to 82M scaled alongside a 17K median | Log-transformed before scaling | A handful of extreme balances no longer drag cluster centroids |
| Silhouette score | 0.71 (inflated by the location bug) | 0.37 (honest, moderate structure) | Sets the right expectation: three real segments, not ten illusory ones |
| Segment names | By dominant city (e.g. "High-Value – Gurgaon") | By balance and spend tier (e.g. "High-Value – Big Spenders") | Names now describe what actually differs between customers |
| Eval/test split | Eval set created but never used; test set doubled as tuning ground | Eval set drives model selection and tuning; test set checked once | The reported 99.1% wasn't shaped by the data it's measured on |
| Feature importance | Location: 91% (circular — location built the clusters, then "predicted" them) | Balance + transaction: 98% combined | The classifier is learning a real signal, not recovering a bug |

## Limitations and Next Steps

- **Sample size.** This analysis runs on a 1.5% sample (15,729 records) of the full transaction history. Directional findings should hold at full scale, but exact silhouette and accuracy numbers should be re-checked once the full dataset is available.
- **Only three segments.** Three segments are easy to act on but coarse. If marketing needs finer targeting, consider clustering separately within the High-Value segment (e.g. by transaction frequency) rather than forcing K-Means toward a higher k that the data doesn't naturally support.
- **The Growth segment deserves a second look.** Near-zero average balances combined with above-average transaction size is worth a manual review by the risk team before assuming this segment is simply "low value."
- **Location isn't useless, just not a segment driver.** It may still be a useful feature for other models (fraud detection, branch staffing) — this analysis only concludes it doesn't separate spending behavior segments.

## How the Two Notebooks Fit Together

1. **`Clustering_Perfected.ipynb`** reads the raw transactions, builds the three segments, and writes `clustered_data_fixed.csv` and `cluster_profile.csv`.
2. **`Classification_Perfected.ipynb`** reads those two files, trains and compares Random Forest and XGBoost, and saves the deployable scoring pipeline (`segment_classifier.pkl` plus its preprocessing artifacts).

Run the clustering notebook first. The classification notebook depends on its output.
