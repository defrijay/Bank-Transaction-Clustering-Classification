# Bank Customer Segmentation: From Raw Transactions to Predictable Segments

![Cover](assets/cover.jpg)

This project turns raw, unlabeled bank transaction data into **3 clear customer segments** based purely on financial behavior (account balance and transaction size), then builds a **Random Forest** model that can instantly predict which segment a brand-new customer belongs to  with **99.1% accuracy** on customers it has never seen before. Once deployed, the model can score a new customer in milliseconds, without needing to re-run the clustering process.

An earlier version of this analysis reported a flashier-looking result  10 segments named after cities, with a silhouette score of 0.71. That number turned out to be a bug: an unscaled city column dominated the math, so the model was really just sorting customers alphabetically by city name, not by how they actually behave financially. This README walks through the corrected analysis and briefly explains what went wrong the first time.

---

## Background

The bank processes millions of transactions every day but treats almost every customer the same way. A customer with a large, stable balance needs a different approach from one making small, occasional transactions  yet marketing sends both the same offers, and the risk team has no easy way to know which accounts need closer attention.

The core problem: raw transaction data has no built-in label. Nothing in the data says "premium customer" or "high-risk customer." This project builds that label from scratch  first by grouping historical customers into behavior-based segments, then by training a model that can place any new customer into the right segment the moment they sign up.

---

## Business Questions

| # | Business Question | Short Answer |
|---|---|---|
| **BQ1** | Does the data actually contain meaningful customer groups? | Yes  silhouette score 0.37, a moderate but genuine grouping |
| **BQ2** | What actually separates one segment from another? | Account balance and transaction size  **not** location, **not** gender |
| **BQ3** | Can we predict a new customer's segment without re-running the clustering? | Yes  a saved pipeline scores one new customer in milliseconds |
| **BQ4** | Which model should the business trust? | Random Forest  99.1% accuracy, checked only once on data it never saw during training |

---

## Dataset

- Historical bank transaction data, with key fields: account balance, transaction value, customer city (1,595 unique cities), and gender.
- Account balances are heavily skewed  from near zero up to 82 million INR, with a typical (median) customer sitting around 17,000 INR. Because of this skew, balance and transaction values were mathematically transformed before being used for grouping.
- This analysis runs on a **1.5% sample of the full transaction history (15,729 rows)**. The overall story (3 segments, driven by balance and transaction size) is expected to hold at full scale, but the exact numbers should be re-confirmed once the complete dataset is available.
- Two notebooks power this workflow:
  1. **`Clustering_Perfected.ipynb`**  reads the raw transactions and forms the 3 customer segments.
  2. **`Classification_Perfected.ipynb`**  trains and compares two prediction models, then saves a ready-to-use scoring tool.

---

## Method

1. **Clean the data before grouping.** Balance and transaction values were rescaled to correct for their skew. The city field (1,595 categories) was deliberately **left out** of the grouping step  it was only used afterward, to describe the segments, not to shape them.
2. **Let the number of segments emerge from the data.** Instead of deciding in advance how many segments to create, the analysis tested group counts from 2 to 10 and measured which split produced the most genuinely distinct groups.
3. **Sanity-check each segment.** Every resulting segment was checked against balance, transaction size, city, and gender to confirm the real differentiator is financial behavior  not an accidental side effect of how the data was encoded.
4. **Train the prediction model.** Two model types (Random Forest and XGBoost) were trained to predict a customer's segment, first compared on a separate tuning set to pick the better model.
5. **One final, honest check.** The chosen model was tested on a completely separate set of customers it had never touched during tuning  and evaluated on that set only **once**, so the final accuracy number wasn't artificially inflated.
6. **Package it for real use.** The full pipeline (data cleaning steps + trained model) was saved as one tool that can score a new customer on the spot  even a customer from a city the model has never seen before.

---

## Insight

### 1. The data holds a real  if moderate  customer grouping

Testing group sizes from 2 to 10 found the best split at **3 segments**, with a silhouette score of **0.37**. A score above zero means the groups are genuinely more distinct than a random split would be; 0.37 sits in the "moderate but real" range  not the sharply separated groups a score above 0.5 would suggest, but a real pattern nonetheless.

The earlier version's 0.71 score looked far more impressive, but it wasn't real  it came from a bug where an un-rescaled city column drowned out every other signal, so the model was effectively just sorting customers by the alphabetical order of their city.

![Silhouette score: before vs. after the fix](assets/04_silhouette_comparison.png)

### 2. Balance and transaction size define the segments  not city, not gender

![Segment sizes](assets/01_segment_size.png)

| Segment | Avg. Balance (INR) | Avg. Transaction (INR) | Customers | Share |
|---|---:|---:|---:|---:|
| High-Value – Big Spenders | 217,839 | 3,338 | 7,014 | 44.6% |
| Mid-Tier – Light Spenders | 54,016 | 189 | 6,615 | 42.1% |
| Growth – Moderate Spenders | 315 | 871 | 2,100 | 13.4% |

![Average balance and transaction size per segment](assets/02_segment_characteristics.png)

Two things worth noting:

- **The Growth segment keeps almost no balance** (315 INR on average) but still transacts *more* per transaction than the Mid-Tier segment (871 vs. 189 INR). This looks less like "customers with less money" and more like customers who let their balance run close to zero between transactions  worth a closer look from the risk team, and a possible opportunity for marketing to convert them into savers.
- **City and gender barely matter.** No single city makes up more than 11.6% of any segment, and the same city (Mumbai) is the top city in all three segments. If location truly drove the grouping, each segment would cluster around a different city  it doesn't. Gender shows the same flat pattern across all three segments.

**What this means for the business:** campaigns and risk rules should be built around balance and transaction behavior, not where a customer lives. A "premium banking for Gurgaon" campaign is targeting the wrong signal  a premium customer in Gurgaon looks financially identical to one in Chennai.

### 3. New customers can be classified instantly

The saved pipeline can score a brand-new customer and return their segment in milliseconds  no historical re-analysis needed. Even a customer from a city the model has never seen simply gets a neutral fallback value instead of causing an error. That means this tool can plug directly into a loan application, a mobile banking sign-up, or a CRM lookup, the moment a new customer's first transaction lands.

### 4. Random Forest is the model worth trusting  and it's learning the right thing

**Random Forest, tested once on unseen data: 99.11% accuracy, 99.02% macro F1 score.**

This high accuracy makes sense once you see *why* the model gets it right: it's largely just re-reading the same two numbers (balance and transaction size) that were used to create the segments in the first place.

![What drives the model's predictions](assets/03_feature_importance.png)

**Balance (58%) and transaction amount (39%) together explain 98% of what the model is looking at**, while city (0.8%) and gender (0.1%) play almost no role. This is the honest version of a story the earlier, buggy analysis told for the wrong reason  back then, city alone accounted for 91% of the model's decisions, because it was simply re-discovering the same encoding bug, not learning anything real about customer behavior.

The earlier version also never used a proper separate tuning set  the same "test" data was used both to fine-tune the model and to report its final score, which risks an overly optimistic number. This version keeps tuning and final testing strictly separate.

---

## Recommendations

- **Base campaign targeting and risk policy on balance and transaction size**  not city or gender. These two factors explain 98% of what separates one segment from another; city has been shown not to matter here.
- **Take a closer look at the Growth segment before writing it off as "low value."** Near-zero balance paired with above-average transaction size is a pattern worth a risk review, and a potential marketing opportunity to turn these customers into savers.
- **Plug the prediction tool into the customer onboarding flow** (mobile sign-up, loan applications, CRM) so every new customer's segment is known from their very first transaction  no waiting for the next batch analysis.
- **Re-confirm these numbers on the full dataset** before using them for major decisions  this analysis is based on a 1.5% sample only.
- **Don't discard the city field from every future model.** It doesn't matter for spending-behavior segmentation, but it may still be useful elsewhere, such as fraud detection or branch staffing.
- **Keep tuning data and final-testing data strictly separate on future projects**  that discipline is exactly what makes the 99.1% figure here trustworthy.

---

## Conclusion

The bank's transaction data does contain a genuine customer grouping: three segments, separated by account balance and transaction size  not by city or gender. The grouping is moderate rather than sharply distinct (silhouette score 0.37), but it's an honest result, unlike the earlier version's inflated 0.71 score, which turned out to be a data bug in disguise.

These three segments  **High-Value Big Spenders**, **Mid-Tier Light Spenders**, and **Growth Moderate Spenders**  can now be predicted for any new customer with 99.1% accuracy, through a ready-to-deploy tool that needs no re-analysis. With a solid, honest foundation in balance and transaction behavior, the bank now has a much sharper basis for targeting campaigns and risk rules  and a useful reminder that results which look "too good" (a 0.71 silhouette score, 91% importance for a single field) deserve a second look before they're trusted.
