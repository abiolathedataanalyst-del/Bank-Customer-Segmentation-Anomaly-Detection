# 🏦 Bank Customer Segmentation & Anomaly Detection

> *Turning raw customer data into actionable intelligence — clustering behaviour, profiling value, and surfacing anomalies to drive smarter retention and risk strategies.*

---

## 📌 Table of Contents

1. [Project Overview](#-project-overview)
2. [Tools & Technologies](#-tools--technologies)
3. [Key Business Questions](#-key-business-questions)
4. [Analytical Workflow](#-analytical-workflow)
5. [Key Insights](#-key-insights)
6. [Strategic Recommendations](#-strategic-recommendations)
7. [Project Outputs](#-project-outputs)
8. [Conclusion](#-conclusion)
9. [Author](#-author)

---

## 🔍 Project Overview

Customer churn is one of the costliest challenges in retail banking. Banks that cannot distinguish their high-value loyalists from disengaged low-activity customers — or detect the anomalous few who signal fraud or systemic risk — are flying blind.

This project applies **unsupervised machine learning** to the BankChurners dataset (10,000+ credit card customers) to:

- **Segment** customers into behaviorally distinct groups using K-Means clustering
- **Visualize** those segments in two-dimensional space via PCA and t-SNE
- **Detect** anomalous customers using Isolation Forest
- **Profile** each segment to inform targeted retention, credit, and engagement strategies

The end result is a data-driven customer intelligence layer that business teams can act on immediately — from marketing personalisation to credit risk controls.

---

## 🛠️ Tools & Technologies

| Category | Tool / Library |
|---|---|
| **Language** | Python 3 |
| **Data Manipulation** | Pandas, NumPy |
| **Visualisation** | Matplotlib, Seaborn, Plotly Express |
| **Machine Learning** | Scikit-learn (K-Means, DBSCAN, Isolation Forest, PCA, t-SNE) |
| **Hierarchical Clustering** | SciPy (Linkage, Dendrogram) |
| **Environment** | Jupyter Notebook |
| **Output Formats** | CSV, Interactive HTML |

---

## ❓ Key Business Questions

This project was designed to answer six critical questions every retail bank should be asking about its customer base:

1. **Who are the bank's customer segments?** — Are customers meaningfully different from one another, and can we describe those differences?
2. **How many natural customer groups exist?** — What is the statistically optimal number of clusters?
3. **Which customer group is the most valuable?** — Where is the bank's revenue concentrated?
4. **Which group is least active?** — Who is most at risk of churning?
5. **Are there abnormal customers in the data?** — Can we flag outliers that may indicate fraud, data errors, or extreme behaviour?
6. **What does customer behaviour look like in reduced 2D space?** — Can we visually confirm that clusters are meaningfully distinct?

---

## 🔬 Analytical Workflow

### 1. Data Ingestion & Cleaning
- Loaded `BankChurners.csv` and performed a full inspection: shape, data types, null values, and column naming conventions
- Standardised column names to `snake_case`
- Dropped two Naive Bayes classifier artifact columns appended to the original dataset

### 2. Feature Engineering
Two new composite features were derived to improve clustering signal:

```python
# Composite transaction value proxy
df["customer_value"] = df["total_trans_amt"] * df["total_trans_ct"]

# Actual credit consumed by the customer
df["credit_used"] = df["credit_limit"] - df["avg_open_to_buy"]
```

### 3. Exploratory Data Analysis (EDA)
- **Correlation heatmap** — identified co-moving features and multicollinearity
- **Histograms** — examined distribution shape across all numeric variables
- **Boxplots** — detected outliers in each numeric feature before scaling

### 4. Feature Scaling
All numeric features standardised using `StandardScaler` to ensure no single variable dominates clustering due to scale differences.

### 5. Clustering

| Method | Purpose | Key Parameter |
|---|---|---|
| **K-Means** | Primary segmentation | `n_clusters = 4` (Elbow Method) |
| **DBSCAN** | Density-based validation & noise detection | `eps=0.8`, `min_samples=10` |
| **Hierarchical (Ward)** | Dendrogram for cluster structure validation | — |

The **Elbow Method** (WCSS vs. K) confirmed 4 as the optimal number of clusters.

### 6. Dimensionality Reduction & Visualisation

| Technique | Use Case |
|---|---|
| **PCA** (2 components) | Variance-preserving 2D projection for cluster scatter plots |
| **t-SNE** (2 components) | Local-structure-preserving projection for visual validation |

### 7. Anomaly Detection
**Isolation Forest** was applied on the scaled data with `contamination=0.02`, flagging approximately 2% of customers as anomalies — visualised on the PCA scatter plot for immediate interpretability.

---

## 💡 Key Insights

### Cluster Profiles

| Cluster | Label | Profile Summary |
|---|---|---|
| **Cluster 0** | Moderate Dependents | Mid-age customers with higher dependent counts; moderate credit usage and transaction activity. A stable but unspectacular segment. |
| **Cluster 1** | Low Engagers ⚠️ | Lowest credit limits, lowest transaction amounts and counts, highest inactivity months. **Primary churn risk group.** |
| **Cluster 2** | Premium Actives ⭐ | Highest credit limits, transaction volumes, and engineered `customer_value` score (~1.44M average). **The bank's most valuable segment.** |
| **Cluster 3** | Mid-Tier Engaged | Solid credit access and moderate-to-good transaction activity. A growth opportunity segment sitting between Clusters 0 and 2. |

### Anomaly Detection
- Isolation Forest identified a small cohort (~2%) of customers with behaviour patterns significantly deviating from the population norm
- These anomalies were visually confirmed in the PCA scatter plot and likely represent candidates for **fraud review**, **data quality investigation**, or **extreme use-case profiling**

### Behavioural Space (PCA & t-SNE)
- PCA projections confirmed that the four K-Means clusters occupy meaningfully distinct regions in 2D space — clusters are not arbitrary
- t-SNE projections revealed tighter intra-cluster cohesion, validating that customers within each segment genuinely behave similarly

---

## 📋 Strategic Recommendations

### 1. 🛡️ Retain Cluster 1 — Proactive Re-engagement
Cluster 1 (Low Engagers) shows all the hallmarks of pre-churn behaviour: low credit usage, low transaction frequency, and high inactivity. The bank should:
- Launch **personalised re-engagement campaigns** (cashback incentives, credit limit reviews, product recommendations)
- Set **automated inactivity alerts** triggered at the 3-month mark — before the customer mentally checks out
- Assign relationship managers or digital nudges based on customer age and card type

### 2. 💎 Protect and Grow Cluster 2 — Premium Loyalty Programme
Cluster 2 generates the highest customer value. Losing even a fraction of this group is disproportionately costly. The bank should:
- Prioritise **premium service access** (dedicated support lines, faster dispute resolution)
- Offer **exclusive product upgrades** — premium cards, higher credit limits, wealth management referrals
- Conduct **regular satisfaction surveys** to surface pain points before they become churn triggers

### 3. 📈 Convert Cluster 3 — Upsell & Cross-sell Pipeline
Cluster 3 sits just below the premium tier, with solid engagement and credit access. This is the bank's most fertile **upsell segment**. Recommended actions:
- Identify the behavioural gap between Clusters 3 and 2 and design targeted product nudges to close it
- Offer **credit limit increases** to high-utilisation Cluster 3 customers
- Introduce **rewards acceleration** campaigns to grow transaction volumes

### 4. 🔍 Investigate Anomalies — Risk & Compliance Review
The ~2% flagged by Isolation Forest should not be ignored. The bank should:
- Route anomaly-flagged accounts to a **fraud review queue** for manual or automated screening
- Cross-reference anomalies with chargeback history, dispute frequency, and KYC completeness
- Establish a **rolling anomaly detection pipeline** to flag new anomalies in real time as new transaction data arrives

### 5. 🏗️ Operationalise Segmentation — Real-Time Scoring
Clustering run on a static snapshot is a starting point, not a destination. The bank should:
- Retrain the segmentation model **quarterly** or after major portfolio events
- Integrate cluster labels into the **CRM system** so relationship managers see each customer's segment in real time
- Build a **cluster migration tracker** to monitor customers moving from high-value to low-engagement segments (an early churn warning signal)

---

## 📁 Project Outputs

| File | Description |
|---|---|
| `BankChurners_Clustered.csv` | Full dataset with `Cluster`, `Anomaly`, `PCA1/2`, `TSNE1/2` columns appended |
| `Customer_Clusters.html` | Interactive PCA scatter plot coloured by K-Means cluster |
| `TSNE_Clusters.html` | Interactive t-SNE scatter plot coloured by cluster |
| `Anomaly_Detection.html` | Interactive PCA scatter plot coloured by Isolation Forest anomaly label |

All files are saved to the `outputs/` directory.

---

## ✅ Conclusion

This project demonstrates that unsupervised machine learning — applied thoughtfully to customer transactional data — can transform a bank's understanding of its portfolio from aggregate statistics into **individual-level behavioural intelligence**.

Four distinct customer personas emerge from the data: a disengaged at-risk group, a stable moderate group, a growth-ready mid-tier, and a high-value premium core. Each segment demands a different strategy, and treating all customers identically is a guaranteed path to avoidable churn and missed revenue.

Combined with anomaly detection, this analysis equips the bank's analytics, marketing, and risk teams with a single coherent view of customer behaviour — and a practical roadmap for acting on it.

> **The best retention strategy is not reactive — it is built on knowing who your customers are before they decide to leave.**

---

## 👤 Author

**Aminu Abiola Friday**
Operations Analyst | Data Analytics Professional

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/aminu-abiola-friday)
[![GitHub](https://img.shields.io/badge/GitHub-Portfolio-black?logo=github)](https://github.com/aminuabiolaFriday)

---

*If you found this project useful, please consider starring ⭐ the repository.*
