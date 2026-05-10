# 🔥 Spark ML Pipeline
### End-to-End Production Data Engineering & Machine Learning System

![Apache Spark](https://img.shields.io/badge/Apache%20Spark-3.5.3-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Delta Lake](https://img.shields.io/badge/Delta%20Lake-3.2.0-003366?style=for-the-badge)
![PySpark MLlib](https://img.shields.io/badge/MLlib-GBT%20%7C%20RF%20%7C%20LR-FF6B35?style=for-the-badge)

> A real-world, production-grade pipeline built on Apache Spark — covering batch ETL, feature engineering, ML training, structured streaming, Delta Lake, drift detection, and Spark UI debugging.

---

## 📌 What This Project Does

Predicts **customer churn** on the [Olist Brazilian E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) dataset using a full end-to-end Spark pipeline — the same architecture used at companies like Uber, Airbnb, and Shopify.

```
Raw CSVs → ETL → Feature Store → ML Model → Real-time Predictions → Delta Lake
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA SOURCES                                │
│   Olist E-Commerce (9 CSVs)    +    RetailRocket (Clickstream)      │
└──────────────────────┬──────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      BATCH PIPELINE (ETL)                           │
│                                                                     │
│   ingest.py → join_data.py → transform.py → read_parquet.py         │
│   Schema enforcement, null handling, dedup, outlier capping         │
└──────────────────────┬──────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FEATURE ENGINEERING                              │
│                                                                     │
│   EDA → feature_engineering.py                                      │
│   RFM features, behavioral signals, churn label (recency > 180d)    │
│   Output: user_features/ (partitioned by customer_state)            │
└──────────────────────┬──────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    ML TRAINING (Spark MLlib)                        │
│                                                                     │
│   pipeline.py → train.py                                            │
│   Imputer → StringIndexer → OHE → VectorAssembler → GBT/RF/LR      │
│   CrossValidator HPO, AUC-ROC evaluation, model serialization       │
└──────────────────────┬──────────────────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
┌─────────────────┐       ┌─────────────────────────────┐
│   DELTA LAKE    │       │    STRUCTURED STREAMING      │
│                 │       │                              │
│ Parquet + ACID  │       │  data_generator.py           │
│ Time Travel     │       │  → stream_predict.py         │
│ MERGE / DELETE  │       │  Real-time churn scoring     │
│ OPTIMIZE/VACUUM │       │  risk_tier: HIGH/MED/LOW     │
└─────────────────┘       └─────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 PRODUCTION MONITORING (Phase 6)                     │
│                                                                     │
│   drift_detector.py  → PSI + KS test on feature distributions       │
│   monitor.py         → row counts, null rates, prediction health    │
│   retrain_trigger.py → auto-retrain when drift > threshold          │
│   spark_optimizer.py → skew detection, broadcast advisor            │
│   debug_utils.py     → stage profiler, anti-pattern detector        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📂 Project Structure

```
spark-ml-project/
│
├── main.py                          # 🎯 Single orchestration entrypoint
├── Makefile                         # One-command pipeline execution
├── .gitignore
│
├── src/
│   ├── config/
│   │   ├── constants.py             # All paths, hyperparams, thresholds
│   │   └── schema.py                # Explicit StructType for all 13 CSVs
│   │
│   ├── batch/
│   │   ├── ingest.py                # Schema-enforced CSV ingestion
│   │   ├── join_data.py             # Multi-table joins → wide table
│   │   ├── transform.py             # Dedup, null handling, outlier cap
│   │   ├── read_parquet.py          # Partition-pruned parquet reads
│   │   └── query_partition.py       # Cohort retention, revenue queries
│   │
│   ├── features/
│   │   ├── eda.py                   # Class balance, distributions, drift
│   │   └── feature_engineering.py  # RFM + 20 behavioral features
│   │
│   ├── ml/
│   │   ├── pipeline.py              # Spark ML preprocessing pipeline
│   │   └── train.py                 # Train, evaluate, save model
│   │
│   ├── streaming/
│   │   ├── data_generator.py        # Synthetic event generator
│   │   └── stream_predict.py        # Structured streaming predictions
│   │
│   ├── delta/
│   │   ├── delta_writer.py          # Write/read Delta tables
│   │   └── delta_ops.py             # MERGE, DELETE, time travel, VACUUM
│   │
│   ├── monitoring/
│   │   ├── drift_detector.py        # PSI + KS + Chi-Square drift
│   │   ├── monitor.py               # Pipeline health monitoring
│   │   └── retrain_trigger.py       # Auto-retrain on drift
│   │
│   └── utils/
│       ├── spark_session.py         # Singleton SparkSession factory
│       ├── spark_optimizer.py       # Skew detection, broadcast advisor
│       └── debug_utils.py           # Stage profiler, plan analyzer
│
└── data/
    ├── raw/olist/                   # 9 Olist CSV files
    ├── raw/retailrocket/            # 4 RetailRocket CSV files
    ├── processed/                   # Intermediate parquet
    ├── features/user_features/      # Feature store (partitioned parquet)
    ├── delta/                       # Delta Lake tables
    └── streaming/input/             # Streaming landing zone
```

---

## 🗂️ Datasets

| Dataset | Source | Rows | Description |
|---|---|---|---|
| Olist Orders | Kaggle | 99,441 | Brazilian e-commerce orders 2016–2018 |
| Olist Customers | Kaggle | 99,441 | Customer location data |
| Olist Payments | Kaggle | 103,886 | Payment methods and values |
| Olist Reviews | Kaggle | 99,224 | Customer review scores |
| Olist Products | Kaggle | 32,951 | Product catalog |
| RetailRocket Events | Kaggle | 2.7M | Clickstream: view/cart/purchase |

**Download:**
```bash
# Olist
https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

# RetailRocket
https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset
```

Place files in:
```
data/raw/olist/
data/raw/retailrocket/
```

---

## ⚙️ Setup

**Requirements:**
- Java 11 or 17
- Python 3.11
- Apache Spark 3.5.3
- uv (Python package manager)

```bash
# 1. Clone the repo
git clone https://github.com/mrityunjay835/spark-ml-project.git
cd spark-ml-project

# 2. Create virtual environment and install dependencies
uv venv
uv add pyspark==3.5.3 delta-spark==3.2.0

# 3. Verify Spark installation
spark-submit --version   # should show 3.5.3

# 4. Download datasets and place in data/raw/

# 5. Run full pipeline
make all
```

---

## 🚀 Running the Pipeline

### Full pipeline (recommended)
```bash
make all
```

### With Delta Lake
```bash
make delta
```

### Individual stages
```bash
make ingest       # Load raw CSVs → parquet
make join         # Join all tables → wide table
make transform    # Clean, dedup, cap outliers
make features     # Build RFM + behavioral features
make train        # Train GBT model
make delta        # Migrate to Delta Lake
make stream       # Start streaming predictions
```

### Explore Spark UI while pipeline runs
```bash
spark-submit \
  --packages io.delta:delta-spark_2.12:3.2.0 \
  --conf "spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension" \
  --conf "spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog" \
  main.py --run ingest,join,transform,features,train --pause-ui

# Open → http://localhost:4040
```

### Streaming (two terminals)
```bash
# Terminal 1 — start predictions
make stream

# Terminal 2 — generate data
make generate
```

---

## 🤖 ML Model

### Target
**Binary churn classification** — customer inactive for > 180 days = churned.

### Features (20 total)
| Category | Features |
|---|---|
| Frequency | `frequency`, `avg_items_per_order`, `total_items_purchased` |
| Monetary | `monetary`, `avg_order_value`, `spend_trend` |
| Satisfaction | `avg_review_score`, `std_review_score`, `min_review_score` |
| Payment | `avg_installments`, `avg_freight_ratio` |
| Engagement | `weekend_purchase_ratio`, `avg_purchase_hour`, `avg_categories_per_order` |
| Operations | `avg_delivery_delay`, `max_delivery_delay`, `total_freight` |

### Results (GBT, no leakage)
| Metric | Score |
|---|---|
| AUC-ROC | 0.81 |
| AUC-PR | 0.86 |
| F1 | 0.73 |
| Accuracy | 0.74 |

### Supported classifiers
```bash
make train      # GBT (default)
make train-rf   # Random Forest
make train-lr   # Logistic Regression
make train-cv   # GBT + CrossValidator HPO
```

---

## 🌊 Delta Lake

All processed tables are migrated from Parquet to Delta:

```python
# Time travel — read any past version
df = read_delta(spark, DELTA_USER_FEATURES, version=0)

# Upsert — update existing rows, insert new
upsert_user_features(new_df)

# GDPR delete
delete_customer(customer_id="abc123")

# Compaction
optimize_table(spark, DELTA_USER_FEATURES)
```

| Table | Partition | Description |
|---|---|---|
| `delta/orders` | `order_status` | Raw orders |
| `delta/customers` | `customer_state` | Customer data |
| `delta/olist_wide` | `customer_state` | Joined wide table |
| `delta/olist_transformed` | `customer_state` | Cleaned data |
| `delta/user_features` | `customer_state` | Feature store |

---

## 📊 Monitoring

```bash
# Health check — row counts, null rates, anomalies
make monitor

# Debug — skew detection, execution plans, UI links
make debug

# Drift detection + auto-retrain if needed
python -m src.monitoring.retrain_trigger
```

### Drift thresholds
| Check | Threshold | Action |
|---|---|---|
| PSI per feature | ≥ 0.20 | Flag as drifted |
| Drifted features | ≥ 3 | Trigger retrain |
| Label drift | ≥ 5% churn shift | Trigger retrain |
| Model age | ≥ 30 days | Trigger retrain |

---

## 🔍 Key Concepts Covered

| Concept | Where |
|---|---|
| Schema enforcement | `config/schema.py`, `batch/ingest.py` |
| Partition pruning | `batch/read_parquet.py`, `batch/query_partition.py` |
| Window functions | `batch/transform.py`, `features/feature_engineering.py` |
| Broadcast joins | `utils/spark_optimizer.py` |
| Data skew + salting | `utils/spark_optimizer.py` |
| AQE (Adaptive Query Execution) | `utils/spark_session.py` |
| ML Pipeline (no leakage) | `ml/pipeline.py`, `ml/train.py` |
| Structured Streaming | `streaming/stream_predict.py` |
| Delta ACID transactions | `delta/delta_ops.py` |
| PSI drift detection | `monitoring/drift_detector.py` |
| Spark UI debugging | `utils/debug_utils.py` |

---

## 🧠 Interview Topics This Project Covers

- RDD vs DataFrame vs Dataset
- Shuffle operations and when they occur
- Partition pruning and predicate pushdown
- Data skew detection and salting
- Broadcast join vs SortMergeJoin
- Adaptive Query Execution (AQE)
- Feature leakage in ML pipelines
- Class imbalance handling
- Structured Streaming triggers and output modes
- Delta Lake vs plain Parquet (ACID, time travel)
- PSI-based drift detection
- Production retraining strategies

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Compute | Apache Spark 3.5.3 |
| Language | Python 3.11, PySpark |
| Storage (batch) | Parquet (Snappy) |
| Storage (prod) | Delta Lake 3.2.0 |
| ML | Spark MLlib (GBT, RF, LR) |
| Streaming | Spark Structured Streaming |
| Package manager | uv |
| Build | Makefile |

---

## 📝 License

MIT License — free to use for learning and portfolio projects.

---

<p align="center">Built for learning production-grade Spark engineering 🚀</p>