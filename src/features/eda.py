"""
EDA module: distributions, correlations, class balance, feature importance proxy.
All computations stay on Spark; only small summary frames collected to driver.
"""
import logging
import os
from typing import List, Optional

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler

from src.config.constants import PROCESSED_DIR, FEATURES_DIR, LABEL_COL
from src.utils.spark_session import get_spark_session

logger = logging.getLogger(__name__)


def class_balance(df: DataFrame, label_col: str = LABEL_COL) -> None:
    """Print class distribution and imbalance ratio."""
    dist = (
        df.groupBy(label_col)
        .count()
        .withColumn("pct", F.round(F.col("count") / df.count() * 100, 2))
        .orderBy(label_col)
    )
    dist.show()

    counts = {row[label_col]: row["count"] for row in dist.collect()}
    if 0 in counts and 1 in counts:
        ratio = counts[0] / counts[1]
        logger.info(f"[eda] Class imbalance ratio (neg/pos): {ratio:.2f}")
        if ratio > 5:
            logger.warning(f"[eda] High imbalance ({ratio:.1f}x). Consider SMOTE / class weights.")


def numeric_summary(df: DataFrame, cols: Optional[List[str]] = None) -> DataFrame:
    """
    Summary stats for numeric columns: min, max, mean, stddev, p25, p50, p75.
    Avoids df.describe() — adds percentiles.
    """
    if cols is None:
        cols = [c for c, dtype in df.dtypes if dtype in ("double", "float", "int", "bigint", "long")]

    # Percentiles via approxQuantile (single pass)
    quantiles = df.approxQuantile(cols, [0.25, 0.5, 0.75], 0.01)
    quantile_map = dict(zip(cols, quantiles))

    agg_exprs = []
    for col_name in cols:
        agg_exprs += [
            F.min(col_name).alias(f"{col_name}_min"),
            F.max(col_name).alias(f"{col_name}_max"),
            F.avg(col_name).alias(f"{col_name}_mean"),
            F.stddev(col_name).alias(f"{col_name}_std"),
            F.sum(F.col(col_name).isNull().cast("int")).alias(f"{col_name}_nulls"),
        ]

    summary_row = df.agg(*agg_exprs).collect()[0].asDict()

    # Print formatted
    print(f"\n{'Col':<35} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10} "
          f"{'P25':>10} {'P50':>10} {'P75':>10} {'Nulls':>8}")
    print("-" * 115)
    for i, col_name in enumerate(cols):
        q = quantile_map.get(col_name, [None, None, None])
        print(
            f"{col_name:<35} "
            f"{_fmt(summary_row.get(f'{col_name}_min')):>10} "
            f"{_fmt(summary_row.get(f'{col_name}_max')):>10} "
            f"{_fmt(summary_row.get(f'{col_name}_mean')):>10} "
            f"{_fmt(summary_row.get(f'{col_name}_std')):>10} "
            f"{_fmt(q[0] if q else None):>10} "
            f"{_fmt(q[1] if q else None):>10} "
            f"{_fmt(q[2] if q else None):>10} "
            f"{summary_row.get(f'{col_name}_nulls', 0):>8}"
        )

    return df.select(cols)


def _fmt(val) -> str:
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.2f}"
    return str(val)


def categorical_distribution(df: DataFrame, col_name: str, top_n: int = 20) -> None:
    """Value counts + pct for a categorical column."""
    total = df.count()
    (
        df.groupBy(col_name)
        .count()
        .withColumn("pct", F.round(F.col("count") / total * 100, 2))
        .orderBy(F.col("count").desc())
        .limit(top_n)
        .show(truncate=False)
    )


def correlation_matrix(df: DataFrame, numeric_cols: List[str]) -> None:
    """
    Pearson correlation matrix via Spark MLlib.
    Collects to driver — only feasible for <50 features.
    """
    assembler = VectorAssembler(inputCols=numeric_cols, outputCol="_corr_vec",
                                handleInvalid="skip")
    vec_df = assembler.transform(df).select("_corr_vec")
    matrix = Correlation.corr(vec_df, "_corr_vec", method="pearson").head()[0]

    # Print top correlations with label
    arr = matrix.toArray()
    print(f"\nCorrelation matrix ({len(numeric_cols)}x{len(numeric_cols)}):")
    pairs = []
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            pairs.append((numeric_cols[i], numeric_cols[j], arr[i][j]))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    for c1, c2, corr in pairs[:20]:
        print(f"  {c1:35} ↔ {c2:35} : {corr:+.4f}")


def churn_feature_stats(df: DataFrame, label_col: str = LABEL_COL) -> None:
    """Mean of each numeric feature by churn label — quick signal check."""
    numeric_cols = [c for c, dtype in df.dtypes
                    if dtype in ("double", "float", "int", "bigint", "long")
                    and c != label_col]

    agg_exprs = [F.avg(c).alias(c) for c in numeric_cols]
    (
        df.groupBy(label_col)
        .agg(*agg_exprs)
        .orderBy(label_col)
        .show(truncate=False)
    )


def check_data_drift(
    reference: DataFrame,
    current: DataFrame,
    numeric_cols: List[str],
    threshold: float = 0.1,
) -> List[str]:
    """
    Simple mean-shift drift detection. Returns list of drifted columns.
    For prod: replace with PSI or KS test on collected samples.
    """
    drifted = []
    ref_means = {row["col"]: row["mean"] for row in
                 reference.select([F.avg(c).alias(c) for c in numeric_cols])
                 .collect()[0].asDict().items()
                 if row != "col"}

    # cleaner approach:
    ref_stats = reference.select([F.avg(c).alias(c) for c in numeric_cols]).collect()[0].asDict()
    cur_stats  = current.select([F.avg(c).alias(c) for c in numeric_cols]).collect()[0].asDict()

    for col_name in numeric_cols:
        ref_val = ref_stats.get(col_name)
        cur_val = cur_stats.get(col_name)
        if ref_val and cur_val and ref_val != 0:
            shift = abs(cur_val - ref_val) / abs(ref_val)
            if shift > threshold:
                drifted.append(col_name)
                logger.warning(
                    f"[eda:drift] '{col_name}': ref={ref_val:.4f}, "
                    f"cur={cur_val:.4f}, shift={shift:.2%}"
                )

    return drifted


# ─── Entrypoint ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    spark = get_spark_session()

    features_path = os.path.join(FEATURES_DIR, "user_features")
    df = spark.read.parquet(features_path)

    print("\n=== Class Balance ===")
    class_balance(df)

    print("\n=== Numeric Summary ===")
    numeric_summary(df)

    print("\n=== Churn Feature Stats ===")
    churn_feature_stats(df)

    numeric_cols = [c for c, dtype in df.dtypes
                    if dtype in ("double", "float", "int", "bigint", "long")
                    and c != LABEL_COL]

    print("\n=== Correlation Matrix ===")
    correlation_matrix(df, numeric_cols[:10])  # cap to 10 for readability

    print("\n=== Customer State Distribution ===")
    categorical_distribution(df, "customer_state")

    spark.stop()