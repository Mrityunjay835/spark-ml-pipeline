"""
Data transformation layer: deduplication, null handling, type normalization,
outlier capping. Runs after join, before feature engineering.
"""
import logging
import os

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src.config.constants import PROCESSED_DIR
from src.utils.spark_session import get_spark_session

logger = logging.getLogger(__name__)


def load_wide(spark: SparkSession) -> DataFrame:
    return spark.read.parquet(os.path.join(PROCESSED_DIR, "olist_wide"))


def deduplicate(df: DataFrame, pk: str = "order_id") -> DataFrame:
    """Keep latest row per PK using order_purchase_timestamp."""
    w = Window.partitionBy(pk).orderBy(F.col("order_purchase_timestamp").desc())
    deduped = (
        df.withColumn("_rn", F.row_number().over(w))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
    )
    dropped = df.count() - deduped.count()
    logger.info(f"[transform] Dedup dropped {dropped:,} duplicate rows on '{pk}'")
    return deduped


def handle_nulls(df: DataFrame) -> DataFrame:
    """
    Strategy per column type:
    - Numeric: fill with median (approximated via percentile_approx)
    - Categorical: fill with literal 'unknown'
    - review_score: fill with 3 (neutral)
    """
    # Numeric fills
    numeric_fills = {}
    for col_name, dtype in df.dtypes:
        if dtype in ("double", "float", "int", "bigint", "long"):
            if col_name in ("review_score",):
                numeric_fills[col_name] = 3
            else:
                # Approximate median — acceptable for prod; exact median is O(n log n)
                median_val = df.approxQuantile(col_name, [0.5], 0.01)
                numeric_fills[col_name] = median_val[0] if median_val else 0.0

    df = df.fillna(numeric_fills)

    # Categorical fills
    cat_cols = [col_name for col_name, dtype in df.dtypes if dtype == "string"]
    df = df.fillna({col: "unknown" for col in cat_cols})

    logger.info(f"[transform] Null handling complete. Numeric fills: {list(numeric_fills.keys())}")
    return df


def cap_outliers(df: DataFrame) -> DataFrame:
    """
    Winsorize numeric columns at 1st and 99th percentile.
    Prevents GBT splits from being dominated by extreme values.
    """
    numeric_cols = [
        "total_payment_value", "total_price", "total_freight",
        "avg_item_price", "num_items", "avg_installments"
    ]
    cap_cols = [c for c in numeric_cols if c in df.columns]

    for col_name in cap_cols:
        bounds = df.approxQuantile(col_name, [0.01, 0.99], 0.01)
        if len(bounds) == 2:
            lower, upper = bounds
            df = df.withColumn(
                col_name,
                F.when(F.col(col_name) < lower, lower)
                 .when(F.col(col_name) > upper, upper)
                 .otherwise(F.col(col_name))
            )

    logger.info(f"[transform] Outlier capping applied to: {cap_cols}")
    return df


def normalize_timestamps(df: DataFrame) -> DataFrame:
    """Cast string timestamps that slipped through, add derived date columns."""
    if "order_purchase_timestamp" in df.columns:
        df = df.withColumn(
            "purchase_year",  F.year("order_purchase_timestamp")
        ).withColumn(
            "purchase_month", F.month("order_purchase_timestamp")
        ).withColumn(
            "purchase_dow",   F.dayofweek("order_purchase_timestamp")
        ).withColumn(
            "purchase_hour",  F.hour("order_purchase_timestamp")
        )
    return df


def compute_delivery_delay(df: DataFrame) -> DataFrame:
    """Actual vs estimated delivery delta in days. Positive = late."""
    if all(c in df.columns for c in [
        "order_delivered_customer_date", "order_estimated_delivery_date"
    ]):
        df = df.withColumn(
            "delivery_delay_days",
            F.datediff(
                F.col("order_delivered_customer_date"),
                F.col("order_estimated_delivery_date")
            )
        )
    return df


def run_transform_pipeline(df: DataFrame) -> DataFrame:
    df = deduplicate(df)
    df = handle_nulls(df)
    df = cap_outliers(df)
    df = normalize_timestamps(df)
    df = compute_delivery_delay(df)
    return df


# ─── Entrypoint ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    spark = get_spark_session()

    raw_wide = load_wide(spark)
    transformed = run_transform_pipeline(raw_wide)

    out_path = os.path.join(PROCESSED_DIR, "olist_transformed")
    transformed.write.mode("overwrite").option("compression", "snappy").parquet(out_path)
    logger.info(f"[transform] Saved transformed data → {out_path}")
    spark.stop()