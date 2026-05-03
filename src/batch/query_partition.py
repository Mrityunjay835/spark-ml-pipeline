"""
Partition-aware query utilities.
Demonstrates predicate pushdown, partition pruning, and efficient aggregation
patterns on the partitioned parquet datasets.
"""
import logging
import os
from typing import List, Optional

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

from src.config.constants import PROCESSED_DIR, FEATURES_DIR
from src.utils.spark_session import get_spark_session

logger = logging.getLogger(__name__)


def read_partitioned(
    spark: SparkSession,
    table: str,
    states: Optional[List[str]] = None,
    base_dir: str = PROCESSED_DIR,
) -> DataFrame:
    """
    Read partitioned parquet with optional partition pruning on customer_state.
    Pushes filter into file scan — avoids reading irrelevant partitions.
    """
    path = os.path.join(base_dir, table)
    df = spark.read.parquet(path)

    if states:
        df = df.filter(F.col("customer_state").isin(states))
        logger.info(f"[query] Partition-pruned on states={states}")

    return df


def get_state_level_stats(spark: SparkSession, states: Optional[List[str]] = None) -> DataFrame:
    """
    State-level aggregations: avg spend, avg review, churn rate.
    Partition-pruned if states provided.
    """
    features = read_partitioned(spark, "user_features", states=states, base_dir=FEATURES_DIR)

    stats = (
        features
        .groupBy("customer_state")
        .agg(
            F.count("*").alias("num_customers"),
            F.avg("total_spend").alias("avg_spend"),
            F.avg("avg_review_score").alias("avg_review"),
            F.avg("churn_label").alias("churn_rate"),
            F.avg("total_orders").alias("avg_orders"),
            F.avg("days_since_last_order").alias("avg_recency_days"),
        )
        .orderBy(F.col("churn_rate").desc())
    )
    return stats


def get_top_churners(
    spark: SparkSession,
    n: int = 1000,
    states: Optional[List[str]] = None,
) -> DataFrame:
    """
    Retrieve top N customers by churn risk score (post-prediction).
    Falls back to heuristic: high recency + low orders if no prediction column.
    """
    features = read_partitioned(spark, "user_features", states=states, base_dir=FEATURES_DIR)

    if "churn_score" in features.columns:
        return features.orderBy(F.col("churn_score").desc()).limit(n)

    # Heuristic fallback
    return (
        features
        .filter(F.col("churn_label") == 1)
        .orderBy(F.col("days_since_last_order").desc(), F.col("total_orders").asc())
        .limit(n)
    )


def get_revenue_by_month(spark: SparkSession) -> DataFrame:
    """Monthly revenue trend from transformed wide table."""
    wide = spark.read.parquet(os.path.join(PROCESSED_DIR, "olist_transformed"))

    return (
        wide
        .filter(F.col("order_status") == "delivered")
        .groupBy("purchase_year", "purchase_month")
        .agg(
            F.sum("total_payment_value").alias("monthly_revenue"),
            F.count("order_id").alias("num_orders"),
            F.avg("review_score").alias("avg_review_score"),
            F.avg("total_freight").alias("avg_freight"),
        )
        .orderBy("purchase_year", "purchase_month")
    )


def get_cohort_retention(spark: SparkSession) -> DataFrame:
    """
    Monthly cohort retention: first purchase month vs subsequent months.
    Classic e-commerce churn analysis pattern.
    """
    wide = spark.read.parquet(os.path.join(PROCESSED_DIR, "olist_transformed"))

    # First purchase month per customer
    first_purchase = (
        wide.groupBy("customer_unique_id")
        .agg(
            F.min("purchase_year").alias("cohort_year"),
            F.min(
                F.when(F.col("purchase_year") == F.min("purchase_year").over(
                    __import__("pyspark.sql.window", fromlist=["Window"])
                    .Window.partitionBy("customer_unique_id")
                ), F.col("purchase_month"))
            ).alias("cohort_month"),
        )
    )

    # Simpler approach without nested window
    w = __import__("pyspark.sql.window", fromlist=["Window"]).Window.partitionBy("customer_unique_id")
    cohort = (
        wide
        .withColumn("first_purchase_ts", F.min("order_purchase_timestamp").over(w))
        .withColumn("cohort_year",  F.year("first_purchase_ts"))
        .withColumn("cohort_month", F.month("first_purchase_ts"))
        .withColumn("order_period_year",  F.col("purchase_year"))
        .withColumn("order_period_month", F.col("purchase_month"))
        .withColumn("months_since_cohort",
            (F.col("order_period_year") - F.col("cohort_year")) * 12 +
            (F.col("order_period_month") - F.col("cohort_month"))
        )
        .groupBy("cohort_year", "cohort_month", "months_since_cohort")
        .agg(F.countDistinct("customer_unique_id").alias("active_customers"))
        .orderBy("cohort_year", "cohort_month", "months_since_cohort")
    )
    return cohort


def get_category_performance(spark: SparkSession) -> DataFrame:
    """Category-level revenue and review performance."""
    wide = spark.read.parquet(os.path.join(PROCESSED_DIR, "olist_transformed"))

    return (
        wide
        .select(F.explode("categories").alias("category"), "total_price", "review_score")
        .groupBy("category")
        .agg(
            F.sum("total_price").alias("total_revenue"),
            F.count("*").alias("num_orders"),
            F.avg("review_score").alias("avg_review"),
        )
        .orderBy(F.col("total_revenue").desc())
    )


# ─── Entrypoint: run all analytical queries ───────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    spark = get_spark_session()

    logger.info("=== State-level stats ===")
    get_state_level_stats(spark).show(10, truncate=False)

    logger.info("=== Revenue by month ===")
    get_revenue_by_month(spark).show(24, truncate=False)

    logger.info("=== Top churners ===")
    get_top_churners(spark, n=20).show(truncate=False)

    logger.info("=== Category performance ===")
    get_category_performance(spark).show(20, truncate=False)

    spark.stop()