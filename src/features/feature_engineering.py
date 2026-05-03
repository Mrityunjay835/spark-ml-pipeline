"""
Feature engineering: RFM (Recency/Frequency/Monetary), behavioral features,
churn label computation. Outputs user-level feature store.
"""
import logging
import os
from datetime import datetime

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src.config.constants import (
    PROCESSED_DIR, FEATURES_DIR,
    CHURN_DAYS_THRESHOLD, LABEL_COL,
)
from src.utils.spark_session import get_spark_session

logger = logging.getLogger(__name__)


def load_transformed(spark: SparkSession) -> DataFrame:
    return spark.read.parquet(os.path.join(PROCESSED_DIR, "olist_transformed"))


def compute_rfm(df: DataFrame, reference_date: str = None) -> DataFrame:
    """
    RFM per customer_unique_id.
    reference_date: ISO string e.g. '2018-09-01'. Defaults to max(order_purchase_timestamp).
    """
    if reference_date:
        ref_ts = F.to_timestamp(F.lit(reference_date))
    else:
        ref_ts = df.select(F.max("order_purchase_timestamp")).collect()[0][0]
        ref_ts = F.lit(ref_ts)

    rfm = (
        df.groupBy("customer_unique_id")
        .agg(
            # Recency: days since last order
            F.datediff(ref_ts, F.max("order_purchase_timestamp")).alias("recency_days"),
            # Frequency: number of orders
            F.countDistinct("order_id").alias("frequency"),
            # Monetary: total spend
            F.sum("total_payment_value").alias("monetary"),
            # Avg order value
            F.avg("total_payment_value").alias("avg_order_value"),
        )
    )
    return rfm


def compute_behavioral_features(df: DataFrame) -> DataFrame:
    """
    Per-customer behavioral aggregations:
    - avg review score, review variance
    - payment diversity
    - freight ratio
    - weekend purchase ratio
    - category diversity
    - delivery delay avg
    """
    behavioral = (
        df.groupBy("customer_unique_id")
        .agg(
            # Review behavior
            F.avg("review_score").alias("avg_review_score"),
            F.stddev("review_score").alias("std_review_score"),
            F.min("review_score").alias("min_review_score"),

            # Payment behavior
            F.avg("num_payment_types").alias("avg_payment_types"),
            F.avg("avg_installments").alias("avg_installments"),
            F.first("primary_payment_type").alias("primary_payment_type"),

            # Freight
            F.sum("total_freight").alias("total_freight"),
            F.avg(
                F.when(F.col("total_payment_value") > 0,
                       F.col("total_freight") / F.col("total_payment_value"))
                .otherwise(0.0)
            ).alias("avg_freight_ratio"),

            # Time of purchase
            F.avg(
                F.when(F.col("purchase_dow").isin(1, 7), 1).otherwise(0)
            ).alias("weekend_purchase_ratio"),
            F.avg("purchase_hour").alias("avg_purchase_hour"),

            # Category diversity
            F.avg("num_categories").alias("avg_categories_per_order"),
            F.size(F.flatten(F.collect_list("categories"))).alias("total_category_count"),

            # Delivery delay
            F.avg("delivery_delay_days").alias("avg_delivery_delay"),
            F.max("delivery_delay_days").alias("max_delivery_delay"),

            # Items per order
            F.avg("num_items").alias("avg_items_per_order"),
            F.sum("num_items").alias("total_items_purchased"),

            # Geographic
            F.first("customer_state").alias("customer_state"),
        )
    )
    return behavioral


def compute_churn_label(df: DataFrame, threshold_days: int = CHURN_DAYS_THRESHOLD) -> DataFrame:
    """
    Churn label: 1 if customer's recency_days > threshold_days, else 0.
    Binary classification target.
    """
    return df.withColumn(
        LABEL_COL,
        F.when(F.col("recency_days") > threshold_days, 1).otherwise(0).cast("int")
    )


def compute_purchase_trend(df: DataFrame) -> DataFrame:
    """
    Per-customer trend: is spend increasing or decreasing over time?
    Simple: compare last 2 orders avg spend vs all-time avg.
    """
    w_last2 = Window.partitionBy("customer_unique_id").orderBy(
        F.col("order_purchase_timestamp").desc()
    )

    last2_spend = (
        df
        .withColumn("rn", F.row_number().over(w_last2))
        .filter(F.col("rn") <= 2)
        .groupBy("customer_unique_id")
        .agg(F.avg("total_payment_value").alias("last2_avg_spend"))
    )

    all_time_spend = (
        df.groupBy("customer_unique_id")
        .agg(F.avg("total_payment_value").alias("alltime_avg_spend"))
    )

    trend = (
        all_time_spend.join(last2_spend, on="customer_unique_id", how="left")
        .withColumn(
            "spend_trend",
            F.when(F.col("alltime_avg_spend") > 0,
                   (F.col("last2_avg_spend") - F.col("alltime_avg_spend")) /
                   F.col("alltime_avg_spend"))
            .otherwise(0.0)
        )
        .select("customer_unique_id", "spend_trend")
    )
    return trend


def build_user_features(spark: SparkSession) -> DataFrame:
    """
    Main feature builder: joins RFM + behavioral + trend → single user feature row.
    Output is written to feature store (FEATURES_DIR/user_features).
    """
    df = load_transformed(spark)

    logger.info("[features] Computing RFM...")
    rfm = compute_rfm(df)

    logger.info("[features] Computing behavioral features...")
    behavioral = compute_behavioral_features(df)

    logger.info("[features] Computing purchase trend...")
    trend = compute_purchase_trend(df)

    # Join all feature groups
    user_features = (
        rfm
        .join(behavioral, on="customer_unique_id", how="inner")
        .join(trend, on="customer_unique_id", how="left")
        .fillna({"spend_trend": 0.0, "std_review_score": 0.0})
    )

    # Add churn label
    user_features = compute_churn_label(user_features)

    # Add derived features
    user_features = user_features.withColumn(
        "high_value_customer",
        F.when(F.col("monetary") > user_features.approxQuantile("monetary", [0.75], 0.01)[0], 1)
        .otherwise(0)
    ).withColumn(
        "is_repeat_buyer",
        F.when(F.col("frequency") > 1, 1).otherwise(0)
    )

    total = user_features.count()
    churn_rate = user_features.filter(F.col(LABEL_COL) == 1).count() / total
    logger.info(f"[features] Total users: {total:,} | Churn rate: {churn_rate:.2%}")

    return user_features


def save_user_features(df: DataFrame) -> None:
    out_path = os.path.join(FEATURES_DIR, "user_features")
    (
        df.write
        .mode("overwrite")
        .option("compression", "snappy")
        .partitionBy("customer_state")
        .parquet(out_path)
    )
    logger.info(f"[features] User features saved → {out_path}")


# ─── Entrypoint ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    spark = get_spark_session()

    user_features = build_user_features(spark)
    save_user_features(user_features)

    logger.info("=== Feature Engineering Complete ===")
    user_features.printSchema()
    user_features.describe().show()
    spark.stop()