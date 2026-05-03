"""
Joins all Olist tables into one wide analytical DataFrame keyed on customer_unique_id.
Also attaches RetailRocket engagement signals where available.
"""
import logging
import os

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

from src.config.constants import PROCESSED_DIR, FEATURES_DIR
from src.utils.spark_session import get_spark_session

logger = logging.getLogger(__name__)


def load_processed(spark: SparkSession, name: str) -> DataFrame:
    path = os.path.join(PROCESSED_DIR, name)
    return spark.read.parquet(path)


def build_olist_wide(spark: SparkSession) -> DataFrame:
    """
    Joins: orders → customers → order_items → payments → reviews → products → category
    Output keyed on order_id (later aggregated per customer in feature_engineering).
    """
    orders      = load_processed(spark, "olist_orders")
    customers   = load_processed(spark, "olist_customers")
    items       = load_processed(spark, "olist_order_items")
    payments    = load_processed(spark, "olist_payments")
    reviews     = load_processed(spark, "olist_reviews")
    products    = load_processed(spark, "olist_products")
    category    = load_processed(spark, "olist_category")

    # ── Aggregate payments per order (multiple payment_sequential rows) ────────
    payments_agg = (
        payments.groupBy("order_id")
        .agg(
            F.sum("payment_value").alias("total_payment_value"),
            F.countDistinct("payment_type").alias("num_payment_types"),
            F.avg("payment_installments").alias("avg_installments"),
            F.first("payment_type").alias("primary_payment_type"),
        )
    )

    # ── Aggregate items per order ──────────────────────────────────────────────
    items_agg = (
        items.groupBy("order_id")
        .agg(
            F.count("order_item_id").alias("num_items"),
            F.sum("price").alias("total_price"),
            F.sum("freight_value").alias("total_freight"),
            F.avg("price").alias("avg_item_price"),
            F.countDistinct("product_id").alias("num_products"),
            F.countDistinct("seller_id").alias("num_sellers"),
        )
    )

    # ── Best review score per order (take max to be conservative) ─────────────
    reviews_agg = (
        reviews.groupBy("order_id")
        .agg(F.max("review_score").alias("review_score"))
    )

    # ── Translate category names ───────────────────────────────────────────────
    products_with_category = (
        products
        .join(category, on="product_category_name", how="left")
        .select(
            "product_id",
            "product_category_name",
            F.coalesce(
                F.col("product_category_name_english"),
                F.col("product_category_name")
            ).alias("category_english"),
        )
    )

    # ── Items + category per order ─────────────────────────────────────────────
    items_cat = (
        items.select("order_id", "product_id")
        .join(products_with_category, on="product_id", how="left")
        .groupBy("order_id")
        .agg(
            F.collect_set("category_english").alias("categories"),
            F.size(F.collect_set("category_english")).alias("num_categories"),
        )
    )

    # ── Combine all ───────────────────────────────────────────────────────────
    wide = (
        orders
        .join(customers,     on="customer_id",  how="left")
        .join(payments_agg,  on="order_id",     how="left")
        .join(items_agg,     on="order_id",     how="left")
        .join(reviews_agg,   on="order_id",     how="left")
        .join(items_cat,     on="order_id",     how="left")
        .filter(F.col("order_status").isin("delivered", "shipped", "invoiced", "processing"))
        .drop("customer_id")          # replaced by customer_unique_id
    )

    row_count = wide.count()
    logger.info(f"[join] Wide table built: {row_count:,} rows")
    return wide


def build_retailrocket_user_signals(spark: SparkSession) -> DataFrame:
    """
    Aggregate RetailRocket events into per-visitor engagement signals.
    view_count, cart_count, purchase_count, view_to_purchase_rate.
    """
    events = load_processed(spark, "retailrocket_events")

    signals = (
        events
        .groupBy("visitorid")
        .agg(
            F.count(F.when(F.col("event") == "view", 1)).alias("rr_view_count"),
            F.count(F.when(F.col("event") == "addtocart", 1)).alias("rr_cart_count"),
            F.count(F.when(F.col("event") == "transaction", 1)).alias("rr_purchase_count"),
            F.countDistinct("itemid").alias("rr_unique_items"),
        )
        .withColumn(
            "rr_view_to_purchase_rate",
            F.when(F.col("rr_view_count") > 0,
                   F.col("rr_purchase_count") / F.col("rr_view_count"))
            .otherwise(0.0)
        )
    )

    logger.info(f"[join] RetailRocket signals: {signals.count():,} visitors")
    return signals


def save_wide_table(df: DataFrame, name: str = "olist_wide") -> None:
    out_path = os.path.join(PROCESSED_DIR, name)
    (
        df.write
        .mode("overwrite")
        .option("compression", "snappy")
        .partitionBy("customer_state")
        .parquet(out_path)
    )
    logger.info(f"[join] Saved wide table to {out_path}")


# ─── Entrypoint ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    spark = get_spark_session()

    wide = build_olist_wide(spark)
    save_wide_table(wide)

    rr_signals = build_retailrocket_user_signals(spark)
    rr_out = os.path.join(PROCESSED_DIR, "retailrocket_user_signals")
    rr_signals.write.mode("overwrite").parquet(rr_out)
    logger.info("=== Join Complete ===")
    spark.stop()