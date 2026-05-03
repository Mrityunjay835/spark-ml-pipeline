"""
Batch ingestion: reads raw CSVs → validated Spark DataFrames.
Validates nulls on PK columns, logs row counts, writes to processed/.
"""
import logging
import os
from typing import Dict

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

from src.config.constants import (
    OLIST_DIR, RETAILROCKET_DIR, PROCESSED_DIR,
    OLIST_FILES, RETAILROCKET_FILES,
)
from src.config import schema as S
from src.utils.spark_session import get_spark_session

logger = logging.getLogger(__name__)


# ─── Schema registry: maps file key → StructType ──────────────────────────────
OLIST_SCHEMAS = {
    "customers":   S.CUSTOMERS_SCHEMA,
    "orders":      S.ORDERS_SCHEMA,
    "order_items": S.ORDER_ITEMS_SCHEMA,
    "payments":    S.PAYMENTS_SCHEMA,
    "reviews":     S.REVIEWS_SCHEMA,
    "products":    S.PRODUCTS_SCHEMA,
    "sellers":     S.SELLERS_SCHEMA,
    "geo":         S.GEO_SCHEMA,
    "category":    S.CATEGORY_SCHEMA,
}

RETAILROCKET_SCHEMAS = {
    "events":        S.EVENTS_SCHEMA,
    "category_tree": S.CATEGORY_TREE_SCHEMA,
    "item_props_1":  S.ITEM_PROPERTIES_SCHEMA,
    "item_props_2":  S.ITEM_PROPERTIES_SCHEMA,
}

# PK columns for null-check validation
PK_COLUMNS = {
    "customers":    "customer_id",
    "orders":       "order_id",
    "order_items":  "order_id",
    "payments":     "order_id",
    "reviews":      "review_id",
    "products":     "product_id",
    "sellers":      "seller_id",
    "events":       "visitorid",
}


def _read_csv(spark: SparkSession, path: str, schema, dataset_key: str) -> DataFrame:
    """Internal: read one CSV with schema enforcement and basic validation."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ingest] Missing file: {path}")

    df = (
        spark.read
        .option("header", "true")
        .option("timestampFormat", "yyyy-MM-dd HH:mm:ss")
        .option("mode", "PERMISSIVE")           # nullify bad rows, log via _corrupt_record
        .option("columnNameOfCorruptRecord", "_corrupt_record")
        .schema(schema)
        .csv(path)
    )

    # Drop fully corrupt rows
    if "_corrupt_record" in df.columns:
        corrupt_count = df.filter(F.col("_corrupt_record").isNotNull()).count()
        if corrupt_count > 0:
            logger.warning(f"[ingest:{dataset_key}] {corrupt_count} corrupt rows dropped.")
        df = df.drop("_corrupt_record")

    # PK null check
    pk = PK_COLUMNS.get(dataset_key)
    if pk and pk in df.columns:
        null_count = df.filter(F.col(pk).isNull()).count()
        if null_count > 0:
            logger.warning(f"[ingest:{dataset_key}] {null_count} rows with null PK '{pk}' found.")
        df = df.filter(F.col(pk).isNotNull())

    total = df.count()
    logger.info(f"[ingest:{dataset_key}] Loaded {total:,} rows from {path}")
    return df


def ingest_olist(spark: SparkSession) -> Dict[str, DataFrame]:
    """Load all Olist CSVs → dict of DataFrames."""
    dfs = {}
    for key, filename in OLIST_FILES.items():
        path = os.path.join(OLIST_DIR, filename)
        schema = OLIST_SCHEMAS[key]
        dfs[key] = _read_csv(spark, path, schema, key)
    return dfs


def ingest_retailrocket(spark: SparkSession) -> Dict[str, DataFrame]:
    """Load all RetailRocket CSVs → dict of DataFrames."""
    dfs = {}
    for key, filename in RETAILROCKET_FILES.items():
        path = os.path.join(RETAILROCKET_DIR, filename)
        schema = RETAILROCKET_SCHEMAS[key]
        dfs[key] = _read_csv(spark, path, schema, key)

    # Combine item_props_1 + item_props_2
    dfs["item_properties"] = dfs.pop("item_props_1").unionByName(dfs.pop("item_props_2"))
    logger.info("[ingest] item_properties_part1 + part2 merged.")
    return dfs


def write_processed(df: DataFrame, name: str, mode: str = "overwrite") -> None:
    """Persist processed DataFrame to parquet in PROCESSED_DIR."""
    out_path = os.path.join(PROCESSED_DIR, name)
    (
        df.write
        .mode(mode)
        .option("compression", "snappy")
        .parquet(out_path)
    )
    logger.info(f"[ingest] Written: {out_path}")


# ─── Entrypoint ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    spark = get_spark_session()

    logger.info("=== Ingesting Olist ===")
    olist = ingest_olist(spark)
    for name, df in olist.items():
        write_processed(df, f"olist_{name}")

    logger.info("=== Ingesting RetailRocket ===")
    rr = ingest_retailrocket(spark)
    for name, df in rr.items():
        write_processed(df, f"retailrocket_{name}")

    logger.info("=== Ingestion Complete ===")
    spark.stop()