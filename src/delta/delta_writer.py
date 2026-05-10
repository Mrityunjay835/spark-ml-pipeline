"""
Delta Writer — converts all pipeline parquet writes to Delta format.

Why Delta over plain Parquet:
  ┌──────────────────────┬─────────────┬──────────────┐
  │ Feature              │ Parquet     │ Delta Lake   │
  ├──────────────────────┼─────────────┼──────────────┤
  │ ACID transactions    │ ❌          │ ✅           │
  │ Update / Delete      │ ❌          │ ✅           │
  │ Time travel          │ ❌          │ ✅           │
  │ Schema evolution     │ manual      │ ✅ automatic │
  │ Small file compaction│ ❌          │ ✅ OPTIMIZE  │
  │ Streaming + Batch    │ separate    │ ✅ unified   │
  └──────────────────────┴─────────────┴──────────────┘

Install:
  pip install delta-spark

SparkSession must have Delta extensions (already added to spark_session.py):
  .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
  .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
"""
import logging
import os
from typing import List, Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.config.constants import (
    PROCESSED_DIR,
    FEATURES_DIR,
    DELTA_DIR,
    DELTA_ORDERS,
    DELTA_CUSTOMERS,
    DELTA_WIDE,
    DELTA_TRANSFORMED,
    DELTA_USER_FEATURES,
    DELTA_PREDICTIONS,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# CORE WRITE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def write_delta(
    df:              DataFrame,
    path:            str,
    mode:            str = "overwrite",          # overwrite | append | ignore | error
    partition_cols:  Optional[List[str]] = None,
    merge_schema:    bool = False,               # allow new columns (schema evolution)
    overwrite_schema:bool = False,               # allow schema type changes
    comment:         str = "",
) -> None:
    """
    Write a DataFrame as a Delta table.

    Args:
        mode:             'overwrite' replaces entire table.
                          'append' adds rows (use for streaming or incremental loads).
        partition_cols:   Partition by these columns for faster partition-pruned reads.
        merge_schema:     True → new columns in df are added to Delta schema.
        overwrite_schema: True → schema can change completely (use with caution).
        comment:          Logged for audit trail.
    """
    os.makedirs(path, exist_ok=True)

    writer = (
        df.write
        .format("delta")
        .mode(mode)
    )

    if partition_cols:
        writer = writer.partitionBy(*partition_cols)

    if merge_schema:
        writer = writer.option("mergeSchema", "true")

    if overwrite_schema:
        writer = writer.option("overwriteSchema", "true")

    writer.save(path)

    row_count = df.count() if mode != "append" else -1
    logger.info(
        f"[delta_writer] ✅ Written Delta table → {path} | "
        f"mode={mode} | "
        f"partitions={partition_cols} | "
        f"rows={row_count:,}" + (f" | {comment}" if comment else "")
    )


def read_delta(
    spark:           SparkSession,
    path:            str,
    version:         Optional[int] = None,       # time travel by version number
    timestamp:       Optional[str] = None,        # time travel by timestamp string
) -> DataFrame:
    """
    Read a Delta table. Optionally time-travel to a past version.

    Args:
        version:   Read data as of this version number.
                   Version 0 = first write. Get versions via delta_ops.get_history().
        timestamp: Read data as of this timestamp. e.g. "2024-01-01 00:00:00"

    Example:
        # Current version
        df = read_delta(spark, DELTA_USER_FEATURES)

        # As of version 3 (before last retrain)
        df = read_delta(spark, DELTA_USER_FEATURES, version=3)

        # As of yesterday
        df = read_delta(spark, DELTA_USER_FEATURES, timestamp="2024-05-09 00:00:00")
    """
    reader = spark.read.format("delta")

    if version is not None:
        reader = reader.option("versionAsOf", version)
        logger.info(f"[delta_writer] Time travel → {path} @ version={version}")
    elif timestamp is not None:
        reader = reader.option("timestampAsOf", timestamp)
        logger.info(f"[delta_writer] Time travel → {path} @ timestamp={timestamp}")

    return reader.load(path)


def delta_table_exists(path: str) -> bool:
    """Check if a Delta table exists at path."""
    delta_log = os.path.join(path, "_delta_log")
    return os.path.exists(delta_log)


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE-SPECIFIC WRITERS
# ══════════════════════════════════════════════════════════════════════════════

def write_orders_delta(df: DataFrame) -> None:
    write_delta(
        df, DELTA_ORDERS,
        mode="overwrite",
        partition_cols=["order_status"],
        comment="raw olist orders",
    )


def write_customers_delta(df: DataFrame) -> None:
    write_delta(
        df, DELTA_CUSTOMERS,
        mode="overwrite",
        partition_cols=["customer_state"],
        comment="raw olist customers",
    )


def write_wide_delta(df: DataFrame) -> None:
    write_delta(
        df, DELTA_WIDE,
        mode="overwrite",
        partition_cols=["customer_state"],
        comment="olist wide joined table",
    )


def write_transformed_delta(df: DataFrame) -> None:
    write_delta(
        df, DELTA_TRANSFORMED,
        mode="overwrite",
        partition_cols=["customer_state"],
        comment="cleaned + transformed wide table",
    )


def write_user_features_delta(df: DataFrame) -> None:
    write_delta(
        df, DELTA_USER_FEATURES,
        mode="overwrite",
        partition_cols=["customer_state"],
        comment="RFM + behavioral feature store",
    )


def write_predictions_delta(df: DataFrame, mode: str = "append") -> None:
    """
    Predictions are always appended — never overwrite historical predictions.
    Each streaming batch appends its predictions here.
    """
    write_delta(
        df, DELTA_PREDICTIONS,
        mode=mode,                     # append by default
        partition_cols=["risk_tier"],
        comment="model churn predictions",
    )


# ══════════════════════════════════════════════════════════════════════════════
# MIGRATE PARQUET → DELTA
# ══════════════════════════════════════════════════════════════════════════════

def migrate_parquet_to_delta(
    spark:         SparkSession,
    parquet_path:  str,
    delta_path:    str,
    partition_cols: Optional[List[str]] = None,
) -> None:
    """
    One-time migration: read existing parquet → write as Delta.
    Safe to run on any existing parquet dataset.

    After migration, update all reads to use read_delta() instead of spark.read.parquet().
    """
    if not os.path.exists(parquet_path):
        logger.warning(f"[delta_writer] Parquet path not found: {parquet_path}")
        return

    if delta_table_exists(delta_path):
        logger.info(f"[delta_writer] Delta table already exists at {delta_path}. Skipping.")
        return

    logger.info(f"[delta_writer] Migrating {parquet_path} → {delta_path}")
    df = spark.read.parquet(parquet_path)
    write_delta(df, delta_path, mode="overwrite", partition_cols=partition_cols)
    logger.info(f"[delta_writer] Migration complete: {delta_path}")


def migrate_all(spark: SparkSession) -> None:
    """Migrate all existing processed parquet tables to Delta."""
    migrations = [
        (os.path.join(PROCESSED_DIR, "olist_orders"),      DELTA_ORDERS,       ["order_status"]),
        (os.path.join(PROCESSED_DIR, "olist_customers"),   DELTA_CUSTOMERS,    ["customer_state"]),
        (os.path.join(PROCESSED_DIR, "olist_wide"),        DELTA_WIDE,         ["customer_state"]),
        (os.path.join(PROCESSED_DIR, "olist_transformed"), DELTA_TRANSFORMED,  ["customer_state"]),
        (os.path.join(FEATURES_DIR,  "user_features"),     DELTA_USER_FEATURES,["customer_state"]),
    ]

    logger.info(f"[delta_writer] Starting migration of {len(migrations)} tables...")
    for parquet_path, delta_path, partitions in migrations:
        migrate_parquet_to_delta(spark, parquet_path, delta_path, partitions)

    logger.info("[delta_writer] ✅ All tables migrated to Delta.")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.utils.spark_session import get_spark_session, stop_spark_session

    spark = get_spark_session()
    migrate_all(spark)
    stop_spark_session()