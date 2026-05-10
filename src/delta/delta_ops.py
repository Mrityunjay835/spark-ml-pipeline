"""
Delta Lake Operations — Phase: Delta

Covers the 5 killer features of Delta that plain Parquet can't do:
  1. MERGE (upsert)     — update existing rows, insert new ones atomically
  2. DELETE             — GDPR compliance, remove specific rows
  3. UPDATE             — correct bad data without rewriting full table
  4. Time Travel        — read any past version
  5. VACUUM             — clean up old files, control storage cost
  6. OPTIMIZE           — compact small files (critical for streaming tables)
  7. Schema Evolution   — add/rename columns without breaking readers
  8. History            — full audit log of every operation
"""
import logging
import os
from typing import Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.config.constants import (
    DELTA_USER_FEATURES,
    DELTA_PREDICTIONS,
    DELTA_TRANSFORMED,
    DELTA_WIDE,
)
from src.delta.delta_writer import read_delta, write_delta, delta_table_exists

logger = logging.getLogger(__name__)


def _get_delta_table(path: str):
    """
    Load a DeltaTable object for write operations (merge/delete/update).
    Requires delta-spark installed.
    """
    try:
        from delta.tables import DeltaTable
        return DeltaTable.forPath(_get_spark(), path)
    except ImportError:
        raise ImportError(
            "delta-spark not installed. Run: pip install delta-spark"
        )


def _get_spark() -> SparkSession:
    spark = SparkSession.getActiveSession()
    if spark is None:
        raise RuntimeError("No active SparkSession. Call get_spark_session() first.")
    return spark


# ══════════════════════════════════════════════════════════════════════════════
# 1. MERGE (UPSERT)
# ══════════════════════════════════════════════════════════════════════════════

def upsert_user_features(new_df: DataFrame) -> None:
    """
    Merge new user features into the Delta feature store.

    Logic:
      - If customer_unique_id already exists → UPDATE all feature columns
      - If customer_unique_id is new → INSERT the new row

    This is the production pattern for incremental feature store updates.
    You don't rewrite the entire table — only affected rows are touched.

    Use case: daily feature refresh — update features for customers
    who had activity in the last 24 hours.
    """
    if not delta_table_exists(DELTA_USER_FEATURES):
        logger.info("[delta_ops] Feature store doesn't exist yet — doing initial write.")
        write_delta(new_df, DELTA_USER_FEATURES, mode="overwrite",
                    partition_cols=["customer_state"])
        return

    from delta.tables import DeltaTable
    spark = _get_spark()

    delta_table = DeltaTable.forPath(spark, DELTA_USER_FEATURES)

    (
        delta_table.alias("existing")
        .merge(
            new_df.alias("updates"),
            "existing.customer_unique_id = updates.customer_unique_id"
        )
        .whenMatchedUpdateAll()     # UPDATE all columns when ID matches
        .whenNotMatchedInsertAll()  # INSERT new rows when ID is new
        .execute()
    )

    logger.info(
        f"[delta_ops] ✅ Upserted user features | "
        f"incoming rows={new_df.count():,}"
    )


def upsert_predictions(new_predictions: DataFrame) -> None:
    """
    Merge predictions: if same customer already has a prediction today,
    update it. Otherwise insert.

    Prevents duplicate predictions for the same customer in the same day.
    """
    if not delta_table_exists(DELTA_PREDICTIONS):
        write_delta(new_predictions, DELTA_PREDICTIONS, mode="overwrite",
                    partition_cols=["risk_tier"])
        return

    from delta.tables import DeltaTable
    spark = _get_spark()

    # Add prediction date for dedup key
    new_with_date = new_predictions.withColumn(
        "prediction_date", F.to_date(F.col("processed_at"))
    )

    delta_table = DeltaTable.forPath(spark, DELTA_PREDICTIONS)

    (
        delta_table.alias("existing")
        .merge(
            new_with_date.alias("new"),
            """
            existing.customer_unique_id = new.customer_unique_id
            AND existing.prediction_date = new.prediction_date
            """
        )
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )

    logger.info(f"[delta_ops] ✅ Upserted predictions.")


# ══════════════════════════════════════════════════════════════════════════════
# 2. DELETE
# ══════════════════════════════════════════════════════════════════════════════

def delete_customer(customer_id: str, table_path: str = None) -> None:
    """
    GDPR deletion: remove all rows for a specific customer.
    Plain Parquet cannot do this — you'd have to rewrite the entire partition.

    Delta deletes are:
      - Atomic (transaction logged)
      - Auditable (history shows the delete operation)
      - Reversible via time travel (until VACUUM runs)
    """
    path = table_path or DELTA_USER_FEATURES

    if not delta_table_exists(path):
        logger.warning(f"[delta_ops] Table not found: {path}")
        return

    from delta.tables import DeltaTable
    spark = _get_spark()

    delta_table = DeltaTable.forPath(spark, path)
    delta_table.delete(
        F.col("customer_unique_id") == customer_id
    )
    logger.info(
        f"[delta_ops] 🗑️  Deleted customer '{customer_id}' from {path}"
    )


def delete_old_predictions(days_to_keep: int = 30) -> None:
    """
    Remove predictions older than N days to control storage cost.
    Run this as a scheduled job (weekly).
    """
    if not delta_table_exists(DELTA_PREDICTIONS):
        return

    from delta.tables import DeltaTable
    spark = _get_spark()

    cutoff = F.date_sub(F.current_date(), days_to_keep)
    delta_table = DeltaTable.forPath(spark, DELTA_PREDICTIONS)
    delta_table.delete(F.col("processed_at") < cutoff)

    logger.info(
        f"[delta_ops] 🗑️  Deleted predictions older than {days_to_keep} days."
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3. UPDATE
# ══════════════════════════════════════════════════════════════════════════════

def update_churn_label(
    customer_id: str,
    new_label:   int,
    table_path:  str = None,
) -> None:
    """
    Correct a churn label for a specific customer.
    Use case: customer service confirms a customer is NOT churned
    despite model prediction.

    In plain Parquet: rewrite entire partition.
    In Delta: single row update, logged in transaction history.
    """
    path = table_path or DELTA_USER_FEATURES

    if not delta_table_exists(path):
        logger.warning(f"[delta_ops] Table not found: {path}")
        return

    from delta.tables import DeltaTable
    spark = _get_spark()

    delta_table = DeltaTable.forPath(spark, path)
    delta_table.update(
        condition=F.col("customer_unique_id") == customer_id,
        set={"churn_label": F.lit(new_label)}
    )
    logger.info(
        f"[delta_ops] ✏️  Updated churn_label={new_label} "
        f"for customer '{customer_id}'"
    )


def correct_bad_delivery_delays(table_path: str = None) -> None:
    """
    Example bulk correction: delivery delays < -30 days are likely data errors.
    Set them to 0 (no delay).

    In Parquet: read → filter → rewrite → overwrite. Risky, slow.
    In Delta: single UPDATE statement, atomic.
    """
    path = table_path or DELTA_TRANSFORMED

    if not delta_table_exists(path):
        logger.warning(f"[delta_ops] Table not found: {path}")
        return

    from delta.tables import DeltaTable
    spark = _get_spark()

    delta_table = DeltaTable.forPath(spark, path)
    delta_table.update(
        condition=F.col("avg_delivery_delay") < -30,
        set={"avg_delivery_delay": F.lit(0.0)}
    )
    logger.info("[delta_ops] ✏️  Corrected anomalous delivery delays.")


# ══════════════════════════════════════════════════════════════════════════════
# 4. TIME TRAVEL
# ══════════════════════════════════════════════════════════════════════════════

def get_history(path: str, limit: int = 10) -> DataFrame:
    """
    Show full operation history of a Delta table.
    Every write/merge/delete/update is logged here.

    Output columns:
      version, timestamp, operation, operationParameters, userMetadata
    """
    from delta.tables import DeltaTable
    spark = _get_spark()

    history = DeltaTable.forPath(spark, path).history(limit)

    logger.info(f"[delta_ops] History for {path}:")
    history.select(
        "version", "timestamp", "operation", "operationParameters"
    ).show(limit, truncate=False)

    return history


def rollback_to_version(
    spark:      SparkSession,
    path:       str,
    version:    int,
) -> None:
    """
    Rollback a Delta table to a previous version.

    Use case: pipeline wrote bad features → roll back feature store
    to last known-good version before the bad run.

    Implementation: read version N, overwrite current with overwriteSchema=True.
    The bad version is still in history (auditable) but current = version N.
    """
    logger.info(f"[delta_ops] ⏪ Rolling back {path} to version {version}...")

    old_df = read_delta(spark, path, version=version)
    write_delta(old_df, path, mode="overwrite", overwrite_schema=True)

    logger.info(f"[delta_ops] ✅ Rollback complete. Table is now at version {version} data.")


def compare_versions(
    spark:    SparkSession,
    path:     str,
    version_a: int,
    version_b: int,
    key_col:  str = "customer_unique_id",
    check_col: str = "churn_label",
) -> DataFrame:
    """
    Compare a column between two versions of a Delta table.
    Useful for debugging: what changed between training run A and run B?
    """
    df_a = read_delta(spark, path, version=version_a).select(
        key_col, F.col(check_col).alias(f"{check_col}_v{version_a}")
    )
    df_b = read_delta(spark, path, version=version_b).select(
        key_col, F.col(check_col).alias(f"{check_col}_v{version_b}")
    )

    diff = (
        df_a.join(df_b, on=key_col, how="inner")
        .filter(
            F.col(f"{check_col}_v{version_a}") !=
            F.col(f"{check_col}_v{version_b}")
        )
    )

    changed = diff.count()
    logger.info(
        f"[delta_ops] Version diff {version_a}→{version_b} on '{check_col}': "
        f"{changed:,} rows changed"
    )
    return diff


# ══════════════════════════════════════════════════════════════════════════════
# 5. VACUUM
# ══════════════════════════════════════════════════════════════════════════════

def vacuum_table(
    path:              str,
    retention_hours:   int = 168,    # 7 days default
    dry_run:           bool = True,  # always dry_run=True first in prod
) -> None:
    """
    VACUUM: delete old parquet files no longer referenced by Delta log.

    After updates/deletes/overwrites, Delta keeps old files for time travel.
    VACUUM removes files older than retention_hours.

    ⚠️  After VACUUM you CANNOT time travel to versions before the cutoff.
    Default retention = 168 hours (7 days). Minimum = 168 hours (enforced by Delta).

    Best practice:
      1. Run with dry_run=True first — see what would be deleted
      2. Confirm the list looks safe
      3. Run with dry_run=False

    Schedule: weekly in prod. Don't run after every write.
    """
    if not delta_table_exists(path):
        logger.warning(f"[delta_ops] Table not found: {path}")
        return

    from delta.tables import DeltaTable
    spark = _get_spark()

    # Delta enforces minimum 168h retention.
    # To override (dangerous in prod): set spark.databricks.delta.retentionDurationCheck.enabled=false
    if retention_hours < 168:
        logger.warning(
            f"[delta_ops] retention_hours={retention_hours} < 168. "
            f"Delta enforces minimum 168h. Overriding to 168."
        )
        retention_hours = 168

    delta_table = DeltaTable.forPath(spark, path)

    if dry_run:
        logger.info(f"[delta_ops] 🧹 VACUUM dry run on {path} (retention={retention_hours}h):")
        delta_table.vacuum(retentionHours=retention_hours)  # dry_run shows files
    else:
        logger.info(f"[delta_ops] 🧹 VACUUM executing on {path}...")
        delta_table.vacuum(retentionHours=retention_hours)
        logger.info("[delta_ops] ✅ VACUUM complete.")


# ══════════════════════════════════════════════════════════════════════════════
# 6. OPTIMIZE (compact small files)
# ══════════════════════════════════════════════════════════════════════════════

def optimize_table(spark: SparkSession, path: str) -> None:
    """
    OPTIMIZE: compact many small files into fewer large files.

    Critical for streaming tables — structured streaming writes one
    small file per micro-batch. After 1000 batches = 1000 tiny files.
    Reads become very slow (too many file open operations).

    OPTIMIZE merges small files into ~128MB target file size.
    Run after every N streaming batches or on a schedule.

    Note: OPTIMIZE is a Spark SQL command (not Python API).
    """
    if not delta_table_exists(path):
        logger.warning(f"[delta_ops] Table not found: {path}")
        return

    logger.info(f"[delta_ops] ⚡ OPTIMIZE running on {path}...")
    spark.sql(f"OPTIMIZE delta.`{path}`")
    logger.info(f"[delta_ops] ✅ OPTIMIZE complete.")


def optimize_with_zorder(
    spark:      SparkSession,
    path:       str,
    z_order_cols: list,
) -> None:
    """
    OPTIMIZE with Z-ORDER: co-locate related data in same files.
    Dramatically speeds up point queries and range filters.

    Example: Z-ORDER BY customer_state → all SP rows in same files
             → reading SP rows skips all other state files

    Use for the most common filter column in your queries.
    """
    cols = ", ".join(z_order_cols)
    logger.info(f"[delta_ops] ⚡ OPTIMIZE ZORDER BY ({cols}) on {path}...")
    spark.sql(f"OPTIMIZE delta.`{path}` ZORDER BY ({cols})")
    logger.info(f"[delta_ops] ✅ OPTIMIZE ZORDER complete.")


# ══════════════════════════════════════════════════════════════════════════════
# 7. TABLE STATS
# ══════════════════════════════════════════════════════════════════════════════

def table_stats(spark: SparkSession, path: str) -> None:
    """
    Print Delta table statistics: version count, file count, size.
    """
    if not delta_table_exists(path):
        logger.warning(f"[delta_ops] Not a Delta table: {path}")
        return

    from delta.tables import DeltaTable

    detail = spark.sql(f"DESCRIBE DETAIL delta.`{path}`")

    print(f"\n{'='*60}")
    print(f"DELTA TABLE: {path}")
    print(f"{'='*60}")
    detail.select(
        "format", "numFiles", "sizeInBytes", "partitionColumns"
    ).show(truncate=False)

    history = DeltaTable.forPath(spark, path).history(5)
    print("Recent history (last 5 operations):")
    history.select("version", "timestamp", "operation").show(truncate=False)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT — demo all Delta operations
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.utils.spark_session import get_spark_session, stop_spark_session
    from src.delta.delta_writer import migrate_all

    spark = get_spark_session()

    # Step 1: migrate existing parquet → Delta
    logger.info("=== Step 1: Migrate Parquet → Delta ===")
    migrate_all(spark)

    # Step 2: show history
    logger.info("=== Step 2: Table History ===")
    if delta_table_exists(DELTA_USER_FEATURES):
        get_history(DELTA_USER_FEATURES)

    # Step 3: time travel demo
    logger.info("=== Step 3: Time Travel ===")
    if delta_table_exists(DELTA_USER_FEATURES):
        df_v0 = read_delta(spark, DELTA_USER_FEATURES, version=0)
        logger.info(f"Version 0 row count: {df_v0.count():,}")

    # Step 4: table stats
    logger.info("=== Step 4: Table Stats ===")
    if delta_table_exists(DELTA_USER_FEATURES):
        table_stats(spark, DELTA_USER_FEATURES)

    # Step 5: optimize
    logger.info("=== Step 5: Optimize ===")
    if delta_table_exists(DELTA_USER_FEATURES):
        optimize_table(spark, DELTA_USER_FEATURES)

    # Step 6: vacuum dry run
    logger.info("=== Step 6: Vacuum (dry run) ===")
    if delta_table_exists(DELTA_USER_FEATURES):
        vacuum_table(DELTA_USER_FEATURES, retention_hours=168, dry_run=True)

    stop_spark_session()