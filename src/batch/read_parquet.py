"""
Parquet read utilities: schema validation, partition listing, metadata inspection.
Used by all downstream modules instead of raw spark.read.parquet().
"""
import logging
import os
from typing import Optional, List, Dict, Any

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType

from src.config.constants import PROCESSED_DIR, FEATURES_DIR, MODELS_DIR
from src.utils.spark_session import get_spark_session

logger = logging.getLogger(__name__)


def read_parquet(
    spark: SparkSession,
    path: str,
    schema: Optional[StructType] = None,
    partition_filter: Optional[Dict[str, Any]] = None,
    cache: bool = False,
) -> DataFrame:
    """
    Read parquet with optional schema enforcement and partition filter pushdown.

    Args:
        schema:           If provided, enforce schema (prevents drift in prod).
        partition_filter: Dict of {partition_col: value} — enables partition pruning.
        cache:            Whether to cache the result (for reused DataFrames).
    """
    if not _path_exists(path):
        raise FileNotFoundError(f"[read_parquet] Path not found: {path}")

    reader = spark.read.option("mergeSchema", "false")

    if schema:
        reader = reader.schema(schema)

    df = reader.parquet(path)

    # Apply partition pruning filters
    if partition_filter:
        for col_name, val in partition_filter.items():
            if isinstance(val, list):
                df = df.filter(F.col(col_name).isin(val))
            else:
                df = df.filter(F.col(col_name) == val)
        logger.info(f"[read_parquet] Partition filter applied: {partition_filter}")

    if cache:
        df.cache()
        df.count()  # trigger materialization
        logger.info(f"[read_parquet] Cached DataFrame from {path}")

    return df


def read_user_features(
    spark: SparkSession,
    states: Optional[List[str]] = None,
    cache: bool = False,
) -> DataFrame:
    """Convenience: read user features with optional state partition filter."""
    pf = {"customer_state": states} if states else None
    return read_parquet(spark, FEATURES_DIR + "/user_features", partition_filter=pf, cache=cache)


def read_processed(spark: SparkSession, name: str, **kwargs) -> DataFrame:
    """Read a named processed dataset."""
    return read_parquet(spark, os.path.join(PROCESSED_DIR, name), **kwargs)


def inspect_parquet(spark: SparkSession, path: str) -> None:
    """Print schema, row count, and null stats for a parquet dataset."""
    if not _path_exists(path):
        logger.error(f"[inspect] Path not found: {path}")
        return

    df = spark.read.parquet(path)
    print(f"\n{'='*60}")
    print(f"Path: {path}")
    print(f"Partitions: {df.rdd.getNumPartitions()}")
    df.printSchema()

    # Row count
    count = df.count()
    print(f"Row count: {count:,}")

    # Null stats per column
    null_counts = df.select([
        F.sum(F.col(c).isNull().cast("int")).alias(c)
        for c in df.columns
    ]).collect()[0].asDict()

    print("\nNull counts:")
    for col_name, nc in null_counts.items():
        pct = (nc / count * 100) if count > 0 else 0
        if nc > 0:
            print(f"  {col_name}: {nc:,} ({pct:.1f}%)")

    print(f"{'='*60}\n")


def list_partitions(path: str) -> List[str]:
    """List partition directories under a partitioned parquet path."""
    if not os.path.exists(path):
        return []
    return [
        d for d in os.listdir(path)
        if os.path.isdir(os.path.join(path, d)) and "=" in d
    ]


def get_parquet_metadata(spark: SparkSession, path: str) -> Dict[str, Any]:
    """Return basic metadata dict for monitoring/logging."""
    df = spark.read.parquet(path)
    return {
        "path": path,
        "num_partitions": df.rdd.getNumPartitions(),
        "num_columns": len(df.columns),
        "columns": df.columns,
        "dtypes": dict(df.dtypes),
    }


def _path_exists(path: str) -> bool:
    """Works for local fs; override with s3a/hdfs check in prod."""
    return os.path.exists(path)


def validate_schema(df: DataFrame, expected_schema: StructType) -> List[str]:
    """
    Compare actual vs expected schema. Returns list of mismatches.
    Use before ML pipeline to catch upstream schema drift.
    """
    expected = {f.name: f.dataType.typeName() for f in expected_schema.fields}
    actual   = {name: dtype for name, dtype in df.dtypes}
    issues   = []

    for col_name, exp_type in expected.items():
        if col_name not in actual:
            issues.append(f"MISSING column: {col_name}")
        elif not actual[col_name].startswith(exp_type.lower()):
            issues.append(
                f"TYPE MISMATCH on '{col_name}': expected={exp_type}, actual={actual[col_name]}"
            )

    if issues:
        logger.warning(f"[validate_schema] {len(issues)} schema issues found.")
    else:
        logger.info("[validate_schema] Schema OK.")

    return issues


# ─── Entrypoint ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    spark = get_spark_session()

    # Inspect all processed datasets
    for dataset in ["olist_orders", "olist_customers", "olist_wide", "olist_transformed"]:
        path = os.path.join(PROCESSED_DIR, dataset)
        inspect_parquet(spark, path)

    # Inspect feature store
    inspect_parquet(spark, os.path.join(FEATURES_DIR, "user_features"))

    spark.stop()