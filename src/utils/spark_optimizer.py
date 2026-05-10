"""
Spark Optimizer Utilities — Phase 6.4

Covers:
  1. Skew detection       — find which partition keys are overloaded
  2. Skew fix (salting)   — redistribute skewed joins/groupBys
  3. Broadcast advisor    — detect when small tables aren't being broadcast
  4. Partition advisor    — recommend optimal partition count
  5. Shuffle optimizer    — detect and fix unnecessary shuffles

Production rule of thumb:
  - Target partition size: 128MB–256MB
  - Target task count: 2–4x number of CPU cores
  - Shuffle partitions: match task count, not default 200
"""
import logging
import math
import os
from typing import Dict, List, Optional, Tuple

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

logger = logging.getLogger(__name__)

# ─── Thresholds ───────────────────────────────────────────────────────────────
SKEW_RATIO_THRESHOLD    = 3.0    # max_partition / avg_partition > 3x = skewed
SKEW_ABS_THRESHOLD_MB   = 200    # partition > 200MB and skewed = critical
BROADCAST_SIZE_LIMIT_MB = 100    # tables smaller than this should be broadcast
TARGET_PARTITION_MB     = 128    # ideal partition size


# ══════════════════════════════════════════════════════════════════════════════
# 1. SKEW DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_skew(
    df: DataFrame,
    group_col: str,
    value_col: str = None,
    top_n: int = 10,
) -> Dict:
    """
    Detect data skew on a groupBy key.

    Returns a dict with:
      - skew_ratio: max_count / avg_count
      - is_skewed:  bool
      - top_keys:   top N heaviest keys and their row counts
      - recommendation: what to do

    Example:
        result = detect_skew(orders_df, group_col="customer_state")
        if result["is_skewed"]:
            df = apply_salting(df, "customer_state")
    """
    logger.info(f"[optimizer] Detecting skew on column: '{group_col}'")

    # Row count per key
    key_counts = (
        df.groupBy(group_col)
        .count()
        .orderBy(F.col("count").desc())
    )

    stats = key_counts.select(
        F.max("count").alias("max_count"),
        F.avg("count").alias("avg_count"),
        F.min("count").alias("min_count"),
        F.stddev("count").alias("std_count"),
        F.count("*").alias("num_distinct_keys"),
    ).collect()[0]

    max_count  = stats["max_count"]  or 1
    avg_count  = stats["avg_count"]  or 1
    skew_ratio = max_count / avg_count

    top_keys = [
        {"key": row[group_col], "count": row["count"]}
        for row in key_counts.limit(top_n).collect()
    ]

    is_skewed = skew_ratio >= SKEW_RATIO_THRESHOLD

    # Build recommendation
    if not is_skewed:
        recommendation = f"No skew detected (ratio={skew_ratio:.1f}x). No action needed."
    elif skew_ratio < 5:
        recommendation = (
            f"Moderate skew (ratio={skew_ratio:.1f}x). "
            f"Consider salting if this column is used in joins/groupBy."
        )
    else:
        recommendation = (
            f"CRITICAL skew (ratio={skew_ratio:.1f}x). "
            f"Apply salting immediately. Top key '{top_keys[0]['key']}' "
            f"has {top_keys[0]['count']:,} rows vs avg {avg_count:.0f}."
        )

    result = {
        "column":           group_col,
        "skew_ratio":       round(skew_ratio, 2),
        "is_skewed":        is_skewed,
        "max_count":        max_count,
        "avg_count":        round(avg_count, 1),
        "num_distinct_keys":stats["num_distinct_keys"],
        "top_keys":         top_keys,
        "recommendation":   recommendation,
    }

    # Log summary
    icon = "🔴" if is_skewed else "🟢"
    logger.info(
        f"[optimizer] {icon} Skew on '{group_col}': "
        f"ratio={skew_ratio:.1f}x | "
        f"max={max_count:,} | avg={avg_count:.0f} | "
        f"distinct_keys={stats['num_distinct_keys']:,}"
    )
    if is_skewed:
        logger.warning(f"[optimizer] {recommendation}")

    return result


def detect_skew_multiple_cols(
    df: DataFrame,
    cols: List[str],
) -> List[Dict]:
    """Run skew detection on multiple columns at once."""
    results = []
    for col in cols:
        try:
            result = detect_skew(df, col)
            results.append(result)
        except Exception as e:
            logger.error(f"[optimizer] Skew detection failed on '{col}': {e}")
    return sorted(results, key=lambda x: x["skew_ratio"], reverse=True)


# ══════════════════════════════════════════════════════════════════════════════
# 2. SKEW FIX — SALTING
# ══════════════════════════════════════════════════════════════════════════════

def apply_salting(
    df: DataFrame,
    join_col: str,
    n_buckets: int = 10,
    salt_col: str = "_salt",
) -> DataFrame:
    """
    Salting: add a random prefix to the join/groupBy key to spread
    skewed partitions across multiple executors.

    Use case: one key (e.g. customer_state='SP') has 42% of all rows.
    After salting: SP_0, SP_1, ... SP_9 each have ~4% → balanced.

    NOTE: For joins, you must also explode the salt on the smaller side.
    See: salt_and_join() for the full pattern.
    """
    logger.info(
        f"[optimizer] Applying salting on '{join_col}' "
        f"with {n_buckets} buckets"
    )
    return df.withColumn(
        salt_col, (F.rand() * n_buckets).cast("int")
    ).withColumn(
        f"{join_col}_salted",
        F.concat(F.col(join_col).cast("string"), F.lit("_"), F.col(salt_col))
    )


def salt_and_join(
    large_df:   DataFrame,
    small_df:   DataFrame,
    join_col:   str,
    n_buckets:  int = 10,
    join_type:  str = "inner",
) -> DataFrame:
    """
    Full salted join pattern for skewed data.

    Pattern:
      1. Add random salt [0..N) to the large (skewed) table
      2. Explode salt [0..N) on the small table (replicate N times)
      3. Join on (original_key + salt)
      4. Drop salt columns

    Cost: small table is replicated N times in memory.
    Only use when large table has severe skew on join key.
    """
    logger.info(
        f"[optimizer] Salted join on '{join_col}' | "
        f"buckets={n_buckets} | type={join_type}"
    )

    # Salt the large table
    large_salted = large_df.withColumn(
        "_salt", (F.rand() * n_buckets).cast("int")
    ).withColumn(
        "_join_key",
        F.concat(F.col(join_col).cast("string"), F.lit("_"), F.col("_salt"))
    )

    # Explode salt on the small table (replicate for each bucket)
    small_exploded = small_df.withColumn(
        "_salt", F.explode(F.array([F.lit(i) for i in range(n_buckets)]))
    ).withColumn(
        "_join_key",
        F.concat(F.col(join_col).cast("string"), F.lit("_"), F.col("_salt"))
    )

    # Join on salted key
    result = large_salted.join(
        small_exploded, on="_join_key", how=join_type
    ).drop("_salt", "_join_key")

    logger.info("[optimizer] Salted join complete.")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 3. BROADCAST ADVISOR
# ══════════════════════════════════════════════════════════════════════════════

def _parse_spark_size_to_mb(value: str) -> float:
    """
    Parse Spark size strings to MB.
    Handles: '50MB', '10485760' (bytes), '1g', '512k', '1tb'
    """
    value = str(value).strip().upper()
    try:
        # Pure number → bytes
        return int(value) / (1024 * 1024)
    except ValueError:
        pass

    units = {
        "TB": 1024 * 1024,
        "GB": 1024,
        "MB": 1,
        "KB": 1 / 1024,
        "B":  1 / (1024 * 1024),
        "G":  1024,
        "M":  1,
        "K":  1 / 1024,
    }
    for suffix, multiplier in units.items():
        if value.endswith(suffix):
            try:
                return float(value[: -len(suffix)]) * multiplier
            except ValueError:
                break

    logger.warning(f"[optimizer] Could not parse size '{value}', defaulting to 10MB")
    return 10.0


def broadcast_advisor(
    spark: SparkSession,
    df: DataFrame,
    table_name: str,
    estimated_mb: float = None,
) -> Tuple[bool, str]:
    """
    Advise whether a DataFrame should be broadcast in joins.

    Returns (should_broadcast: bool, reason: str)

    In prod: use df.rdd.toDebugString() or df.explain() to confirm
    Spark's physical plan chose BroadcastHashJoin.
    """
    raw       = spark.conf.get("spark.sql.autoBroadcastJoinThreshold", "10485760")
    threshold_mb = _parse_spark_size_to_mb(raw)

    if estimated_mb is None:
        # Estimate from row count × avg row size
        # Rough heuristic: 100 bytes/row for typical feature tables
        row_count    = df.count()
        estimated_mb = (row_count * 100) / (1024 * 1024)

    should_broadcast = estimated_mb < BROADCAST_SIZE_LIMIT_MB

    if should_broadcast and estimated_mb > threshold_mb:
        reason = (
            f"Table '{table_name}' is {estimated_mb:.1f}MB — small enough to broadcast "
            f"but ABOVE auto-broadcast threshold ({threshold_mb:.0f}MB). "
            f"Use F.broadcast(df) explicitly or raise "
            f"spark.sql.autoBroadcastJoinThreshold."
        )
        logger.warning(f"[optimizer] ⚠️  {reason}")
    elif should_broadcast:
        reason = (
            f"Table '{table_name}' is {estimated_mb:.1f}MB — "
            f"will be auto-broadcast. ✅"
        )
        logger.info(f"[optimizer] {reason}")
    else:
        reason = (
            f"Table '{table_name}' is {estimated_mb:.1f}MB — "
            f"too large to broadcast. Use SortMergeJoin."
        )
        logger.info(f"[optimizer] {reason}")

    return should_broadcast, reason


def get_broadcast_hint(df: DataFrame) -> DataFrame:
    """
    Wrap a DataFrame with a broadcast hint.
    Spark will use BroadcastHashJoin regardless of size threshold.

    Use for known-small lookup tables:
        category_df = get_broadcast_hint(category_df)
        result = large_df.join(category_df, on="category_id")
    """
    return F.broadcast(df)


# ══════════════════════════════════════════════════════════════════════════════
# 4. PARTITION ADVISOR
# ══════════════════════════════════════════════════════════════════════════════

def partition_advisor(
    spark:         SparkSession,
    df:            DataFrame,
    operation:     str = "groupby",   # groupby | join | write
    target_mb:     int = TARGET_PARTITION_MB,
) -> Dict:
    """
    Recommend optimal partition count for an operation.

    Formula:
      total_data_mb / target_partition_mb = num_partitions
      Round up to nearest power of 2 for cache efficiency.

    Returns dict with recommendation and current settings.
    """
    current_partitions = df.rdd.getNumPartitions()
    current_shuffle = int(
        spark.conf.get("spark.sql.shuffle.partitions", "200").split(".")[0]
        .replace("MB","").replace("GB","").replace("KB","").strip() or "200"
    )

    # Estimate data size (rough: serialized row size × count)
    sample = df.limit(1000).toPandas()
    if len(sample) > 0:
        avg_row_bytes  = sample.memory_usage(deep=True).sum() / len(sample)
        estimated_rows = df.count()
        estimated_mb   = (avg_row_bytes * estimated_rows) / (1024 * 1024)
    else:
        estimated_mb = 0

    recommended = max(1, math.ceil(estimated_mb / target_mb))
    # Round to nearest power of 2
    recommended = 2 ** math.ceil(math.log2(recommended)) if recommended > 1 else 1

    # Cap at reasonable bounds
    cpu_cores = os.cpu_count() or 4
    recommended = max(recommended, cpu_cores * 2)   # at least 2x cores
    recommended = min(recommended, 2000)             # cap at 2000 for local

    result = {
        "estimated_data_mb":      round(estimated_mb, 1),
        "current_partitions":     current_partitions,
        "current_shuffle_parts":  current_shuffle,
        "recommended_partitions": recommended,
        "target_partition_mb":    target_mb,
        "operation":              operation,
    }

    if abs(current_shuffle - recommended) / max(current_shuffle, 1) > 0.5:
        logger.warning(
            f"[optimizer] ⚠️  shuffle.partitions={current_shuffle} but "
            f"recommended={recommended} for {estimated_mb:.0f}MB of data. "
            f"Set: spark.conf.set('spark.sql.shuffle.partitions', '{recommended}')"
        )
    else:
        logger.info(
            f"[optimizer] ✅ shuffle.partitions={current_shuffle} is "
            f"reasonable for {estimated_mb:.0f}MB."
        )

    return result


def set_optimal_shuffle_partitions(
    spark: SparkSession,
    df: DataFrame,
) -> int:
    """
    Automatically set spark.sql.shuffle.partitions based on data size.
    Returns the value that was set.
    """
    advice = partition_advisor(spark, df)
    recommended = advice["recommended_partitions"]
    spark.conf.set("spark.sql.shuffle.partitions", str(recommended))
    logger.info(
        f"[optimizer] Set shuffle.partitions={recommended} "
        f"(data={advice['estimated_data_mb']}MB)"
    )
    return recommended


# ══════════════════════════════════════════════════════════════════════════════
# 5. SHUFFLE OPTIMIZER
# ══════════════════════════════════════════════════════════════════════════════

def explain_plan(df: DataFrame, mode: str = "simple") -> str:
    """
    Return the execution plan as a string.
    mode: simple | extended | codegen | cost | formatted

    Look for in the output:
      ✅ BroadcastHashJoin   → small table broadcast (fast)
      ✅ Filter pushed down  → scan reads less data
      ⚠️  SortMergeJoin      → both sides shuffled (expensive)
      ❌ CartesianProduct    → cross join (disaster)
      ❌ Exchange            → shuffle (check if avoidable)
    """
    import io
    from contextlib import redirect_stdout

    f = io.StringIO()
    with redirect_stdout(f):
        df.explain(mode=mode)
    plan = f.getvalue()

    # Annotate key patterns
    warnings = []
    if "CartesianProduct" in plan:
        warnings.append("🔴 CRITICAL: CartesianProduct detected — likely a missing join condition")
    if "SortMergeJoin" in plan:
        warnings.append("⚠️  SortMergeJoin detected — check if one side can be broadcast")
    if "BroadcastHashJoin" in plan:
        warnings.append("✅ BroadcastHashJoin in use — optimal for small table joins")
    if "Exchange hashpartitioning" in plan:
        count = plan.count("Exchange hashpartitioning")
        warnings.append(f"⚠️  {count} shuffle(s) detected — review if all are necessary")

    if warnings:
        logger.info("[optimizer] Plan analysis:\n  " + "\n  ".join(warnings))

    return plan


def check_filter_pushdown(df: DataFrame, filter_col: str) -> bool:
    """
    Verify that a filter on a partition column is being pushed down
    into the parquet scan (not applied after reading all data).
    Returns True if pushdown is happening.
    """
    plan = explain_plan(df, mode="extended")
    pushed = "PushedFilters" in plan and filter_col in plan
    if pushed:
        logger.info(f"[optimizer] ✅ Filter on '{filter_col}' is pushed down into scan.")
    else:
        logger.warning(
            f"[optimizer] ⚠️  Filter on '{filter_col}' is NOT pushed down. "
            f"Ensure the column is a partition column and filter uses it directly."
        )
    return pushed


# ══════════════════════════════════════════════════════════════════════════════
# 6. FULL OPTIMIZATION REPORT
# ══════════════════════════════════════════════════════════════════════════════

def run_optimization_report(
    spark:       SparkSession,
    df:          DataFrame,
    table_name:  str,
    join_cols:   Optional[List[str]] = None,
    group_cols:  Optional[List[str]] = None,
) -> None:
    """
    Run all optimizations checks on a DataFrame and print a full report.
    Call this on your largest DataFrames (wide table, feature store).
    """
    print(f"\n{'='*65}")
    print(f"OPTIMIZATION REPORT: {table_name}")
    print(f"{'='*65}")

    # Partition advice
    print("\n── Partition Advice ──────────────────────────────────────")
    advice = partition_advisor(spark, df)
    print(f"  Data size (est.)   : {advice['estimated_data_mb']} MB")
    print(f"  Current partitions : {advice['current_partitions']}")
    print(f"  Shuffle partitions : {advice['current_shuffle_parts']}")
    print(f"  Recommended        : {advice['recommended_partitions']}")

    # Skew detection
    check_cols = (join_cols or []) + (group_cols or [])
    if check_cols:
        print("\n── Skew Detection ────────────────────────────────────────")
        skew_results = detect_skew_multiple_cols(df, check_cols)
        for r in skew_results:
            icon = "🔴" if r["is_skewed"] else "🟢"
            print(
                f"  {icon} {r['column']:<30} "
                f"skew_ratio={r['skew_ratio']:.1f}x | "
                f"distinct_keys={r['num_distinct_keys']:,}"
            )
            if r["is_skewed"]:
                print(f"     → {r['recommendation']}")

    # Broadcast advice
    print("\n── Broadcast Advice ──────────────────────────────────────")
    should_bc, reason = broadcast_advisor(spark, df, table_name)
    print(f"  {reason}")

    print(f"\n{'='*65}\n")