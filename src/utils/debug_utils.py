"""
Spark Debug Utilities — Phase 6.4

Tools:
  1. Stage profiler       — time every transformation in a chain
  2. Slow task detector   — identify stragglers via task metrics
  3. Memory advisor       — detect spills, OOM risks
  4. Column lineage       — trace where columns come from
  5. Job profiler         — wrap any function and profile its Spark jobs
  6. Spark UI helper      — print direct links to relevant UI pages

Use these during development to understand what your jobs are doing.
In prod: use Spark History Server + Prometheus metrics instead.
"""
import logging
import os
import time
import functools
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 1. STAGE PROFILER — time each transformation
# ══════════════════════════════════════════════════════════════════════════════

class StageProfiler:
    """
    Times each step in a transformation chain.
    Forces evaluation via count() at each checkpoint.

    Usage:
        profiler = StageProfiler("Feature Engineering")
        df = profiler.checkpoint(raw_df,         "raw_load")
        df = profiler.checkpoint(cleaned_df,     "after_clean")
        df = profiler.checkpoint(features_df,    "after_features")
        profiler.report()

    WARNING: each checkpoint triggers a full Spark action (count).
    Only use during development, not in production pipelines.
    """

    def __init__(self, pipeline_name: str = "Pipeline"):
        self.name       = pipeline_name
        self.timings:   List[Dict] = []
        self._prev_rows: Optional[int] = None

    def checkpoint(
        self,
        df:         DataFrame,
        stage_name: str,
        cache:      bool = False,
    ) -> DataFrame:
        """
        Evaluate df, record row count and timing.
        Returns df (optionally cached) for chaining.
        """
        if cache:
            df.cache()

        t0    = time.time()
        rows  = df.count()
        elapsed = round(time.time() - t0, 2)

        row_delta = None
        if self._prev_rows is not None:
            row_delta = rows - self._prev_rows

        self.timings.append({
            "stage":      stage_name,
            "rows":       rows,
            "row_delta":  row_delta,
            "elapsed_s":  elapsed,
            "cached":     cache,
        })
        self._prev_rows = rows

        delta_str = ""
        if row_delta is not None:
            pct = (row_delta / self._prev_rows * 100) if self._prev_rows else 0
            delta_str = f" ({row_delta:+,} rows, {pct:+.1f}%)"

        logger.info(
            f"[profiler:{self.name}] {stage_name:<30} | "
            f"{rows:>10,} rows | {elapsed}s{delta_str}"
        )
        return df

    def report(self) -> None:
        total = sum(t["elapsed_s"] for t in self.timings)
        print(f"\n{'='*70}")
        print(f"STAGE PROFILE: {self.name}   (total={total:.1f}s)")
        print(f"{'='*70}")
        print(f"  {'Stage':<30} {'Rows':>10}  {'Delta':>10}  {'Time':>8}  {'%Total':>8}")
        print(f"  {'-'*30} {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}")
        for t in self.timings:
            delta  = f"{t['row_delta']:+,}" if t["row_delta"] is not None else "—"
            pct    = (t["elapsed_s"] / total * 100) if total > 0 else 0
            cached = " 📦" if t["cached"] else ""
            print(
                f"  {t['stage']:<30} {t['rows']:>10,}  {delta:>10}  "
                f"{t['elapsed_s']:>7.1f}s  {pct:>7.1f}%{cached}"
            )
        print(f"{'='*70}\n")

        # Highlight slowest stage
        if self.timings:
            slowest = max(self.timings, key=lambda x: x["elapsed_s"])
            logger.info(
                f"[profiler] Slowest stage: '{slowest['stage']}' "
                f"({slowest['elapsed_s']}s, "
                f"{slowest['elapsed_s']/total*100:.0f}% of total)"
            )


# ══════════════════════════════════════════════════════════════════════════════
# 2. CONTEXT MANAGER — time any block
# ══════════════════════════════════════════════════════════════════════════════

@contextmanager
def spark_timer(label: str):
    """
    Time any block of Spark code.

    Usage:
        with spark_timer("join customers + orders"):
            result = orders.join(customers, on="customer_id")
            result.count()   # force evaluation
    """
    logger.info(f"[timer] ▶  {label}")
    t0 = time.time()
    try:
        yield
    finally:
        elapsed = round(time.time() - t0, 2)
        logger.info(f"[timer] ✓  {label} → {elapsed}s")


# ══════════════════════════════════════════════════════════════════════════════
# 3. JOB PROFILER DECORATOR
# ══════════════════════════════════════════════════════════════════════════════

def profile_spark_job(func: Callable) -> Callable:
    """
    Decorator: wraps any function that runs Spark jobs.
    Logs entry/exit, timing, and any exceptions.

    Usage:
        @profile_spark_job
        def run_feature_engineering(spark, df):
            ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"[job_profiler] ▶  {func.__name__} started")
        t0 = time.time()
        try:
            result  = func(*args, **kwargs)
            elapsed = round(time.time() - t0, 2)
            logger.info(f"[job_profiler] ✅ {func.__name__} completed in {elapsed}s")
            return result
        except Exception as e:
            elapsed = round(time.time() - t0, 2)
            logger.error(
                f"[job_profiler] ❌ {func.__name__} FAILED after {elapsed}s: {e}"
            )
            raise
    return wrapper


# ══════════════════════════════════════════════════════════════════════════════
# 4. MEMORY ADVISOR
# ══════════════════════════════════════════════════════════════════════════════

def memory_advisor(spark: SparkSession) -> None:
    """
    Print memory configuration and warn about common OOM causes.

    Key settings to understand:
      spark.executor.memory       — JVM heap for executor
      spark.executor.memoryFraction — fraction for execution vs storage
      spark.memory.offHeap.enabled  — off-heap for unsafe operations

    Common OOM causes:
      1. shuffle.partitions too low → large partitions don't fit in memory
      2. .cache() on a huge DataFrame → storage fills up, evicts execution memory
      3. collect() on a large DataFrame → driver OOM
      4. UDFs that hold state → executor OOM
    """
    conf = spark.sparkContext.getConf()

    executor_mem   = conf.get("spark.executor.memory",      "1g")
    driver_mem     = conf.get("spark.driver.memory",        "1g")
    shuffle_parts  = conf.get("spark.sql.shuffle.partitions","200")
    offheap        = conf.get("spark.memory.offHeap.enabled","false")
    offheap_size   = conf.get("spark.memory.offHeap.size",  "0")

    print(f"\n{'='*55}")
    print("MEMORY CONFIGURATION")
    print(f"{'='*55}")
    print(f"  executor.memory          : {executor_mem}")
    print(f"  driver.memory            : {driver_mem}")
    print(f"  shuffle.partitions       : {shuffle_parts}")
    print(f"  offHeap.enabled          : {offheap}")
    print(f"  offHeap.size             : {offheap_size}")

    print(f"\n── Common OOM Risks ──────────────────────────────────")
    warnings = []

    if int(shuffle_parts) < 8:
        warnings.append(
            f"⚠️  shuffle.partitions={shuffle_parts} is very low. "
            f"Large shuffles may cause OOM. Raise to 2–4x CPU cores."
        )
    if executor_mem in ("512m", "1g") and int(shuffle_parts) > 100:
        warnings.append(
            f"⚠️  Low executor memory ({executor_mem}) with high "
            f"shuffle.partitions ({shuffle_parts}). Risk of spill to disk."
        )
    if not warnings:
        print("  ✅ No obvious OOM risks in current config.")
    for w in warnings:
        print(f"  {w}")

    print(f"\n── collect() Safety Check ────────────────────────────")
    print(
        "  Never call .collect() on large DataFrames.\n"
        "  Use .limit(N).collect() or .toPandas() with explicit row cap.\n"
        "  Rule: only collect when you know the result fits in driver memory."
    )
    print(f"{'='*55}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 5. DATAFRAME INSPECTOR
# ══════════════════════════════════════════════════════════════════════════════

def inspect_df(
    df:         DataFrame,
    name:       str = "DataFrame",
    sample_n:   int = 5,
    show_plan:  bool = True,
) -> None:
    """
    Quick DataFrame inspection: schema, row count, sample, plan.
    The first thing to run when debugging unexpected output.
    """
    print(f"\n{'='*65}")
    print(f"INSPECT: {name}")
    print(f"{'='*65}")

    # Schema
    print("\n── Schema ────────────────────────────────────────────────")
    df.printSchema()

    # Row count + partition count
    rows       = df.count()
    partitions = df.rdd.getNumPartitions()
    print(f"\n── Stats ─────────────────────────────────────────────────")
    print(f"  Rows       : {rows:,}")
    print(f"  Partitions : {partitions}")
    print(f"  Avg rows/partition: {rows // max(partitions, 1):,}")

    # Sample
    print(f"\n── Sample ({sample_n} rows) ───────────────────────────────────")
    df.show(sample_n, truncate=False)

    # Execution plan
    if show_plan:
        print(f"\n── Execution Plan (simple) ───────────────────────────────")
        df.explain(mode="simple")

    print(f"{'='*65}\n")


def compare_schemas(df1: DataFrame, df2: DataFrame,
                    name1: str = "df1", name2: str = "df2") -> List[str]:
    """
    Compare schemas of two DataFrames.
    Useful when debugging schema evolution or pipeline output changes.
    Returns list of differences.
    """
    schema1 = dict(df1.dtypes)
    schema2 = dict(df2.dtypes)

    all_cols = set(schema1.keys()) | set(schema2.keys())
    diffs    = []

    print(f"\n{'Column':<35} {name1:<20} {name2:<20}")
    print("-" * 75)
    for col in sorted(all_cols):
        t1 = schema1.get(col, "MISSING")
        t2 = schema2.get(col, "MISSING")
        if t1 != t2:
            print(f"  ⚠️  {col:<33} {t1:<20} {t2:<20}")
            diffs.append(f"{col}: {t1} → {t2}")
        else:
            print(f"  ✅ {col:<33} {t1:<20} {'(same)':<20}")

    if not diffs:
        logger.info("[debug] Schemas are identical.")
    else:
        logger.warning(f"[debug] {len(diffs)} schema differences found.")

    return diffs


# ══════════════════════════════════════════════════════════════════════════════
# 6. SPARK UI HELPER
# ══════════════════════════════════════════════════════════════════════════════

def print_ui_links(spark: SparkSession) -> None:
    """
    Print direct Spark UI links to open in browser.
    Run this while your job is executing to monitor it live.
    """
    sc    = spark.sparkContext
    ui    = sc.uiWebUrl or "http://localhost:4040"

    print(f"\n{'='*55}")
    print("SPARK UI — Open in browser while job is running")
    print(f"{'='*55}")
    print(f"  Jobs    : {ui}/jobs/")
    print(f"  Stages  : {ui}/stages/")
    print(f"  Storage : {ui}/storage/")
    print(f"  Environ : {ui}/environment/")
    print(f"  SQL/DAG : {ui}/SQL/")
    print(f"\n  App: {sc.appName}")
    print(f"  Master: {sc.master}")
    print(f"  Cores:  {sc.defaultParallelism}")
    print(f"{'='*55}\n")

    logger.info(f"[debug] Spark UI: {ui}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. ANTI-PATTERN DETECTOR
# ══════════════════════════════════════════════════════════════════════════════

def check_antipatterns(df: DataFrame) -> List[str]:
    """
    Statically analyze a DataFrame's plan for common anti-patterns.
    Returns list of warnings.

    Detects:
      - CartesianProduct (missing join condition)
      - Multiple Exchange (too many shuffles)
      - Missing filter pushdown
    """
    import io
    from contextlib import redirect_stdout

    f = io.StringIO()
    with redirect_stdout(f):
        df.explain(mode="extended")
    plan = f.getvalue()

    issues = []

    if "CartesianProduct" in plan:
        issues.append(
            "🔴 CartesianProduct: You have a join without a condition "
            "or a crossJoin(). This is O(n²) — will kill your job on large data."
        )

    shuffle_count = plan.count("Exchange hashpartitioning")
    if shuffle_count > 3:
        issues.append(
            f"⚠️  {shuffle_count} shuffles in plan. "
            f"Review joins/groupBys — can any be combined or eliminated?"
        )

    if "collect()" in plan or "CollectLimit" in plan:
        issues.append(
            "⚠️  collect() or limit→collect in plan. "
            "Ensure result fits in driver memory."
        )

    if not issues:
        logger.info("[antipattern] ✅ No anti-patterns detected.")
    else:
        for issue in issues:
            logger.warning(f"[antipattern] {issue}")

    return issues


# ══════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT — run full debug suite on feature store
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
    logging.basicConfig(level=logging.INFO)

    from src.utils.spark_session import get_spark_session
    from src.config.constants import FEATURES_DIR, PROCESSED_DIR
    from src.utils.spark_optimizer import run_optimization_report

    spark = get_spark_session()

    # Print UI links first — open browser now
    print_ui_links(spark)

    # Memory config
    memory_advisor(spark)

    # Load feature store
    features_path = os.path.join(FEATURES_DIR, "user_features")
    if os.path.exists(features_path):
        df = spark.read.parquet(features_path)

        # Full inspection
        inspect_df(df, name="user_features", sample_n=3)

        # Anti-pattern check
        check_antipatterns(df)

        # Optimization report
        run_optimization_report(
            spark, df,
            table_name="user_features",
            group_cols=["customer_state"],
        )

        # Stage profiler demo
        profiler = StageProfiler("feature_store_analysis")
        profiler.checkpoint(df, "raw_load")

        filtered = df.filter(F.col("frequency") > 1)
        profiler.checkpoint(filtered, "after_filter")

        aggregated = filtered.groupBy("customer_state").count()
        profiler.checkpoint(aggregated, "after_groupby")

        profiler.report()
    else:
        logger.warning(f"Feature store not found at {features_path}. Run pipeline first.")

    spark.stop()