"""
SparkSession factory — singleton pattern safe for driver reuse.
Supports local dev, YARN, and Kubernetes via SPARK_MASTER env override.
"""
import os
import logging
from pyspark.sql import SparkSession
from pyspark import SparkConf

logger = logging.getLogger(__name__)


def get_spark_session(
    app_name: str = None,
    master: str = None,
    executor_memory: str = None,
    driver_memory: str = None,
    shuffle_partitions: int = None,
    extra_configs: dict = None,
    enable_hive: bool = False,
) -> SparkSession:
    """
    Returns an existing or new SparkSession.

    Priority: explicit arg → env var → constants default.
    Call once at entrypoint; reuse everywhere via SparkSession.getActiveSession().
    """
    from src.config.constants import (
        SPARK_APP_NAME, SPARK_MASTER,
        SPARK_EXECUTOR_MEM, SPARK_DRIVER_MEM,
        SPARK_SHUFFLE_PARTS,
    )

    _app_name         = app_name        or os.getenv("SPARK_APP_NAME", SPARK_APP_NAME)
    _master           = master          or os.getenv("SPARK_MASTER",   SPARK_MASTER)
    _executor_mem     = executor_memory or os.getenv("SPARK_EXECUTOR_MEM", SPARK_EXECUTOR_MEM)
    _driver_mem       = driver_memory   or os.getenv("SPARK_DRIVER_MEM",   SPARK_DRIVER_MEM)
    _shuffle_parts    = shuffle_partitions or int(
        os.getenv("SPARK_SHUFFLE_PARTS", str(SPARK_SHUFFLE_PARTS))
    )

    # ── Return existing session if alive ───────────────────────────────────────
    existing = SparkSession.getActiveSession()
    if existing is not None:
        logger.info(f"Reusing active SparkSession: {existing.conf.get('spark.app.name')}")
        return existing

    # ── Build new session ──────────────────────────────────────────────────────
    conf = SparkConf()
    conf.setAll([
        ("spark.app.name",                          _app_name),
        ("spark.master",                            _master),
        ("spark.executor.memory",                   _executor_mem),
        ("spark.driver.memory",                     _driver_mem),
        ("spark.sql.shuffle.partitions",            str(_shuffle_parts)),
        # Parquet optimizations
        ("spark.sql.parquet.compression.codec",     "snappy"),
        ("spark.sql.parquet.mergeSchema",           "false"),   # perf: skip schema merge
        ("spark.sql.parquet.filterPushdown",        "true"),
        # Adaptive Query Execution (Spark 3.x)
        ("spark.sql.adaptive.enabled",              "true"),
        ("spark.sql.adaptive.coalescePartitions.enabled", "true"),
        # Broadcast join threshold
        ("spark.sql.autoBroadcastJoinThreshold",    "50MB"),
        # Kryo serialization
        ("spark.serializer",                        "org.apache.spark.serializer.KryoSerializer"),
        # Arrow-based pandas conversion — disable if PyArrow not installed
        ("spark.sql.execution.arrow.pyspark.enabled",         "false"),
        # S3 / GCS compatibility (no-op local)
        ("spark.hadoop.fs.s3a.impl",                "org.apache.hadoop.fs.s3a.S3AFileSystem"),
    ])

    if extra_configs:
        conf.setAll(list(extra_configs.items()))

    builder = SparkSession.builder.config(conf=conf)

    if enable_hive:
        builder = builder.enableHiveSupport()

    # Delta Lake: configure_spark_with_delta_pip injects the correct JAR
    # and sets the two required extensions automatically.
    # This is the ONLY correct way to enable Delta locally with pip install.
    try:
        from delta import configure_spark_with_delta_pip
        builder = configure_spark_with_delta_pip(builder)
        logger.info("[spark] Delta Lake enabled via configure_spark_with_delta_pip")
    except ImportError:
        logger.warning(
            "[spark] delta-spark not installed. Delta features unavailable. "
            "Run: pip install delta-spark"
        )

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    logger.info(
        f"SparkSession created | app={_app_name} | master={_master} "
        f"| executor_mem={_executor_mem} | driver_mem={_driver_mem}"
    )
    return spark


def stop_spark_session() -> None:
    """Graceful shutdown. Call at end of batch job entrypoints."""
    session = SparkSession.getActiveSession()
    if session:
        session.stop()
        logger.info("SparkSession stopped.")