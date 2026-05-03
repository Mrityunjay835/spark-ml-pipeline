"""
Streaming data generator: writes synthetic customer events to STREAMING_INPUT_DIR
as parquet files. Simulates a Kafka → landing zone pattern for structured streaming.

Run in a separate process/thread while stream_predict.py is running.
"""
import logging
import os
import time
import random
import uuid
from datetime import datetime, timedelta
from typing import List

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, DoubleType, IntegerType

from src.config.constants import STREAMING_INPUT_DIR, FEATURES_DIR
from src.utils.spark_session import get_spark_session

logger = logging.getLogger(__name__)


# ─── Synthetic data config ─────────────────────────────────────────────────────
STATES = [
    "SP", "RJ", "MG", "RS", "PR", "SC", "BA", "GO", "ES", "CE",
    "PE", "MT", "MS", "PA", "RN", "PB", "AM", "MA", "AL", "SE",
]
PAYMENT_TYPES = ["credit_card", "boleto", "voucher", "debit_card"]
EVENT_TYPES   = ["view", "addtocart", "purchase", "review"]

STREAMING_SCHEMA = StructType([
    StructField("customer_unique_id",    StringType(),   False),
    StructField("event_timestamp",       TimestampType(),True),
    StructField("event_type",            StringType(),   True),
    StructField("item_id",               IntegerType(),  True),
    StructField("price",                 DoubleType(),   True),
    StructField("customer_state",        StringType(),   True),
    StructField("primary_payment_type",  StringType(),   True),
    # Features needed for prediction (pre-computed from feature store in real system)
    StructField("recency_days",          DoubleType(),   True),
    StructField("frequency",             DoubleType(),   True),
    StructField("monetary",              DoubleType(),   True),
    StructField("avg_order_value",       DoubleType(),   True),
    StructField("avg_review_score",      DoubleType(),   True),
    StructField("std_review_score",      DoubleType(),   True),
    StructField("min_review_score",      DoubleType(),   True),
    StructField("avg_installments",      DoubleType(),   True),
    StructField("total_freight",         DoubleType(),   True),
    StructField("avg_freight_ratio",     DoubleType(),   True),
    StructField("weekend_purchase_ratio",DoubleType(),   True),
    StructField("avg_purchase_hour",     DoubleType(),   True),
    StructField("avg_categories_per_order", DoubleType(), True),
    StructField("avg_delivery_delay",    DoubleType(),   True),
    StructField("max_delivery_delay",    DoubleType(),   True),
    StructField("avg_items_per_order",   DoubleType(),   True),
    StructField("total_items_purchased", DoubleType(),   True),
    StructField("spend_trend",           DoubleType(),   True),
    StructField("high_value_customer",   DoubleType(),   True),
    StructField("is_repeat_buyer",       DoubleType(),   True),
])


def _generate_customer_event(customer_id: str = None) -> dict:
    """Generate one synthetic customer event with feature values."""
    is_churner = random.random() < 0.3     # 30% churners

    # Churners have high recency, low frequency, low monetary
    recency = random.gauss(120, 30) if is_churner else random.gauss(20, 15)
    frequency = random.gauss(1.2, 0.5) if is_churner else random.gauss(3.5, 2.0)
    monetary = random.gauss(80, 40) if is_churner else random.gauss(350, 200)

    return {
        "customer_unique_id":     customer_id or str(uuid.uuid4()),
        "event_timestamp":        datetime.now() - timedelta(seconds=random.randint(0, 300)),
        "event_type":             random.choice(EVENT_TYPES),
        "item_id":                random.randint(1, 100000),
        "price":                  round(random.uniform(10, 500), 2),
        "customer_state":         random.choice(STATES),
        "primary_payment_type":   random.choice(PAYMENT_TYPES),
        # Features
        "recency_days":           max(0.0, recency),
        "frequency":              max(1.0, frequency),
        "monetary":               max(0.0, monetary),
        "avg_order_value":        max(0.0, monetary / max(1, frequency)),
        "avg_review_score":       round(random.gauss(3.5 if is_churner else 4.2, 0.8), 2),
        "std_review_score":       round(abs(random.gauss(0.8, 0.3)), 2),
        "min_review_score":       float(random.randint(1, 3 if is_churner else 4)),
        "avg_installments":       round(random.gauss(2.0, 1.5), 2),
        "total_freight":          round(random.gauss(20, 10), 2),
        "avg_freight_ratio":      round(random.uniform(0.05, 0.30), 4),
        "weekend_purchase_ratio": round(random.uniform(0.0, 1.0), 4),
        "avg_purchase_hour":      round(random.gauss(14, 4), 2),
        "avg_categories_per_order": round(random.uniform(1.0, 4.0), 2),
        "avg_delivery_delay":     round(random.gauss(2 if is_churner else -1, 3), 2),
        "max_delivery_delay":     round(random.gauss(5 if is_churner else 0, 5), 2),
        "avg_items_per_order":    round(random.gauss(1.5, 0.8), 2),
        "total_items_purchased":  max(1.0, round(frequency * random.gauss(1.5, 0.5), 0)),
        "spend_trend":            round(random.gauss(-0.1 if is_churner else 0.1, 0.3), 4),
        "high_value_customer":    0.0 if is_churner else float(random.random() > 0.5),
        "is_repeat_buyer":        0.0 if frequency <= 1 else 1.0,
    }


def generate_batch(spark: SparkSession, n_events: int = 100, batch_id: int = 0) -> None:
    import glob, shutil
    os.makedirs(STREAMING_INPUT_DIR, exist_ok=True)

    records = [_generate_customer_event() for _ in range(n_events)]
    df = spark.createDataFrame(records, schema=STREAMING_SCHEMA)

    tmp_dir = os.path.join(STREAMING_INPUT_DIR, f"_tmp_batch_{batch_id:06d}")
    df.coalesce(1).write.mode("overwrite").parquet(tmp_dir)

    part_files = glob.glob(os.path.join(tmp_dir, "part-*.parquet"))
    flat_path = ""
    if part_files:
        flat_name = f"batch_{batch_id:06d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        flat_path = os.path.join(STREAMING_INPUT_DIR, flat_name)
        os.rename(part_files[0], flat_path)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    logger.info(f"[generator] Batch {batch_id}: wrote {n_events} events → {flat_path}")


def run_continuous_generation(
    spark: SparkSession,
    events_per_batch: int = 50,
    interval_seconds: int = 15,
    max_batches: int = None,
) -> None:
    """
    Continuously generate streaming data batches.
    Runs indefinitely (or up to max_batches) to feed stream_predict.py.
    """
    logger.info(
        f"[generator] Starting continuous generation: "
        f"{events_per_batch} events/{interval_seconds}s"
    )

    batch_id = 0
    while max_batches is None or batch_id < max_batches:
        try:
            generate_batch(spark, n_events=events_per_batch, batch_id=batch_id)
            batch_id += 1
            time.sleep(interval_seconds)
        except KeyboardInterrupt:
            logger.info("[generator] Stopped by user.")
            break
        except Exception as e:
            logger.error(f"[generator] Error in batch {batch_id}: {e}", exc_info=True)
            time.sleep(5)

    logger.info(f"[generator] Generated {batch_id} total batches.")


# ─── Entrypoint ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=int, default=50)
    parser.add_argument("--interval", type=int, default=15, help="Seconds between batches")
    parser.add_argument("--batches", type=int, default=None, help="Max batches (None=infinite)")
    args = parser.parse_args()

    spark = get_spark_session(app_name="DataGenerator")
    run_continuous_generation(
        spark,
        events_per_batch=args.events,
        interval_seconds=args.interval,
        max_batches=args.batches,
    )
    spark.stop()