"""
Spark Structured Streaming: reads from STREAMING_INPUT_DIR, applies loaded
PipelineModel for real-time churn prediction, writes predictions to output sink.

Design:
  - Source:  File-based (simulates Kafka via landing zone)
  - Model:   Loaded once in driver, broadcast via closure (Spark ML handles this)
  - Sink:    Parquet (replace with Kafka/Delta/Cassandra in prod)
  - Mode:    Append (stateless per micro-batch)
"""
import logging
import os

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

from src.config.constants import (
    STREAMING_INPUT_DIR, STREAMING_CHECKPOINT,
    STREAM_TRIGGER_INTERVAL, STREAM_MAX_FILES,
    MODELS_DIR, LABEL_COL, PREDICTION_COL, PROB_COL,
)
from src.ml.pipeline import load_pipeline, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from src.streaming.data_generator import STREAMING_SCHEMA
from src.utils.spark_session import get_spark_session

logger = logging.getLogger(__name__)

# Output location for predictions
PREDICTIONS_OUTPUT = os.path.join(MODELS_DIR, "../data/streaming/predictions")
PREDICTIONS_CHECKPOINT = os.path.join(STREAMING_CHECKPOINT, "predictions")


def load_model(classifier_type: str = "gbt"):
    """Load trained PipelineModel from disk."""
    model_path = os.path.join(MODELS_DIR, f"model_{classifier_type}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"[stream] Model not found at {model_path}. "
            f"Run train.py first."
        )
    return load_pipeline(path=model_path)


def build_streaming_source(spark: SparkSession) -> DataFrame:
    return (
        spark.readStream
        .schema(STREAMING_SCHEMA)
        .option("maxFilesPerTrigger", STREAM_MAX_FILES)
        .option("latestFirst", "false")
        .option("recursiveFileLookup", "true")
        .option("pathGlobFilter", "*.parquet")
        .parquet(STREAMING_INPUT_DIR)
    )

def add_prediction_metadata(df: DataFrame) -> DataFrame:
    """Post-process predictions: extract churn_probability scalar, risk tier."""
    # probability column is a DenseVector — extract P(churn=1) at index 1
    extract_prob = F.udf(lambda v: float(v[1]) if v is not None else None, DoubleType())

    df = df.withColumn("churn_probability", extract_prob(F.col(PROB_COL)))

    df = df.withColumn(
        "risk_tier",
        F.when(F.col("churn_probability") >= 0.75, "HIGH")
         .when(F.col("churn_probability") >= 0.50, "MEDIUM")
         .when(F.col("churn_probability") >= 0.25, "LOW")
         .otherwise("SAFE")
    ).withColumn(
        "processed_at", F.current_timestamp()
    )
    return df


def run_streaming_predictions(
    spark: SparkSession,
    model,
    classifier_type: str = "gbt",
    output_mode: str = "parquet",  # parquet | console | memory
    trigger_interval: str = STREAM_TRIGGER_INTERVAL,
) -> None:
    """
    Main streaming loop:
    1. Read parquet batches from landing zone
    2. Apply pipeline model (transform = feature engineering + predict)
    3. Enrich predictions with metadata
    4. Write to configured sink
    """
    os.makedirs(PREDICTIONS_OUTPUT, exist_ok=True)
    os.makedirs(PREDICTIONS_CHECKPOINT, exist_ok=True)

    source_df = build_streaming_source(spark)

    # Validate required columns are present
    required_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    missing = [c for c in required_cols if c not in source_df.columns]
    if missing:
        raise ValueError(f"[stream] Source missing required feature columns: {missing}")

    # Apply model (Spark ML transform is streaming-compatible)
    predictions_df = model.transform(source_df)
    predictions_df = add_prediction_metadata(predictions_df)

    # Select output columns
    output_df = predictions_df.select(
        "customer_unique_id",
        "event_timestamp",
        "event_type",
        "customer_state",
        PREDICTION_COL,
        "churn_probability",
        "risk_tier",
        "processed_at",
    )

    # ── Configure sink ─────────────────────────────────────────────────────────
    if output_mode == "console":
        query = (
            output_df.writeStream
            .outputMode("append")
            .format("console")
            .option("truncate", False)
            .option("numRows", 20)
            .trigger(processingTime=trigger_interval)
            .start()
        )

    elif output_mode == "memory":
        query = (
            output_df.writeStream
            .outputMode("append")
            .format("memory")
            .queryName("churn_predictions")
            .trigger(processingTime=trigger_interval)
            .start()
        )

    else:  # parquet (default)
        query = (
            output_df.writeStream
            .outputMode("append")
            .format("parquet")
            .option("path", PREDICTIONS_OUTPUT)
            .option("checkpointLocation", PREDICTIONS_CHECKPOINT)
            .partitionBy("risk_tier")
            .trigger(processingTime=trigger_interval)
            .start()
        )

    logger.info(
        f"[stream] Streaming query started | sink={output_mode} | "
        f"trigger={trigger_interval} | model={classifier_type}"
    )

    try:
        query.awaitTermination()
    except KeyboardInterrupt:
        logger.info("[stream] Stopping streaming query...")
        query.stop()
    finally:
        _print_query_progress(query)


def _print_query_progress(query) -> None:
    """Log recent query progress metrics."""
    try:
        progress = query.recentProgress
        if progress:
            last = progress[-1]
            logger.info(
                f"[stream] Last batch: id={last.get('batchId')} | "
                f"input_rows={last.get('numInputRows')} | "
                f"duration_ms={last.get('batchDuration')}"
            )
    except Exception:
        pass


def read_predictions_from_memory(spark: SparkSession, limit: int = 100) -> DataFrame:
    """
    Read from memory sink for debugging / testing.
    Only works when output_mode='memory'.
    """
    return spark.sql(f"SELECT * FROM churn_predictions ORDER BY processed_at DESC LIMIT {limit}")


# ─── Entrypoint ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", default="gbt", choices=["gbt", "rf", "lr"])
    parser.add_argument("--sink", default="parquet",
                        choices=["parquet", "console", "memory"])
    parser.add_argument("--trigger", default=STREAM_TRIGGER_INTERVAL,
                        help="e.g. '30 seconds'")
    args = parser.parse_args()

    spark = get_spark_session(app_name="StreamChurnPredict")

    logger.info(f"[stream] Loading model: {args.classifier}")
    model = load_model(args.classifier)

    run_streaming_predictions(
        spark,
        model=model,
        classifier_type=args.classifier,
        output_mode=args.sink,
        trigger_interval=args.trigger,
    )
    spark.stop()