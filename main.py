"""
Single entrypoint for the Spark ML pipeline.

Execution order:
  ingest → join → transform → features → eda → train → [stream]

Usage:
  # Full pipeline
  python main.py --run all

  # Partial (resume from features onward)
  python main.py --run features,train

  # Individual stage
  python main.py --run train --classifier rf --cv

  # With streaming
  python main.py --run all --stream

  # Skip EDA (faster CI runs)
  python main.py --run all --skip eda
"""

import argparse
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from typing import List, Optional

from src.utils.spark_session import get_spark_session, stop_spark_session
from src.config.constants import LOGS_DIR

# ─── Logging setup ────────────────────────────────────────────────────────────
os.makedirs(LOGS_DIR, exist_ok=True)
log_file = os.path.join(LOGS_DIR, f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("pipeline")

# ─── Stage registry ───────────────────────────────────────────────────────────
# Defines valid stage names and their execution order
STAGE_ORDER = ["ingest", "join", "transform", "eda", "features", "train", "monitor", "stream"]


# ─── Stage runners ────────────────────────────────────────────────────────────

def run_ingest(spark, **kwargs):
    from src.batch.ingest import ingest_olist, ingest_retailrocket, write_processed
    logger.info("[ingest] Loading Olist CSVs...")
    olist = ingest_olist(spark)
    for name, df in olist.items():
        write_processed(df, f"olist_{name}")

    logger.info("[ingest] Loading RetailRocket CSVs...")
    rr = ingest_retailrocket(spark)
    for name, df in rr.items():
        write_processed(df, f"retailrocket_{name}")


def run_join(spark, **kwargs):
    from src.batch.join_data import build_olist_wide, build_retailrocket_user_signals, save_wide_table
    from src.config.constants import PROCESSED_DIR
    import os

    wide = build_olist_wide(spark)
    save_wide_table(wide)

    rr_signals = build_retailrocket_user_signals(spark)
    rr_out = os.path.join(PROCESSED_DIR, "retailrocket_user_signals")
    rr_signals.write.mode("overwrite").parquet(rr_out)


def run_transform(spark, **kwargs):
    from src.batch.transform import load_wide, run_transform_pipeline
    from src.config.constants import PROCESSED_DIR
    import os

    raw_wide = load_wide(spark)
    transformed = run_transform_pipeline(raw_wide)
    out_path = os.path.join(PROCESSED_DIR, "olist_transformed")
    transformed.write.mode("overwrite").option("compression", "snappy").parquet(out_path)


def run_features(spark, **kwargs):
    from src.features.feature_engineering import build_user_features, save_user_features
    user_features = build_user_features(spark)
    save_user_features(user_features)


def run_eda(spark, **kwargs):
    from src.features.eda import (
        class_balance, numeric_summary,
        churn_feature_stats, categorical_distribution,
    )
    from src.batch.transform import compute_delivery_delay
    from src.config.constants import PROCESSED_DIR, LABEL_COL
    from pyspark.sql import functions as F
    import os

    # Read transformed wide table (pre-feature-engineering)
    df = spark.read.parquet(os.path.join(PROCESSED_DIR, "olist_transformed"))

    logger.info("[eda] Numeric summary on transformed data:")
    numeric_summary(df)

    logger.info("[eda] Review score distribution:")
    categorical_distribution(df, "review_score")

    logger.info("[eda] Customer state distribution:")
    categorical_distribution(df, "customer_state")

    logger.info("[eda] Order status distribution:")
    categorical_distribution(df, "order_status")

    logger.info("[eda] Payment type distribution:")
    categorical_distribution(df, "primary_payment_type")


def run_monitor(spark, **kwargs):
    from src.monitoring.monitor import run_health_check
    from src.monitoring.retrain_trigger import run_drift_and_retrain_cycle, RetriggerConfig
    run_health_check(spark)
    run_drift_and_retrain_cycle(spark)


def run_train(spark, classifier: str = "gbt", use_cv: bool = False, **kwargs):
    from src.ml.train import run_training
    model, metrics = run_training(spark, classifier_type=classifier, use_cv=use_cv)
    logger.info(f"[train] AUC-ROC={metrics['auc_roc']} | F1={metrics['f1']}")


def run_stream(spark, classifier: str = "gbt", sink: str = "console",
               trigger: str = "30 seconds", **kwargs):
    from src.streaming.stream_predict import load_model, run_streaming_predictions
    model = load_model(classifier_type=classifier)
    run_streaming_predictions(
        spark, model=model,
        classifier_type=classifier,
        output_mode=sink,
        trigger_interval=trigger,
    )


# ─── Stage map ────────────────────────────────────────────────────────────────
STAGES = {
    "ingest":    run_ingest,
    "join":      run_join,
    "transform": run_transform,
    "eda":       run_eda,
    "features":  run_features,
    "monitor":   run_monitor,
    "train":     run_train,
    "stream":    run_stream,
}


# ─── Orchestrator ─────────────────────────────────────────────────────────────

class PipelineOrchestrator:
    def __init__(self, stages_to_run: List[str], skip: List[str], **stage_kwargs):
        self.stages_to_run = stages_to_run
        self.skip          = set(skip or [])
        self.stage_kwargs  = stage_kwargs
        self.results       = {}   # stage → {status, duration_s, error}

    def resolve_stages(self) -> List[str]:
        """Return ordered list of stages to execute."""
        if self.stages_to_run == ["all"]:
            ordered = STAGE_ORDER[:]
        else:
            # Validate names
            for s in self.stages_to_run:
                if s not in STAGES:
                    raise ValueError(
                        f"Unknown stage: '{s}'. Valid: {list(STAGES.keys())}"
                    )
            # Sort by canonical order so user can pass in any order
            ordered = [s for s in STAGE_ORDER if s in self.stages_to_run]

        # Remove skipped
        ordered = [s for s in ordered if s not in self.skip]
        return ordered

    def run(self, spark) -> bool:
        """
        Execute pipeline stages in order.
        Returns True if all stages passed, False if any failed.
        """
        stages = self.resolve_stages()
        total  = len(stages)
        logger.info(f"\n{'='*60}")
        logger.info(f"PIPELINE START | stages={stages}")
        logger.info(f"{'='*60}\n")

        all_passed = True
        for i, stage_name in enumerate(stages, 1):
            logger.info(f"[{i}/{total}] ── Stage: {stage_name.upper()} ──────────────")
            t0 = time.time()
            try:
                STAGES[stage_name](spark, **self.stage_kwargs)
                duration = round(time.time() - t0, 2)
                self.results[stage_name] = {"status": "✅ PASSED", "duration_s": duration}
                logger.info(f"[{stage_name}] PASSED in {duration}s\n")

            except Exception as e:
                duration = round(time.time() - t0, 2)
                self.results[stage_name] = {
                    "status":     "❌ FAILED",
                    "duration_s": duration,
                    "error":      str(e),
                }
                logger.error(f"[{stage_name}] FAILED after {duration}s: {e}")
                logger.error(traceback.format_exc())
                all_passed = False

                # Abort pipeline — downstream stages depend on upstream output
                logger.error(f"Aborting pipeline at stage '{stage_name}'. Fix and rerun.")
                break

        self._print_summary()
        return all_passed

    def _print_summary(self):
        logger.info(f"\n{'='*60}")
        logger.info("PIPELINE SUMMARY")
        logger.info(f"{'='*60}")
        for stage, result in self.results.items():
            err = f" → {result['error']}" if "error" in result else ""
            logger.info(
                f"  {result['status']}  {stage:<15} "
                f"({result['duration_s']}s){err}"
            )
        total_time = sum(r["duration_s"] for r in self.results.values())
        logger.info(f"\nTotal time: {total_time:.1f}s")
        logger.info(f"Log file:   {log_file}")
        logger.info(f"{'='*60}\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Spark ML Pipeline Orchestrator",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--run",
        default="all",
        help=(
            "Comma-separated stages to run, or 'all'.\n"
            f"Valid: {', '.join(STAGE_ORDER)}\n"
            "Examples:\n"
            "  --run all\n"
            "  --run features,train\n"
            "  --run train"
        ),
    )
    parser.add_argument(
        "--skip",
        default="",
        help="Comma-separated stages to skip. e.g. --skip eda",
    )
    parser.add_argument(
        "--classifier",
        default="gbt",
        choices=["gbt", "rf", "lr"],
        help="Classifier for train/stream stages.",
    )
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Enable CrossValidator HPO in train stage.",
    )
    parser.add_argument(
        "--sink",
        default="console",
        choices=["console", "parquet", "memory"],
        help="Output sink for stream stage.",
    )
    parser.add_argument(
        "--trigger",
        default="30 seconds",
        help="Streaming trigger interval. e.g. '10 seconds'",
    )
    parser.add_argument(
        "--list-stages",
        action="store_true",
        help="Print stage order and exit.",
    )
    return parser.parse_args()


# ─── Entrypoint ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.list_stages:
        print("\nPipeline stage order:")
        for i, s in enumerate(STAGE_ORDER, 1):
            print(f"  {i}. {s}")
        print()
        sys.exit(0)

    stages_to_run = [s.strip() for s in args.run.split(",")]
    skip_stages   = [s.strip() for s in args.skip.split(",") if s.strip()]

    stage_kwargs = {
        "classifier": args.classifier,
        "use_cv":     args.cv,
        "sink":       args.sink,
        "trigger":    args.trigger,
    }

    spark = get_spark_session()
    orchestrator = PipelineOrchestrator(
        stages_to_run=stages_to_run,
        skip=skip_stages,
        **stage_kwargs,
    )

    success = orchestrator.run(spark)
    stop_spark_session()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()