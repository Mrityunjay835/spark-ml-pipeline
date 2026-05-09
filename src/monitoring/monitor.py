"""
Pipeline Health Monitor — Phase 6.2

Tracks:
  - Row counts per stage (catch silent data loss)
  - Null rates per column (catch upstream schema issues)
  - Stage latency (catch slowdowns)
  - Prediction distribution (catch model degradation)
  - Anomaly alerts (threshold-based)

In prod: replace file-based metrics with Prometheus + Grafana.
"""
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.config.constants import (
    PROCESSED_DIR, FEATURES_DIR, MODELS_DIR,
    LABEL_COL, PREDICTION_COL,
)
from src.utils.spark_session import get_spark_session

logger = logging.getLogger(__name__)

METRICS_DIR = os.path.join(os.path.dirname(MODELS_DIR), "logs", "metrics")


# ─── Metric dataclasses ───────────────────────────────────────────────────────

@dataclass
class StageMetrics:
    stage:        str
    row_count:    int
    null_rates:   Dict[str, float]    # col → null %
    duration_s:   float
    timestamp:    str
    anomalies:    List[str] = field(default_factory=list)


@dataclass
class PredictionMetrics:
    batch_id:         str
    total_predictions:int
    churn_rate:       float           # % predicted as churned
    avg_churn_prob:   float
    risk_distribution:Dict[str, int]  # HIGH/MEDIUM/LOW/SAFE counts
    timestamp:        str
    anomalies:        List[str] = field(default_factory=list)


# ─── Monitor class ────────────────────────────────────────────────────────────

class PipelineMonitor:
    """
    Wraps each pipeline stage to collect health metrics automatically.

    Usage:
        monitor = PipelineMonitor()

        with monitor.track("transform"):
            df = run_transform_pipeline(raw_df)
            monitor.record_dataframe("transform", df)

        monitor.save_report()
    """

    # Thresholds — tune per dataset
    MAX_NULL_RATE        = 0.05     # >5% nulls in any column → alert
    MIN_ROW_COUNT        = 100      # fewer rows than this → alert
    MAX_CHURN_RATE       = 0.60     # >60% predicted churn → model likely broken
    MIN_CHURN_RATE       = 0.01     # <1% predicted churn → model likely broken
    MAX_ROW_DROP_PCT     = 0.20     # >20% row drop vs previous stage → alert

    def __init__(self):
        self.stage_metrics:      List[StageMetrics]      = []
        self.prediction_metrics: List[PredictionMetrics] = []
        self._prev_row_count:    Optional[int]           = None
        self._active_stage:      Optional[str]           = None
        self._stage_start:       Optional[float]         = None

    # ── Context manager for timing ────────────────────────────────────────────

    def track(self, stage_name: str):
        """Use as context manager: `with monitor.track('ingest'): ...`"""
        return _StageTimer(self, stage_name)

    def _start_stage(self, stage_name: str):
        self._active_stage = stage_name
        self._stage_start  = time.time()

    def _end_stage(self) -> float:
        duration = round(time.time() - self._stage_start, 2)
        self._active_stage = None
        return duration

    # ── DataFrame health check ────────────────────────────────────────────────

    def record_dataframe(
        self,
        stage: str,
        df: DataFrame,
        duration_s: float = 0.0,
        critical_cols: Optional[List[str]] = None,
    ) -> StageMetrics:
        """
        Compute health metrics for a DataFrame output from a pipeline stage.
        critical_cols: columns where ANY null is a critical alert.
        """
        row_count = df.count()
        anomalies = []

        # Row count checks
        if row_count < self.MIN_ROW_COUNT:
            msg = f"[{stage}] ⚠️  Only {row_count} rows — possible data loss"
            logger.warning(msg)
            anomalies.append(msg)

        if self._prev_row_count is not None:
            drop_pct = (self._prev_row_count - row_count) / self._prev_row_count
            if drop_pct > self.MAX_ROW_DROP_PCT:
                msg = (f"[{stage}] ⚠️  Row count dropped {drop_pct:.1%} "
                       f"({self._prev_row_count:,} → {row_count:,})")
                logger.warning(msg)
                anomalies.append(msg)

        self._prev_row_count = row_count

        # Null rate per column
        null_rates = {}
        if row_count > 0:
            null_exprs = [
                F.avg(F.col(c).isNull().cast("double")).alias(c)
                for c in df.columns
            ]
            null_row = df.select(null_exprs).collect()[0].asDict()

            for col_name, rate in null_row.items():
                rate = rate or 0.0
                null_rates[col_name] = round(rate, 4)

                if rate > self.MAX_NULL_RATE:
                    msg = f"[{stage}] ⚠️  High null rate on '{col_name}': {rate:.1%}"
                    logger.warning(msg)
                    anomalies.append(msg)

                if critical_cols and col_name in critical_cols and rate > 0:
                    msg = f"[{stage}] 🔴 CRITICAL null on '{col_name}': {rate:.1%}"
                    logger.error(msg)
                    anomalies.append(msg)

        metrics = StageMetrics(
            stage=stage,
            row_count=row_count,
            null_rates=null_rates,
            duration_s=duration_s,
            timestamp=datetime.now().isoformat(),
            anomalies=anomalies,
        )
        self.stage_metrics.append(metrics)

        status = "✅" if not anomalies else "⚠️ "
        logger.info(
            f"[monitor] {status} {stage:<15} | "
            f"rows={row_count:>8,} | "
            f"duration={duration_s}s | "
            f"anomalies={len(anomalies)}"
        )
        return metrics

    # ── Prediction health ─────────────────────────────────────────────────────

    def record_predictions(
        self,
        df: DataFrame,
        batch_id: str = "batch_0",
    ) -> PredictionMetrics:
        """
        Monitor model output distribution.
        Catches model degradation via prediction shift.
        """
        total = df.count()
        if total == 0:
            logger.warning("[monitor] Empty prediction batch.")
            return None

        anomalies = []

        # Churn rate
        churn_count = df.filter(F.col(PREDICTION_COL) == 1).count()
        churn_rate  = churn_count / total

        if churn_rate > self.MAX_CHURN_RATE:
            msg = f"[monitor] 🔴 Churn rate too high: {churn_rate:.1%} (max={self.MAX_CHURN_RATE:.0%})"
            logger.error(msg)
            anomalies.append(msg)
        elif churn_rate < self.MIN_CHURN_RATE:
            msg = f"[monitor] 🔴 Churn rate too low: {churn_rate:.1%} (min={self.MIN_CHURN_RATE:.0%})"
            logger.error(msg)
            anomalies.append(msg)

        # Average churn probability
        avg_prob = 0.0
        if "churn_probability" in df.columns:
            avg_prob = df.select(F.avg("churn_probability")).collect()[0][0] or 0.0

        # Risk tier distribution
        risk_dist = {}
        if "risk_tier" in df.columns:
            for row in df.groupBy("risk_tier").count().collect():
                risk_dist[row["risk_tier"]] = row["count"]

        metrics = PredictionMetrics(
            batch_id=batch_id,
            total_predictions=total,
            churn_rate=round(churn_rate, 4),
            avg_churn_prob=round(avg_prob, 4),
            risk_distribution=risk_dist,
            timestamp=datetime.now().isoformat(),
            anomalies=anomalies,
        )
        self.prediction_metrics.append(metrics)

        logger.info(
            f"[monitor] Predictions | total={total:,} | "
            f"churn_rate={churn_rate:.1%} | "
            f"avg_prob={avg_prob:.3f} | "
            f"risk={risk_dist}"
        )
        return metrics

    # ── Row count lineage ─────────────────────────────────────────────────────

    def print_lineage(self):
        """Print row count at each stage — quickly spots where data is lost."""
        print(f"\n{'='*55}")
        print("PIPELINE ROW COUNT LINEAGE")
        print(f"{'='*55}")
        prev = None
        for m in self.stage_metrics:
            if prev is not None:
                delta = m.row_count - prev
                pct   = delta / prev * 100 if prev > 0 else 0
                arrow = f"({pct:+.1f}%)"
            else:
                arrow = "(start)"
            status = "⚠️ " if m.anomalies else "✅"
            print(f"  {status} {m.stage:<20} {m.row_count:>10,} rows  {arrow}")
            prev = m.row_count
        print(f"{'='*55}\n")

    # ── Save report ───────────────────────────────────────────────────────────

    def save_report(self) -> str:
        os.makedirs(METRICS_DIR, exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(METRICS_DIR, f"pipeline_health_{ts}.json")

        report = {
            "timestamp": ts,
            "stages": [asdict(m) for m in self.stage_metrics],
            "predictions": [asdict(m) for m in self.prediction_metrics],
            "total_anomalies": sum(
                len(m.anomalies) for m in self.stage_metrics + self.prediction_metrics
            ),
        }
        with open(path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"[monitor] Report saved → {path}")
        return path


# ─── Context manager helper ───────────────────────────────────────────────────

class _StageTimer:
    def __init__(self, monitor: PipelineMonitor, stage: str):
        self.monitor = monitor
        self.stage   = stage
        self._start  = None

    def __enter__(self):
        self._start = time.time()
        logger.info(f"[monitor] ▶  Starting stage: {self.stage}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = round(time.time() - self._start, 2)
        if exc_type:
            logger.error(f"[monitor] ✗  Stage '{self.stage}' FAILED after {duration}s")
        else:
            logger.info(f"[monitor] ✓  Stage '{self.stage}' done in {duration}s")
        return False  # don't suppress exceptions


# ─── Standalone health check ──────────────────────────────────────────────────

def run_health_check(spark: SparkSession) -> None:
    """Quick health check on all existing processed datasets."""
    monitor = PipelineMonitor()

    datasets = {
        "olist_orders":      os.path.join(PROCESSED_DIR, "olist_orders"),
        "olist_wide":        os.path.join(PROCESSED_DIR, "olist_wide"),
        "olist_transformed": os.path.join(PROCESSED_DIR, "olist_transformed"),
        "user_features":     os.path.join(FEATURES_DIR,  "user_features"),
    }

    for stage, path in datasets.items():
        if os.path.exists(path):
            df = spark.read.parquet(path)
            monitor.record_dataframe(stage, df)
        else:
            logger.warning(f"[monitor] Dataset not found: {path}")

    monitor.print_lineage()
    monitor.save_report()


# ─── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    spark = get_spark_session()
    run_health_check(spark)
    spark.stop()