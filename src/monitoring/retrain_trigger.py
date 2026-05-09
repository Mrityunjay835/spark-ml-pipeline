"""
Retrain Trigger — Phase 6.3

Automatically triggers model retraining when:
  1. PSI drift detected on > N features
  2. Predicted churn rate shifts > threshold vs baseline
  3. Scheduled interval exceeded (time-based fallback)

In prod: replace subprocess call with Airflow DAG trigger / Spark job submission.
"""
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from pyspark.sql import SparkSession

from src.config.constants import FEATURES_DIR, MODELS_DIR, LABEL_COL
from src.monitoring.drift_detector import DriftDetector, DriftReport, save_drift_report
from src.utils.spark_session import get_spark_session

logger = logging.getLogger(__name__)

TRIGGER_LOG_PATH = os.path.join(
    os.path.dirname(MODELS_DIR), "logs", "retrain_triggers.json"
)


# ─── Trigger config ───────────────────────────────────────────────────────────

@dataclass
class RetriggerConfig:
    # Drift thresholds
    max_drifted_features:  int   = 3       # retrain if > N features drift
    max_drift_rate:        float = 0.20    # retrain if > 20% features drift
    max_label_drift:       float = 0.05    # retrain if churn rate shifts > 5%

    # Scheduled retraining fallback
    max_days_since_train:  int   = 30      # retrain every 30 days regardless

    # Training config
    classifier_type:       str   = "gbt"
    use_cv:                bool  = False


# ─── Retrain trigger ──────────────────────────────────────────────────────────

class RetrainTrigger:
    """
    Evaluates drift report + training history → decides whether to retrain.

    Decision logic (any condition triggers retrain):
      1. Too many drifted features
      2. Label drift too high
      3. Time since last train exceeds max_days
    """

    def __init__(self, config: RetriggerConfig = None):
        self.config = config or RetriggerConfig()

    def should_retrain(self, report: DriftReport) -> tuple[bool, str]:
        """
        Returns (should_retrain: bool, reason: str).
        """
        # ── Condition 1: Feature drift ────────────────────────────────────────
        n_drifted   = len(report.drifted)
        drift_rate  = report.drift_rate

        if n_drifted >= self.config.max_drifted_features:
            return True, (
                f"{n_drifted} features drifted "
                f"(threshold={self.config.max_drifted_features})"
            )

        if drift_rate >= self.config.max_drift_rate:
            return True, (
                f"Drift rate {drift_rate:.1%} exceeded "
                f"threshold {self.config.max_drift_rate:.0%}"
            )

        # ── Condition 2: Label drift ──────────────────────────────────────────
        if report.label_drift is not None:
            if abs(report.label_drift) >= self.config.max_label_drift:
                return True, (
                    f"Label drift {report.label_drift:+.3f} exceeded "
                    f"threshold ±{self.config.max_label_drift}"
                )

        # ── Condition 3: Time-based ───────────────────────────────────────────
        days_since = self._days_since_last_train()
        if days_since is not None and days_since >= self.config.max_days_since_train:
            return True, (
                f"Model is {days_since} days old "
                f"(max={self.config.max_days_since_train} days)"
            )

        return False, "All checks passed — no retraining needed"

    def _days_since_last_train(self) -> Optional[int]:
        """
        Check most recent metrics JSON for training timestamp.
        Returns None if no training history found.
        """
        metrics_files = []
        for f in os.listdir(MODELS_DIR):
            if f.startswith("metrics_") and f.endswith(".json"):
                metrics_files.append(os.path.join(MODELS_DIR, f))

        if not metrics_files:
            return None

        latest = max(metrics_files, key=os.path.getmtime)
        try:
            with open(latest) as f:
                data = json.load(f)
            ts_str = data.get("timestamp")
            if ts_str:
                ts = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
                return (datetime.now() - ts).days
        except Exception:
            pass
        return None

    def trigger_retrain(self, reason: str) -> bool:
        """
        Execute retraining. In prod replace with:
          - Airflow: AirflowClient().trigger_dag('retrain_pipeline')
          - Databricks: requests.post(jobs_api, json={"job_id": retrain_job_id})
          - Spark submit: spark-submit src/ml/train.py
        """
        logger.info(f"\n{'='*55}")
        logger.info(f"🔄 RETRAINING TRIGGERED")
        logger.info(f"Reason   : {reason}")
        logger.info(f"Classifier: {self.config.classifier_type}")
        logger.info(f"Timestamp : {datetime.now().isoformat()}")
        logger.info(f"{'='*55}")

        self._log_trigger(reason)

        # ── Local execution (dev/demo) ─────────────────────────────────────────
        cmd = [
            sys.executable, "-m", "src.ml.train",
            "--classifier", self.config.classifier_type,
        ]
        if self.config.use_cv:
            cmd.append("--cv")

        logger.info(f"[retrain] Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            )
            if result.returncode == 0:
                logger.info("[retrain] ✅ Retraining completed successfully.")
                return True
            else:
                logger.error(f"[retrain] ❌ Retraining failed:\n{result.stderr}")
                return False
        except Exception as e:
            logger.error(f"[retrain] ❌ Failed to launch training: {e}")
            return False

    def _log_trigger(self, reason: str) -> None:
        """Append trigger event to audit log."""
        os.makedirs(os.path.dirname(TRIGGER_LOG_PATH), exist_ok=True)

        history = []
        if os.path.exists(TRIGGER_LOG_PATH):
            try:
                with open(TRIGGER_LOG_PATH) as f:
                    history = json.load(f)
            except Exception:
                pass

        history.append({
            "timestamp":  datetime.now().isoformat(),
            "reason":     reason,
            "classifier": self.config.classifier_type,
        })

        with open(TRIGGER_LOG_PATH, "w") as f:
            json.dump(history, f, indent=2)


# ─── Full drift + retrain cycle ───────────────────────────────────────────────

def run_drift_and_retrain_cycle(
    spark:          SparkSession,
    reference_path: str = None,
    current_path:   str = None,
    config:         RetriggerConfig = None,
) -> None:
    """
    Full cycle:
      1. Load reference + current data
      2. Run drift detection
      3. Decide if retrain needed
      4. Trigger retrain if yes
    """
    ref_path = reference_path or os.path.join(FEATURES_DIR, "user_features")
    cur_path = current_path   or ref_path   # in prod: use last 7-day window

    logger.info("[cycle] Loading reference and current data...")
    full_df = spark.read.parquet(ref_path)
    total   = full_df.count()

    # Simulate reference vs current split
    # In prod: reference = training date snapshot, current = recent predictions
    ref_df = full_df.limit(int(total * 0.7))
    cur_df = full_df.subtract(ref_df)

    logger.info("[cycle] Running drift detection...")
    detector = DriftDetector(ref_df, cur_df)
    report   = detector.run()
    report.show_summary()
    save_drift_report(report)

    # Decide
    trigger    = RetrainTrigger(config or RetriggerConfig())
    do_retrain, reason = trigger.should_retrain(report)

    if do_retrain:
        logger.warning(f"[cycle] ⚠️  Retraining triggered: {reason}")
        success = trigger.trigger_retrain(reason)
        if success:
            logger.info("[cycle] ✅ Retrain cycle complete.")
        else:
            logger.error("[cycle] ❌ Retrain failed. Manual intervention required.")
    else:
        logger.info(f"[cycle] ✅ {reason}")


# ─── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", default="gbt", choices=["gbt", "rf", "lr"])
    parser.add_argument("--cv", action="store_true")
    parser.add_argument("--max-drifted", type=int, default=3)
    parser.add_argument("--max-days",    type=int, default=30)
    args = parser.parse_args()

    config = RetriggerConfig(
        classifier_type=args.classifier,
        use_cv=args.cv,
        max_drifted_features=args.max_drifted,
        max_days_since_train=args.max_days,
    )

    spark = get_spark_session()
    run_drift_and_retrain_cycle(spark, config=config)
    spark.stop()