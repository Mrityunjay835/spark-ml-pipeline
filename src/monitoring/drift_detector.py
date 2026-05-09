"""
Data Drift Detector — Phase 6.1

Two detection methods:
  1. PSI  (Population Stability Index) — industry standard for numeric drift
  2. KS   (Kolmogorov-Smirnov test)    — distribution shift for numeric features
  3. Chi-Square                         — categorical feature drift

PSI interpretation (industry standard):
  PSI < 0.10  → No drift        (safe)
  PSI < 0.20  → Minor drift     (monitor closely)
  PSI >= 0.20 → Major drift     (retrain now)

Usage:
  detector = DriftDetector(reference_df, current_df, spark)
  report   = detector.run()
  report.show_summary()
"""
import logging
import os
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

from src.config.constants import FEATURES_DIR, MODELS_DIR, LABEL_COL
from src.utils.spark_session import get_spark_session

logger = logging.getLogger(__name__)

# ─── PSI thresholds ───────────────────────────────────────────────────────────
PSI_SAFE    = 0.10
PSI_WARNING = 0.20   # >= this → retrain

# ─── Result dataclasses ───────────────────────────────────────────────────────

@dataclass
class FeatureDriftResult:
    feature:      str
    method:       str        # psi | ks | chi2
    score:        float
    status:       str        # SAFE | WARNING | DRIFT
    ref_mean:     Optional[float] = None
    cur_mean:     Optional[float] = None
    ref_std:      Optional[float] = None
    cur_std:      Optional[float] = None


@dataclass
class DriftReport:
    timestamp:      str
    total_features: int
    drifted:        List[FeatureDriftResult] = field(default_factory=list)
    warnings:       List[FeatureDriftResult] = field(default_factory=list)
    safe:           List[FeatureDriftResult] = field(default_factory=list)
    label_drift:    Optional[float] = None

    @property
    def has_critical_drift(self) -> bool:
        return len(self.drifted) > 0

    @property
    def drift_rate(self) -> float:
        return len(self.drifted) / self.total_features if self.total_features > 0 else 0.0

    def show_summary(self):
        print(f"\n{'='*65}")
        print(f"DRIFT REPORT  |  {self.timestamp}")
        print(f"{'='*65}")
        print(f"Features checked : {self.total_features}")
        print(f"🔴 DRIFT         : {len(self.drifted)}")
        print(f"🟡 WARNING       : {len(self.warnings)}")
        print(f"🟢 SAFE          : {len(self.safe)}")
        print(f"Drift rate       : {self.drift_rate:.1%}")
        if self.label_drift is not None:
            status = "🔴" if abs(self.label_drift) > 0.05 else "🟢"
            print(f"Label drift      : {status} {self.label_drift:+.4f}")

        if self.drifted or self.warnings:
            print(f"\n{'Feature':<35} {'Method':<6} {'Score':>8}  {'Status':<10} "
                  f"{'Ref Mean':>10} {'Cur Mean':>10}")
            print("-" * 85)
            for r in sorted(self.drifted + self.warnings,
                            key=lambda x: x.score, reverse=True):
                icon = "🔴" if r.status == "DRIFT" else "🟡"
                ref_m = f"{r.ref_mean:.3f}" if r.ref_mean is not None else "N/A"
                cur_m = f"{r.cur_mean:.3f}" if r.cur_mean is not None else "N/A"
                print(f"{r.feature:<35} {r.method:<6} {r.score:>8.4f}  "
                      f"{icon} {r.status:<8} {ref_m:>10} {cur_m:>10}")
        print(f"{'='*65}\n")

    def to_dict(self) -> dict:
        return {
            "timestamp":      self.timestamp,
            "total_features": self.total_features,
            "drift_rate":     self.drift_rate,
            "has_critical":   self.has_critical_drift,
            "label_drift":    self.label_drift,
            "drifted":  [r.feature for r in self.drifted],
            "warnings": [r.feature for r in self.warnings],
        }


# ─── Core detector ────────────────────────────────────────────────────────────

class DriftDetector:
    """
    Compares reference (training) distribution vs current (incoming) distribution.

    Args:
        reference_df: DataFrame used during model training (feature store snapshot)
        current_df:   DataFrame of recent incoming data (last N days / stream batch)
        n_bins:       Number of bins for PSI histogram (10 is standard)
    """

    def __init__(
        self,
        reference_df: DataFrame,
        current_df:   DataFrame,
        n_bins:       int = 10,
    ):
        self.ref = reference_df
        self.cur = current_df
        self.n_bins = n_bins

        # Cache both (will be used multiple times)
        self.ref.cache()
        self.cur.cache()

        self._ref_count = self.ref.count()
        self._cur_count = self.cur.count()
        logger.info(
            f"[drift] Reference: {self._ref_count:,} rows | "
            f"Current: {self._cur_count:,} rows"
        )

    # ── PSI ───────────────────────────────────────────────────────────────────

    def compute_psi(self, col_name: str) -> float:
        """
        PSI = Σ (actual% - expected%) * ln(actual% / expected%)

        Uses quantile-based binning on reference distribution.
        This ensures bins are meaningful (equal-frequency, not equal-width).
        """
        # Get bin edges from reference distribution
        quantile_points = [i / self.n_bins for i in range(1, self.n_bins)]
        try:
            bin_edges = self.ref.approxQuantile(col_name, quantile_points, 0.01)
        except Exception:
            return 0.0

        if not bin_edges or len(set(bin_edges)) < 2:
            return 0.0  # constant column — no drift possible

        # Deduplicate and sort edges
        bin_edges = sorted(set(bin_edges))

        def _bin_distribution(df: DataFrame, total: int) -> List[float]:
            """Assign rows to bins, return % per bin."""
            buckets = []

            # Below first edge
            count = df.filter(F.col(col_name) <= bin_edges[0]).count()
            buckets.append(count / total if total > 0 else 0)

            # Between edges
            for i in range(len(bin_edges) - 1):
                count = df.filter(
                    (F.col(col_name) > bin_edges[i]) &
                    (F.col(col_name) <= bin_edges[i + 1])
                ).count()
                buckets.append(count / total if total > 0 else 0)

            # Above last edge
            count = df.filter(F.col(col_name) > bin_edges[-1]).count()
            buckets.append(count / total if total > 0 else 0)

            return buckets

        ref_dist = _bin_distribution(self.ref, self._ref_count)
        cur_dist = _bin_distribution(self.cur, self._cur_count)

        # PSI formula — add small epsilon to avoid log(0)
        eps = 1e-6
        psi = 0.0
        for ref_pct, cur_pct in zip(ref_dist, cur_dist):
            ref_pct = max(ref_pct, eps)
            cur_pct = max(cur_pct, eps)
            psi += (cur_pct - ref_pct) * (
                __import__("math").log(cur_pct / ref_pct)
            )

        return round(psi, 6)

    # ── KS Test (approximated on Spark) ───────────────────────────────────────

    def compute_ks(self, col_name: str) -> float:
        """
        Kolmogorov-Smirnov statistic: max absolute difference between CDFs.
        Approximated using percentile comparison at N points.
        KS > 0.1 with large samples indicates significant drift.
        """
        points = [i / 100 for i in range(5, 100, 5)]  # 5th to 95th percentile
        try:
            ref_quantiles = self.ref.approxQuantile(col_name, points, 0.01)
            cur_quantiles = self.cur.approxQuantile(col_name, points, 0.01)
        except Exception:
            return 0.0

        if not ref_quantiles or not cur_quantiles:
            return 0.0

        # Normalize to [0,1] range for comparison
        ref_min = min(ref_quantiles)
        ref_max = max(ref_quantiles)
        ref_range = ref_max - ref_min if ref_max != ref_min else 1.0

        max_diff = 0.0
        for r, c in zip(ref_quantiles, cur_quantiles):
            diff = abs((c - r) / ref_range)
            max_diff = max(max_diff, diff)

        return round(max_diff, 6)

    # ── Mean shift ────────────────────────────────────────────────────────────

    def _get_stats(self, df: DataFrame, col_name: str) -> Tuple[float, float]:
        """Return (mean, std) for a column."""
        row = df.select(
            F.avg(col_name).alias("mean"),
            F.stddev(col_name).alias("std")
        ).collect()[0]
        return (row["mean"] or 0.0, row["std"] or 0.0)

    # ── Categorical drift (Chi-Square proxy) ──────────────────────────────────

    def compute_categorical_drift(self, col_name: str) -> float:
        """
        Normalized Chi-Square statistic for categorical columns.
        Measures how much the category distribution has shifted.
        Returns value in [0, 1] — higher = more drift.
        """
        ref_dist = (
            self.ref.groupBy(col_name)
            .count()
            .withColumn("ref_pct", F.col("count") / self._ref_count)
            .select(col_name, F.col("ref_pct"))
        )
        cur_dist = (
            self.cur.groupBy(col_name)
            .count()
            .withColumn("cur_pct", F.col("count") / self._cur_count)
            .select(col_name, F.col("cur_pct"))
        )

        joined = ref_dist.join(cur_dist, on=col_name, how="outer").fillna(0.0)
        total_variation = joined.select(
            F.sum(F.abs(F.col("ref_pct") - F.col("cur_pct"))).alias("tv")
        ).collect()[0]["tv"]

        return round((total_variation or 0.0) / 2.0, 6)  # normalize to [0,1]

    # ── Label drift ───────────────────────────────────────────────────────────

    def compute_label_drift(self) -> Optional[float]:
        """
        Shift in churn rate between reference and current.
        > 0.05 absolute shift = meaningful.
        """
        if LABEL_COL not in self.ref.columns or LABEL_COL not in self.cur.columns:
            return None

        ref_rate = self.ref.filter(
            F.col(LABEL_COL) == 1
        ).count() / self._ref_count

        cur_rate = self.cur.filter(
            F.col(LABEL_COL) == 1
        ).count() / self._cur_count

        shift = cur_rate - ref_rate
        logger.info(
            f"[drift] Label drift: ref_churn={ref_rate:.3f} | "
            f"cur_churn={cur_rate:.3f} | shift={shift:+.3f}"
        )
        return round(shift, 6)

    # ── Full report ───────────────────────────────────────────────────────────

    def run(
        self,
        numeric_cols:     Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
    ) -> DriftReport:
        """
        Run drift detection on all specified columns.
        Defaults to all numeric + string columns present in both DataFrames.
        """
        # Default: use columns present in both
        common_cols = set(self.ref.columns) & set(self.cur.columns)

        if numeric_cols is None:
            numeric_cols = [
                c for c, dtype in self.ref.dtypes
                if c in common_cols
                and dtype in ("double", "float", "int", "bigint", "long")
                and c != LABEL_COL
            ]

        if categorical_cols is None:
            categorical_cols = [
                c for c, dtype in self.ref.dtypes
                if c in common_cols and dtype == "string"
            ]

        report = DriftReport(
            timestamp=datetime.now().isoformat(),
            total_features=len(numeric_cols) + len(categorical_cols),
        )

        # ── Numeric: PSI ──────────────────────────────────────────────────────
        logger.info(f"[drift] Checking {len(numeric_cols)} numeric features via PSI...")
        for col_name in numeric_cols:
            try:
                psi   = self.compute_psi(col_name)
                r_mean, r_std = self._get_stats(self.ref, col_name)
                c_mean, c_std = self._get_stats(self.cur, col_name)

                if psi >= PSI_WARNING:
                    status = "DRIFT"
                elif psi >= PSI_SAFE:
                    status = "WARNING"
                else:
                    status = "SAFE"

                result = FeatureDriftResult(
                    feature=col_name, method="psi", score=psi, status=status,
                    ref_mean=r_mean, cur_mean=c_mean,
                    ref_std=r_std, cur_std=c_std,
                )

                if status == "DRIFT":
                    report.drifted.append(result)
                    logger.warning(f"[drift] 🔴 DRIFT  | {col_name:<35} PSI={psi:.4f}")
                elif status == "WARNING":
                    report.warnings.append(result)
                    logger.warning(f"[drift] 🟡 WARN   | {col_name:<35} PSI={psi:.4f}")
                else:
                    report.safe.append(result)
                    logger.info(f"[drift] 🟢 SAFE   | {col_name:<35} PSI={psi:.4f}")

            except Exception as e:
                logger.error(f"[drift] Error on column '{col_name}': {e}")

        # ── Categorical: Total Variation Distance ─────────────────────────────
        logger.info(f"[drift] Checking {len(categorical_cols)} categorical features...")
        for col_name in categorical_cols:
            try:
                score = self.compute_categorical_drift(col_name)

                if score >= 0.20:
                    status = "DRIFT"
                elif score >= 0.10:
                    status = "WARNING"
                else:
                    status = "SAFE"

                result = FeatureDriftResult(
                    feature=col_name, method="cat", score=score, status=status
                )
                if status == "DRIFT":
                    report.drifted.append(result)
                elif status == "WARNING":
                    report.warnings.append(result)
                else:
                    report.safe.append(result)

            except Exception as e:
                logger.error(f"[drift] Error on column '{col_name}': {e}")

        # ── Label drift ───────────────────────────────────────────────────────
        report.label_drift = self.compute_label_drift()

        return report


# ─── Save / load report ───────────────────────────────────────────────────────

def save_drift_report(report: DriftReport, output_dir: str = None) -> str:
    out_dir = output_dir or os.path.join(MODELS_DIR, "../logs/drift")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    logger.info(f"[drift] Report saved → {path}")
    return path


# ─── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    spark = get_spark_session()

    features_path = os.path.join(FEATURES_DIR, "user_features")
    full_df = spark.read.parquet(features_path)

    # Simulate drift: use first 70% as reference, last 30% as "current"
    # In prod: reference = training snapshot, current = last 7 days of scored data
    total = full_df.count()
    ref_df = full_df.limit(int(total * 0.7))
    cur_df = full_df.subtract(ref_df)

    logger.info("[drift] Running drift detection (simulated split)...")
    detector = DriftDetector(ref_df, cur_df)
    report   = detector.run()
    report.show_summary()
    save_drift_report(report)

    if report.has_critical_drift:
        logger.warning(
            f"[drift] ⚠️  {len(report.drifted)} features drifted. "
            f"Retraining recommended."
        )
    else:
        logger.info("[drift] ✅ No critical drift detected.")

    spark.stop()