"""
Training entrypoint: loads features → builds pipeline → trains → evaluates → saves.
Supports GBT / RF / LR with optional CrossValidator hyperparameter tuning.
"""
import logging
import os
import json
from datetime import datetime
from typing import Dict, Tuple, Optional

from pyspark.sql import SparkSession, DataFrame
from pyspark.ml import PipelineModel
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, CrossValidatorModel
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.sql import functions as F

from src.config.constants import (
    FEATURES_DIR, MODELS_DIR, MODEL_PATH,
    LABEL_COL, PREDICTION_COL, PROB_COL,
    TRAIN_RATIO, TEST_RATIO, SPLIT_SEED,
)
from src.ml.pipeline import (
    build_preprocessing_pipeline,
    get_feature_importance,
    save_pipeline,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES,
)
from src.utils.spark_session import get_spark_session

logger = logging.getLogger(__name__)


def load_features(spark: SparkSession) -> DataFrame:
    path = os.path.join(FEATURES_DIR, "user_features")
    df = spark.read.parquet(path)
    logger.info(f"[train] Loaded features: {df.count():,} rows, {len(df.columns)} cols")
    return df


def train_test_split(df: DataFrame) -> Tuple[DataFrame, DataFrame]:
    train, test = df.randomSplit([TRAIN_RATIO, TEST_RATIO], seed=SPLIT_SEED)
    train.cache()
    test.cache()
    logger.info(f"[train] Split → train={train.count():,} | test={test.count():,}")
    return train, test


def compute_class_weights(train_df: DataFrame) -> Dict[int, float]:
    """
    Balanced class weights: weight_i = total / (num_classes * count_i).
    Handles imbalanced churn data without SMOTE.
    """
    total = train_df.count()
    counts = {
        row[LABEL_COL]: row["count"]
        for row in train_df.groupBy(LABEL_COL).count().collect()
    }
    num_classes = len(counts)
    weights = {
        label: total / (num_classes * count)
        for label, count in counts.items()
    }
    logger.info(f"[train] Class weights: {weights}")
    return weights


def add_sample_weights(df: DataFrame, weights: Dict[int, float]) -> DataFrame:
    """Add weightCol for classifiers that support it (RF, LR)."""
    mapping = F.create_map(*[
        item for label, w in weights.items()
        for item in (F.lit(label), F.lit(w))
    ])
    return df.withColumn("sample_weight", mapping[F.col(LABEL_COL)])


def train_model(
    train_df: DataFrame,
    classifier_type: str = "gbt",
    use_cv: bool = False,
    cv_folds: int = 3,
) -> PipelineModel:
    """
    Train a pipeline model. Optionally runs CrossValidator for HPO.

    Args:
        classifier_type: 'gbt' | 'rf' | 'lr'
        use_cv:         If True, run 3-fold CV with param grid
        cv_folds:       Number of CV folds
    """
    pipeline, feature_cols = build_preprocessing_pipeline(
        numeric_cols=NUMERIC_FEATURES,
        categorical_cols=CATEGORICAL_FEATURES,
        classifier_type=classifier_type,
    )

    evaluator = BinaryClassificationEvaluator(
        labelCol=LABEL_COL,
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    )

    if use_cv:
        classifier = pipeline.getStages()[-1]
        param_grid = _build_param_grid(classifier, classifier_type)

        cv = CrossValidator(
            estimator=pipeline,
            estimatorParamMaps=param_grid,
            evaluator=evaluator,
            numFolds=cv_folds,
            seed=42,
            parallelism=2,          # parallel grid search
        )
        logger.info(f"[train] Running CrossValidator ({cv_folds} folds, "
                    f"{len(param_grid)} param combos)...")
        cv_model = cv.fit(train_df)
        model = cv_model.bestModel
        logger.info(f"[train] Best CV AUC-ROC: {max(cv_model.avgMetrics):.4f}")
    else:
        logger.info(f"[train] Training {classifier_type} pipeline...")
        model = pipeline.fit(train_df)

    return model


def _build_param_grid(classifier, classifier_type: str):
    """Minimal but useful param grid per classifier type."""
    grid = ParamGridBuilder()

    if classifier_type == "gbt":
        grid = (
            grid
            .addGrid(classifier.maxDepth, [3, 5, 7])
            .addGrid(classifier.stepSize, [0.05, 0.1, 0.2])
        )
    elif classifier_type == "rf":
        grid = (
            grid
            .addGrid(classifier.numTrees, [50, 100, 200])
            .addGrid(classifier.maxDepth, [5, 8])
        )
    elif classifier_type == "lr":
        grid = (
            grid
            .addGrid(classifier.regParam, [0.001, 0.01, 0.1])
            .addGrid(classifier.elasticNetParam, [0.0, 0.5, 1.0])
        )

    return grid.build()


def evaluate_model(model: PipelineModel, test_df: DataFrame) -> Dict[str, float]:
    """
    Compute AUC-ROC, AUC-PR, accuracy, F1, precision, recall.
    Returns metrics dict for logging / experiment tracking.
    """
    predictions = model.transform(test_df)

    # Binary evaluator (AUC-ROC, AUC-PR)
    bin_eval = BinaryClassificationEvaluator(
        labelCol=LABEL_COL,
        rawPredictionCol="rawPrediction",
    )
    auc_roc = bin_eval.evaluate(predictions, {bin_eval.metricName: "areaUnderROC"})
    auc_pr  = bin_eval.evaluate(predictions, {bin_eval.metricName: "areaUnderPR"})

    # Multiclass evaluator (accuracy, F1, precision, recall)
    mc_eval = MulticlassClassificationEvaluator(
        labelCol=LABEL_COL,
        predictionCol=PREDICTION_COL,
    )
    accuracy  = mc_eval.evaluate(predictions, {mc_eval.metricName: "accuracy"})
    f1        = mc_eval.evaluate(predictions, {mc_eval.metricName: "f1"})
    precision = mc_eval.evaluate(predictions, {mc_eval.metricName: "weightedPrecision"})
    recall    = mc_eval.evaluate(predictions, {mc_eval.metricName: "weightedRecall"})

    metrics = {
        "auc_roc":   round(auc_roc,   4),
        "auc_pr":    round(auc_pr,    4),
        "accuracy":  round(accuracy,  4),
        "f1":        round(f1,        4),
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
    }

    logger.info("[train] Evaluation metrics:")
    for k, v in metrics.items():
        logger.info(f"  {k:<15}: {v}")

    # Confusion matrix
    cm = (
        predictions
        .groupBy(LABEL_COL, PREDICTION_COL)
        .count()
        .orderBy(LABEL_COL, PREDICTION_COL)
    )
    logger.info("[train] Confusion matrix:")
    cm.show()

    return metrics


def save_metrics(metrics: Dict[str, float], classifier_type: str) -> None:
    """Persist metrics JSON for experiment tracking (replace with MLflow in prod)."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_path = os.path.join(MODELS_DIR, f"metrics_{classifier_type}_{ts}.json")
    metrics["classifier"] = classifier_type
    metrics["timestamp"] = ts

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"[train] Metrics saved → {metrics_path}")


def run_training(
    spark: SparkSession,
    classifier_type: str = "gbt",
    use_cv: bool = False,
) -> Tuple[PipelineModel, Dict[str, float]]:
    """Full train pipeline: load → split → train → evaluate → save."""
    features = load_features(spark)

    # Drop rows with null label
    features = features.filter(F.col(LABEL_COL).isNotNull())

    train_df, test_df = train_test_split(features)

    model = train_model(train_df, classifier_type=classifier_type, use_cv=use_cv)
    metrics = evaluate_model(model, test_df)

    # Feature importance
    get_feature_importance(
        model,
        feature_names=NUMERIC_FEATURES,
        classifier_type=classifier_type,
    )

    # Save
    model_path = os.path.join(MODELS_DIR, f"model_{classifier_type}")
    save_pipeline(model, path=model_path)
    save_metrics(metrics, classifier_type)

    return model, metrics


# ─── Entrypoint ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", default="gbt", choices=["gbt", "rf", "lr"])
    parser.add_argument("--cv", action="store_true", help="Enable CrossValidator HPO")
    args = parser.parse_args()

    spark = get_spark_session()
    model, metrics = run_training(spark, classifier_type=args.classifier, use_cv=args.cv)

    logger.info("=== Training Complete ===")
    logger.info(f"AUC-ROC: {metrics['auc_roc']} | F1: {metrics['f1']}")
    spark.stop()