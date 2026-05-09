"""
Spark ML preprocessing pipeline:
StringIndexer → OneHotEncoder → VectorAssembler → StandardScaler.
Returns fitted PipelineModel (serializable).
"""
import logging
import os
from typing import List, Tuple

from pyspark.sql import DataFrame
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    StandardScaler,
    Imputer,
    QuantileDiscretizer,
)
from pyspark.ml.classification import (
    GBTClassifier,
    RandomForestClassifier,
    LogisticRegression,
)

from src.config.constants import (
    MODELS_DIR, PIPELINE_PATH,
    LABEL_COL, FEATURES_COL,
    MAX_CATEGORIES,
    GBT_MAX_ITER, GBT_MAX_DEPTH, GBT_STEP_SIZE, GBT_SUBSAMPLING, GBT_SEED,
    RF_NUM_TREES, RF_MAX_DEPTH, RF_SEED,
    LR_MAX_ITER, LR_REG_PARAM, LR_ELASTIC_NET,
)

logger = logging.getLogger(__name__)


# ─── Feature column definitions ───────────────────────────────────────────────

NUMERIC_FEATURES = [
    # ── DO NOT include recency_days — it directly defines the churn label ──
    # churn_label = (recency_days > threshold) → using it is pure leakage

    # Frequency signals
    "frequency",              # num orders — legitimate predictor
    "avg_items_per_order",
    "total_items_purchased",

    # Monetary signals
    "monetary",
    "avg_order_value",
    "spend_trend",            # trajectory matters, not just total

    # Satisfaction signals
    "avg_review_score",
    "std_review_score",       # inconsistent reviews = at-risk customer
    "min_review_score",       # worst experience they had

    # Payment behavior
    "avg_installments",
    "avg_freight_ratio",

    # Engagement signals
    "weekend_purchase_ratio",
    "avg_purchase_hour",
    "avg_categories_per_order",

    # Operational signals
    "avg_delivery_delay",     # bad delivery experience → churn risk
    "max_delivery_delay",
    "total_freight",

    # Derived
    "high_value_customer",
    "is_repeat_buyer",
]

CATEGORICAL_FEATURES = [
    "customer_state",
    "primary_payment_type",
]


def build_preprocessing_pipeline(
    numeric_cols: List[str] = None,
    categorical_cols: List[str] = None,
    classifier_type: str = "gbt",
) -> Tuple[Pipeline, List[str]]:
    """
    Build the full preprocessing + classifier pipeline.

    Args:
        numeric_cols:     Override default numeric feature list.
        categorical_cols: Override default categorical feature list.
        classifier_type:  'gbt' | 'rf' | 'lr'

    Returns:
        (Pipeline, final_feature_cols)
    """
    num_cols = numeric_cols or NUMERIC_FEATURES
    cat_cols = categorical_cols or CATEGORICAL_FEATURES

    stages = []

    # ── 1. Impute remaining nulls (safety net after transform.py) ─────────────
    imputer = Imputer(
        inputCols=num_cols,
        outputCols=[f"{c}_imputed" for c in num_cols],
        strategy="median",
    )
    imputed_num_cols = [f"{c}_imputed" for c in num_cols]
    stages.append(imputer)

    # ── 2. StringIndexer for categoricals ─────────────────────────────────────
    indexed_cols = []
    for cat_col in cat_cols:
        out_col = f"{cat_col}_idx"
        indexer = StringIndexer(
            inputCol=cat_col,
            outputCol=out_col,
            handleInvalid="keep",           # 'keep' → unknown → last index
            stringOrderType="frequencyDesc",
        )
        stages.append(indexer)
        indexed_cols.append(out_col)

    # ── 3. OneHotEncoder for indexed categoricals ──────────────────────────────
    ohe_cols = [f"{c}_ohe" for c in indexed_cols]
    ohe = OneHotEncoder(
        inputCols=indexed_cols,
        outputCols=ohe_cols,
        dropLast=True,
    )
    stages.append(ohe)

    # ── 4. Assemble all features ───────────────────────────────────────────────
    assembler_inputs = imputed_num_cols + ohe_cols
    assembler = VectorAssembler(
        inputCols=assembler_inputs,
        outputCol="raw_features",
        handleInvalid="skip",
    )
    stages.append(assembler)

    # ── 5. Scale (only needed for LR; GBT/RF are scale-invariant) ─────────────
    if classifier_type == "lr":
        scaler = StandardScaler(
            inputCol="raw_features",
            outputCol=FEATURES_COL,
            withStd=True,
            withMean=False,          # sparse → withMean=False
        )
        stages.append(scaler)
        final_feature_col = FEATURES_COL
    else:
        # Rename for consistency
        assembler.setOutputCol(FEATURES_COL)
        final_feature_col = FEATURES_COL

    # ── 6. Classifier ──────────────────────────────────────────────────────────
    classifier = _build_classifier(classifier_type)
    stages.append(classifier)

    pipeline = Pipeline(stages=stages)
    logger.info(
        f"[pipeline] Built {len(stages)}-stage pipeline | "
        f"classifier={classifier_type} | "
        f"num_features={len(num_cols)} | cat_features={len(cat_cols)}"
    )
    return pipeline, assembler_inputs


def _build_classifier(classifier_type: str):
    """Factory for supported classifiers."""
    if classifier_type == "gbt":
        return GBTClassifier(
            featuresCol=FEATURES_COL,
            labelCol=LABEL_COL,
            maxIter=GBT_MAX_ITER,
            maxDepth=GBT_MAX_DEPTH,
            stepSize=GBT_STEP_SIZE,
            subsamplingRate=GBT_SUBSAMPLING,
            seed=GBT_SEED,
            # GBT doesn't support multiclass — binary only
        )
    elif classifier_type == "rf":
        return RandomForestClassifier(
            featuresCol=FEATURES_COL,
            labelCol=LABEL_COL,
            numTrees=RF_NUM_TREES,
            maxDepth=RF_MAX_DEPTH,
            seed=RF_SEED,
            featureSubsetStrategy="sqrt",
        )
    elif classifier_type == "lr":
        return LogisticRegression(
            featuresCol=FEATURES_COL,
            labelCol=LABEL_COL,
            maxIter=LR_MAX_ITER,
            regParam=LR_REG_PARAM,
            elasticNetParam=LR_ELASTIC_NET,
            family="binomial",
        )
    else:
        raise ValueError(f"Unsupported classifier: {classifier_type}. Choose gbt|rf|lr")


def get_feature_importance(
    pipeline_model: PipelineModel,
    feature_names: List[str],
    classifier_type: str = "gbt",
    top_n: int = 20,
) -> List[Tuple[str, float]]:
    """
    Extract feature importances from fitted tree model.
    Returns sorted list of (feature_name, importance).
    """
    # Last stage is the classifier
    classifier_model = pipeline_model.stages[-1]

    if not hasattr(classifier_model, "featureImportances"):
        logger.warning("[pipeline] Classifier has no featureImportances (LR uses coefficients).")
        return []

    importances = classifier_model.featureImportances.toArray()

    # Pad names if lengths don't match (OHE expands categoricals)
    if len(feature_names) != len(importances):
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    ranked = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]

    logger.info(f"[pipeline] Top {top_n} feature importances:")
    for name, imp in ranked:
        logger.info(f"  {name:<40} {imp:.6f}")

    return ranked


def save_pipeline(pipeline_model: PipelineModel, path: str = PIPELINE_PATH) -> None:
    pipeline_model.write().overwrite().save(path)
    logger.info(f"[pipeline] Saved pipeline → {path}")


def load_pipeline(path: str = PIPELINE_PATH) -> PipelineModel:
    model = PipelineModel.load(path)
    logger.info(f"[pipeline] Loaded pipeline from {path}")
    return model