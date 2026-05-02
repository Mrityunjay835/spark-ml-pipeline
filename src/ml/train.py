from src.utils.spark_session import create_spark_session
from pyspark.sql.functions import col, when

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator


def main():
    spark = create_spark_session("ML Training")

    # -------------------------------
    # 1. Load Features
    # -------------------------------
    df = spark.read.parquet("data/features/user_features")

    # -------------------------------
    # 2. Create Balanced Label
    # -------------------------------
    quantile = df.approxQuantile("recency_days", [0.5], 0.0)[0]

    df = df.withColumn(
        "label",
        when(col("recency_days") <= quantile, 1).otherwise(0)
    )

    # -------------------------------
    # 3. Check Label Distribution
    # -------------------------------
    print("Label Distribution:")
    df.groupBy("label").count().show()

    # -------------------------------
    # 4. Feature Selection (NO LEAKAGE)
    # -------------------------------
    feature_cols = [
        "total_orders",
        "total_spent",
        "avg_order_value",
        "active_days",
        "order_density",
        "recent_orders"
    ]

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features"
    )

    df = assembler.transform(df)

    # -------------------------------
    # 5. Train/Test Split
    # -------------------------------
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    # -------------------------------
    # 6. Train Model
    # -------------------------------
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label"
    )

    model = lr.fit(train_df)

    # -------------------------------
    # 7. Predictions
    # -------------------------------
    predictions = model.transform(test_df)

    predictions.select("label", "prediction", "probability").show(5, truncate=False)

    # -------------------------------
    # 8. Evaluate Model
    # -------------------------------
    evaluator = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )

    auc = evaluator.evaluate(predictions)

    print(f"Model AUC: {auc}")

    # -------------------------------
    # 9. Save Model
    # -------------------------------
    model.write().overwrite().save("models/logistic_model")

    print("✅ Model saved")

    spark.stop()


if __name__ == "__main__":
    main()