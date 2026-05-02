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
    # 2. Create Label (High Spender)
    # -------------------------------
    quantile = df.approxQuantile("total_spent", [0.7], 0.0)[0]

    df = df.withColumn(
        "label",
        when(col("total_spent") >= quantile, 1).otherwise(0)
    )
    df = df.withColumn(
        "label",
        (col("total_orders") > 1).cast("int")
    )

    print("Label Distribution:")
    df.groupBy("label").count().show()

    # -------------------------------
    # 3. Feature Selection (NO LEAKAGE)
    # -------------------------------
    feature_cols = [
        "avg_order_value",
        "max_price",
        "price_variance"
    ]

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features",
        handleInvalid="skip"   # 🔥 important
    )

    df = assembler.transform(df)

    # -------------------------------
    # 4. Train/Test Split
    # -------------------------------
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    # -------------------------------
    # 5. Train Model
    # -------------------------------
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label"
    )

    model = lr.fit(train_df)

    # -------------------------------
    # 6. Predictions
    # -------------------------------
    predictions = model.transform(test_df)

    predictions.select("label", "prediction", "probability").show(5, truncate=False)

    # -------------------------------
    # 7. Evaluate Model
    # -------------------------------
    evaluator = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )

    auc = evaluator.evaluate(predictions)

    print(f"🔥 Model AUC: {auc}")

    # -------------------------------
    # 8. Save Model
    # -------------------------------
    model.write().overwrite().save("models/logistic_model")

    print("Model saved")

    spark.stop()


if __name__ == "__main__":
    main()