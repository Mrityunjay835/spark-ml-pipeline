from src.utils.spark_session import create_spark_session
from pyspark.sql.functions import col, avg


def main():
    spark = create_spark_session("EDA")

    # Load features
    df = spark.read.parquet("data/features/user_features")

    # -------------------------------
    # Create Label (same as training)
    # -------------------------------
    quantile = df.approxQuantile("total_spent", [0.7], 0.0)[0]

    df = df.withColumn(
        "label",
        (col("total_spent") >= quantile).cast("int")
    )

    # -------------------------------
    # Basic Distribution
    # -------------------------------
    print("Label Distribution:")
    df.groupBy("label").count().show()

    # -------------------------------
    # Feature Comparison
    # -------------------------------
    print("Feature Comparison by Label:")
    df.groupBy("label").agg(
        avg("total_orders").alias("avg_orders"),
        avg("avg_order_value").alias("avg_order_value"),
        # avg("order_density").alias("avg_density"),
        # avg("recent_orders").alias("avg_recent_orders")
    ).show()

    spark.stop()


if __name__ == "__main__":
    main()