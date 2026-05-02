from src.utils.spark_session import create_spark_session
from pyspark.sql.functions import (
    col, count, sum, avg, max, min,
    datediff, lit, when
)


def main():
    spark = create_spark_session("Feature Engineering")

    # -------------------------------
    # 1. Load Final Dataset
    # -------------------------------
    df = spark.read.parquet("data/processed/final_dataset")

    # -------------------------------
    # 2. Define Reference Date
    # -------------------------------
    reference_date = "2018-10-01"

    # -------------------------------
    # 3. Create Recent Order Flag (IMPORTANT)
    # -------------------------------
    df = df.withColumn(
        "is_recent_order",
        when(
            datediff(lit(reference_date), col("order_date")) <= 30,
            1
        ).otherwise(0)
    )

    # -------------------------------
    # 4. Aggregate User-Level Features
    # -------------------------------
    features_df = df.groupBy("customer_id").agg(
        count("order_id").alias("total_orders"),
        sum("price").alias("total_spent"),
        avg("price").alias("avg_order_value"),
        sum("is_recent_order").alias("recent_orders"),  # ✅ FIXED
        max("order_date").alias("last_order_date"),
        min("order_date").alias("first_order_date")
    )

    # -------------------------------
    # 5. Active Days Feature
    # -------------------------------
    features_df = features_df.withColumn(
        "active_days",
        datediff(col("last_order_date"), col("first_order_date"))
    )

    # -------------------------------
    # 6. Order Density Feature
    # -------------------------------
    features_df = features_df.withColumn(
        "order_density",
        col("total_orders") / (col("active_days") + 1)
    )

    # -------------------------------
    # 7. Recency (for label creation only)
    # -------------------------------
    features_df = features_df.withColumn(
        "recency_days",
        datediff(lit(reference_date), col("last_order_date"))
    )

    # -------------------------------
    # 8. Handle Missing Values
    # -------------------------------
    features_df = features_df.fillna({
        "total_orders": 0,
        "total_spent": 0,
        "avg_order_value": 0,
        "recent_orders": 0,
        "active_days": 0,
        "order_density": 0,
        "recency_days": 999
    })

    # -------------------------------
    # 9. Save Features
    # -------------------------------
    features_df.write \
        .mode("overwrite") \
        .parquet("data/features/user_features")

    print("Feature engineering completed")

    spark.stop()


if __name__ == "__main__":
    main()