from src.utils.spark_session import create_spark_session
from src.config.constants import *

from pyspark.sql.functions import col, to_date


def main():
    # Create Spark Session
    spark = create_spark_session("Olist Transformation")

    # -------------------------------
    # 1. Read Raw Orders Data
    # -------------------------------
    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv("data/raw/olist/olist_orders_dataset.csv")
    )

    print("Initial Rows:", df.count())

    # -------------------------------
    # 2. Filter Valid Orders
    # -------------------------------
    df = df.filter(col(ORDER_STATUS) == "delivered")

    # -------------------------------
    # 3. Handle Missing Values
    # -------------------------------
    df = df.dropna(subset=[ORDER_TIMESTAMP])

    # -------------------------------
    # 4. Select Required Columns
    # -------------------------------
    df = df.select(
        ORDER_ID,
        CUSTOMER_ID,
        ORDER_TIMESTAMP
    )

    # -------------------------------
    # 5. Remove Duplicates
    # -------------------------------
    df = df.dropDuplicates([ORDER_ID])

    # -------------------------------
    # 6. Feature Preparation (Date Extraction)
    # -------------------------------
    df = df.withColumn(
        ORDER_DATE,
        to_date(col(ORDER_TIMESTAMP))
    )

    print("Cleaned Rows:", df.count())

    # -------------------------------
    # 7. Write as Parquet (Partitioned)
    # -------------------------------
    df.write \
        .mode("overwrite") \
        .partitionBy(ORDER_DATE) \
        .parquet("data/processed/orders_partitioned")

    print("✅ Data successfully written in partitioned Parquet format")

    spark.stop()


if __name__ == "__main__":
    main()