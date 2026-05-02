from src.utils.spark_session import create_spark_session
from pyspark.sql.functions import col

def main():
    # Create Spark Session
    spark = create_spark_session("Olist Query Partition")

    # -------------------------------
    # 1. Read Transformed Orders Data
    # -------------------------------
    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .parquet("data/processed/orders_partitioned")
    )

    print("Total Rows:", df.count())

    # -------------------------------
    # 2. Apply Filter on a Partition Column
    # -------------------------------

    df_filtered = df.filter(col("order_date") == "2017-01-01")
    print("Filtered Rows (2017-01-01):", df_filtered.count())

    # -------------------------------
    # 3. Show Sample of Filtered Data
    # -------------------------------
    df_filtered.show(5 , truncate=False)

    # -------------------------------
    # 4. Explain Plan
    # -------------------------------
    print("Execution Plan for Filtered Query:")
    df_filtered.explain()

    spark.stop()

if __name__ == "__main__":
    main()