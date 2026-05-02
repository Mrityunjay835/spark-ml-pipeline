from src.utils.spark_session import create_spark_session

# create a SparkSession
spark = create_spark_session("Olist Parquet Reader")

# Read Parquet data
df = (
    spark.read
    .parquet("data/processed/orders")
)

# Show schema
df.printSchema()


# Show sample data
df.show(5)

spark.stop()