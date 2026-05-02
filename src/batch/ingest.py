from src.utils.spark_session import create_spark_session

# create a SparkSession
spark = create_spark_session("Olist Ingestion")

# Read datasets

df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv("data/raw/olist/olist_orders_dataset.csv")
)

# Show schema
df.printSchema()

# Show sample data
df.show(5, truncate=False)

# Count
print(f"Total records: {df.count()}")
df.explain()

spark.stop()