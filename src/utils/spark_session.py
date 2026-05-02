from pyspark.sql import SparkSession

def create_spark_session(app_name: str = "SparkApp") -> SparkSession:
    """
    Create a SparkSession with the given application name.
    - optimized configuration for local development
    """
    spark = (
        SparkSession.builder
        .appName(app_name)
        
        #master for Run Locally
        .master("local[*]")

        # performance optimization 
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "2g")

        # Parquet optimization
        .config("spark.sql.parquet.compression.codec", "snappy")

        .getOrCreate()
    )
    return spark