from src.utils.spark_session import create_spark_session
from src.config.constants import *
from src.config.schema import customers_schema, items_schema, orders_schema


from pyspark.sql.functions import col
from pyspark.sql.functions import broadcast


def main():
    # Create Spark Session
    spark = create_spark_session("Olist Join Data")

    # -------------------------------
    # 1. Load Orders (Processed)
    # -------------------------------

    orders_df = (
        spark.read
        .schema(orders_schema)
        .option("header", True)
        .parquet("data/processed/orders_partitioned")
    )

    # -------------------------------
    # 2. Load Order Item 
    # -------------------------------

    items_df = (
        spark.read
        .schema(items_schema)
        .option("header", True)
        .csv("data/raw/olist/olist_order_items_dataset.csv")
    ).select(
        ORDER_ID,
        PRODUCT_ID,
        PRICE
    )

    
    # -------------------------------
    # 3. Load Customers
    # -------------------------------

    customers_df  = (
        spark.read
        .schema(customers_schema)
        .option("header", True)
        .csv("data/raw/olist/olist_customers_dataset.csv")
    ).select(
        CUSTOMER_ID,
        CUSTOMER_CITY,
        CUSTOMER_STATE
    )
    # -------------------------------
    # 4. Join Orders + Customers
    # -------------------------------
    orders_customers_df = orders_df.join(
        broadcast(customers_df),
        on = CUSTOMER_ID,
        how = "inner"
    )

    # -------------------------------
    # 5. Join with Order Items
    # -------------------------------

    final_df = orders_customers_df.join(
        broadcast(items_df),
        on = ORDER_ID,
        how = "inner"
    )

    # -------------------------------
    # 6. Select Final Column
    # -------------------------------
    final_df = final_df.select(
        ORDER_ID,
        CUSTOMER_ID,
        CUSTOMER_CITY,
        CUSTOMER_STATE,
        PRODUCT_ID,
        PRICE,
        ORDER_DATE
    )
    # -------------------------------
    # 7. Show Sample
    # -------------------------------
    final_df.show(5)

    print("Final row count:", final_df.count())

    # -------------------------------
    # 8. Save Final Dataset
    # -------------------------------
    (
        final_df.write
        .mode("overwrite")
        .partitionBy(ORDER_DATE)
        .parquet("data/processed/final_dataset")
    )


    #-------------------------------
    # 9. Explain the final dataset
    #-------------------------------
    final_df.explain(True)
    print(" Final dataset  saved")
    spark.stop()

if __name__ == "__main__":
    main()
