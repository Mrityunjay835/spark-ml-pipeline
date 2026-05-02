from pyspark.sql.types import (
    StructType, StructField,
    StringType, TimestampType, DoubleType
)

# StructField("",StringType(), True),
# StructField("",TimestampType(), True),
# StructField("",DoubleType(), True),

# -------------------------
# Orders Schema
# -------------------------
c = StringType([
    StructField("order_id", StringType(), True),
    StructField("customer_id", StringType(), True),
    StructField("order_status", StringType(), True),
    StructField("order_purchase_timestamp", TimestampType(), True),
    StructField("order_approved_at", TimestampType(), True),
    StructField("order_delivered_carrier_date", TimestampType(), True),
    StructField("order_delivered_customer_date", TimestampType(), True),
    StructField("order_estimated_delivery_date", TimestampType(), True),
])

# -------------------------
# Customers Schema
# -------------------------
customers_schema = StructType([
    StructField("customer_id",StringType(), True),
    StructField("customer_unique_id",StringType(), True),
    StructField("customer_zip_code_prefix",StringType(), True),
    StructField("customer_city",StringType(), True),
    StructField("customer_state",StringType(), True),
])

# -------------------------
# Order Items Schema
# -------------------------
items_schema = StructType([
    StructField("order_id",StringType(), True),
    StructField("order_item_id",StringType(), True),
    StructField("product_id",StringType(), True),
    StructField("seller_id",StringType(), True),
    StructField("shipping_limit_date",TimestampType(), True),
    StructField("price",DoubleType(), True),
    StructField("freight_value",DoubleType(), True),
])