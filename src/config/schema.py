"""
Explicit StructType schemas for all raw CSVs.
Enforcing schemas at read-time prevents silent type coercion bugs in prod.
"""
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, DoubleType, TimestampType, LongType
)

# ─── Olist Schemas ─────────────────────────────────────────────────────────────

CUSTOMERS_SCHEMA = StructType([
    StructField("customer_id",               StringType(),    False),
    StructField("customer_unique_id",        StringType(),    False),
    StructField("customer_zip_code_prefix",  StringType(),    True),
    StructField("customer_city",             StringType(),    True),
    StructField("customer_state",            StringType(),    True),
])

ORDERS_SCHEMA = StructType([
    StructField("order_id",                  StringType(),    False),
    StructField("customer_id",               StringType(),    False),
    StructField("order_status",              StringType(),    True),
    StructField("order_purchase_timestamp",  TimestampType(), True),
    StructField("order_approved_at",         TimestampType(), True),
    StructField("order_delivered_carrier_date", TimestampType(), True),
    StructField("order_delivered_customer_date", TimestampType(), True),
    StructField("order_estimated_delivery_date", TimestampType(), True),
])

ORDER_ITEMS_SCHEMA = StructType([
    StructField("order_id",           StringType(),  False),
    StructField("order_item_id",      IntegerType(), True),
    StructField("product_id",         StringType(),  True),
    StructField("seller_id",          StringType(),  True),
    StructField("shipping_limit_date",TimestampType(),True),
    StructField("price",              DoubleType(),  True),
    StructField("freight_value",      DoubleType(),  True),
])

PAYMENTS_SCHEMA = StructType([
    StructField("order_id",               StringType(),  False),
    StructField("payment_sequential",     IntegerType(), True),
    StructField("payment_type",           StringType(),  True),
    StructField("payment_installments",   IntegerType(), True),
    StructField("payment_value",          DoubleType(),  True),
])

REVIEWS_SCHEMA = StructType([
    StructField("review_id",                StringType(),    False),
    StructField("order_id",                 StringType(),    False),
    StructField("review_score",             IntegerType(),   True),
    StructField("review_comment_title",     StringType(),    True),
    StructField("review_comment_message",   StringType(),    True),
    StructField("review_creation_date",     TimestampType(), True),
    StructField("review_answer_timestamp",  TimestampType(), True),
])

PRODUCTS_SCHEMA = StructType([
    StructField("product_id",                 StringType(),  False),
    StructField("product_category_name",      StringType(),  True),
    StructField("product_name_lenght",        IntegerType(), True),   # typo is in raw data
    StructField("product_description_lenght", IntegerType(), True),
    StructField("product_photos_qty",         IntegerType(), True),
    StructField("product_weight_g",           DoubleType(),  True),
    StructField("product_length_cm",          DoubleType(),  True),
    StructField("product_height_cm",          DoubleType(),  True),
    StructField("product_width_cm",           DoubleType(),  True),
])

SELLERS_SCHEMA = StructType([
    StructField("seller_id",              StringType(), False),
    StructField("seller_zip_code_prefix", StringType(), True),
    StructField("seller_city",            StringType(), True),
    StructField("seller_state",           StringType(), True),
])

GEO_SCHEMA = StructType([
    StructField("geolocation_zip_code_prefix", StringType(), True),
    StructField("geolocation_lat",             DoubleType(), True),
    StructField("geolocation_lng",             DoubleType(), True),
    StructField("geolocation_city",            StringType(), True),
    StructField("geolocation_state",           StringType(), True),
])

CATEGORY_SCHEMA = StructType([
    StructField("product_category_name",         StringType(), True),
    StructField("product_category_name_english", StringType(), True),
])

# ─── RetailRocket Schemas ──────────────────────────────────────────────────────

EVENTS_SCHEMA = StructType([
    StructField("timestamp",   LongType(),   False),   # Unix ms
    StructField("visitorid",   IntegerType(),False),
    StructField("event",       StringType(), False),   # view / addtocart / transaction
    StructField("itemid",      IntegerType(),True),
    StructField("transactionid",StringType(),True),
])

CATEGORY_TREE_SCHEMA = StructType([
    StructField("categoryid", IntegerType(), False),
    StructField("parentid",   IntegerType(), True),
])

ITEM_PROPERTIES_SCHEMA = StructType([
    StructField("timestamp",  LongType(),   False),
    StructField("itemid",     IntegerType(),False),
    StructField("property",   StringType(), True),
    StructField("value",      StringType(), True),
])

# ─── Feature Store Schema (written to parquet) ─────────────────────────────────
# Reference only — actual schema enforced via Spark read
USER_FEATURES_SCHEMA = StructType([
    StructField("customer_unique_id",        StringType(),  False),
    StructField("total_orders",              IntegerType(), True),
    StructField("total_spend",               DoubleType(),  True),
    StructField("avg_order_value",           DoubleType(),  True),
    StructField("avg_review_score",          DoubleType(),  True),
    StructField("days_since_last_order",     IntegerType(), True),
    StructField("num_payment_types",         IntegerType(), True),
    StructField("avg_installments",          DoubleType(),  True),
    StructField("total_freight",             DoubleType(),  True),
    StructField("num_product_categories",    IntegerType(), True),
    StructField("customer_state",            StringType(),  True),
    StructField("churn_label",               IntegerType(), True),
])

# ─── Streaming event schema (for structured streaming) ────────────────────────
STREAMING_EVENT_SCHEMA = StructType([
    StructField("customer_unique_id",    StringType(),  False),
    StructField("event_timestamp",       TimestampType(),True),
    StructField("event_type",            StringType(),  True),
    StructField("item_id",               IntegerType(), True),
    StructField("price",                 DoubleType(),  True),
])