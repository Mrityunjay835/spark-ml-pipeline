# Orders
ORDER_ID = "order_id"
CUSTOMER_ID = "customer_id"
ORDER_PURCHASE_TIMESTAMP = "order_purchase_timestamp"
ORDER_DATE = "order_date"
ORDER_STATUS = "order_status"
TOTAL_ORDERS = "total_orders"
AVG_ORDER_VALUE = "avg_order_value"
MAX_PRICE = "max_price"
PRICE_VARIANCE = "price_variance"
FREIGHT_VALUE ="freight_value"
SHIPPING_LIMIT_DATE ="shipping_limit_date"
ORDER_DELIVERED_CARRIER_DATE = "order_delivered_carrier_date"
ORDER_DELIVERED_CUSTOMER_DATE = "order_delivered_customer_date"
ORDER_APPROVED_AT = "order_approved_at"
ORDER_ESTIMATED_DELIVERY_DATE = "order_estimated_delivery_date"

# Sheller
SELLER_ID="seller_id"

# Customers
CUSTOMER_CITY = "customer_city"
CUSTOMER_STATE = "customer_state"
CUSTOMER_UNIQUE_ID= "customer_unique_id"
CUSTOMER_ZIP_CODE_PREFIX= "customer_zip_code_prefix"

# Items
PRODUCT_ID = "product_id"
PRICE = "price"
ORDER_ITEM_ID="order_item_id"

# other
REFERENCE_DATE = "2018-10-01"

"""
Project-wide constants. Single source of truth for all paths, model params, and config.
"""
import os

# ─── Base Paths ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Raw data paths
RAW_DIR         = os.path.join(DATA_DIR, "raw")
OLIST_DIR       = os.path.join(RAW_DIR, "olist")
RETAILROCKET_DIR = os.path.join(RAW_DIR, "retailrocket")

# Processed / Feature paths
PROCESSED_DIR   = os.path.join(DATA_DIR, "processed")
FEATURES_DIR    = os.path.join(DATA_DIR, "features")
USER_FEATURES_DIR = os.path.join(FEATURES_DIR, "user_features")

# Model artifacts
MODELS_DIR      = os.path.join(BASE_DIR, "models")
PIPELINE_PATH   = os.path.join(MODELS_DIR, "spark_pipeline")
MODEL_PATH      = os.path.join(MODELS_DIR, "churn_model")

# Streaming
STREAMING_INPUT_DIR  = os.path.join(DATA_DIR, "streaming", "input")
STREAMING_CHECKPOINT = os.path.join(DATA_DIR, "streaming", "checkpoint")

# Logs
LOGS_DIR        = os.path.join(BASE_DIR, "logs")
ERROR_LOG       = os.path.join(BASE_DIR, "error.log")

# ─── Olist CSV File Names ──────────────────────────────────────────────────────
OLIST_FILES = {
    "customers":    "olist_customers_dataset.csv",
    "orders":       "olist_orders_dataset.csv",
    "order_items":  "olist_order_items_dataset.csv",
    "payments":     "olist_order_payments_dataset.csv",
    "reviews":      "olist_order_reviews_dataset.csv",
    "products":     "olist_products_dataset.csv",
    "sellers":      "olist_sellers_dataset.csv",
    "geo":          "olist_geolocation_dataset.csv",
    "category":     "product_category_name_translation.csv",
}

# RetailRocket file names
RETAILROCKET_FILES = {
    "events":       "events.csv",
    "category_tree":"category_tree.csv",
    "item_props_1": "item_properties_part1.csv",
    "item_props_2": "item_properties_part2.csv",
}

# ─── Spark Config ──────────────────────────────────────────────────────────────
SPARK_APP_NAME      = "SparkMLProject"
SPARK_MASTER        = "local[*]"               # override with yarn/k8s in prod
SPARK_EXECUTOR_MEM  = "4g"
SPARK_DRIVER_MEM    = "4g"
SPARK_SHUFFLE_PARTS = 8                        # tune to 2x cores in prod

# ─── Feature Engineering ───────────────────────────────────────────────────────
CHURN_DAYS_THRESHOLD    = 90       # no purchase in N days → churned
RECENCY_QUANTILES       = 5
MAX_CATEGORIES          = 50       # StringIndexer cardinality cap

# ─── ML Hyperparameters (GBT defaults; tune via CrossValidator) ────────────────
GBT_MAX_ITER        = 50
GBT_MAX_DEPTH       = 5
GBT_STEP_SIZE       = 0.1
GBT_SUBSAMPLING     = 0.8
GBT_SEED            = 42

RF_NUM_TREES        = 100
RF_MAX_DEPTH        = 8
RF_SEED             = 42

LR_MAX_ITER         = 100
LR_REG_PARAM        = 0.01
LR_ELASTIC_NET      = 0.5

# ─── Train/Test Split ──────────────────────────────────────────────────────────
TRAIN_RATIO         = 0.8
TEST_RATIO          = 0.2
SPLIT_SEED          = 42

# ─── Streaming ─────────────────────────────────────────────────────────────────
STREAM_TRIGGER_INTERVAL = "30 seconds"
STREAM_FORMAT           = "parquet"
STREAM_MAX_FILES        = 5         # maxFilesPerTrigger

# ─── Target Column ─────────────────────────────────────────────────────────────
LABEL_COL       = "churn_label"
FEATURES_COL    = "features"
PREDICTION_COL  = "prediction"
PROB_COL        = "probability"