"""
Standalone script to run PySpark feature engineering.
Called by DVC pipeline to reproduce the feature dataset.
"""
import os
import sys

os.environ["JAVA_HOME"] = "/opt/homebrew/opt/openjdk@17"
os.environ["PATH"]      = "/opt/homebrew/opt/openjdk@17/bin:" + os.environ["PATH"]

from dotenv import load_dotenv
load_dotenv("/Users/azeemkhalipha/mlops-retail-platform/.env")

PROJECT_ROOT  = os.getenv("PROJECT_ROOT")
RAW_DATA_PATH = f"{PROJECT_ROOT}/data/raw/online_retail_II.csv"
FEATURES_PATH = f"{PROJECT_ROOT}/data/features"

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

spark = SparkSession.builder \
    .appName("FeatureEngineering") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# Load and clean
df = spark.read.csv(RAW_DATA_PATH, header=True, inferSchema=True)
df = df.withColumnRenamed("Customer ID", "customer_id")
df = df.filter(F.col("Quantity") > 0)
df = df.filter(F.col("Price") > 0)
df = df.filter(~F.col("Invoice").startswith("C"))
df = df.dropna(subset=["customer_id"])

# Date features
df = df.withColumn("invoice_date", F.to_date(F.col("InvoiceDate")))
df = df.withColumn("revenue", F.round(F.col("Quantity") * F.col("Price"), 2))

# Daily aggregation
daily = df.groupBy("StockCode", "invoice_date").agg(
    F.sum("Quantity").alias("daily_qty"),
    F.sum("revenue").alias("daily_revenue"),
    F.countDistinct("Invoice").alias("daily_orders")
)

# Lag and rolling features
w   = Window.partitionBy("StockCode").orderBy("invoice_date")
w7  = Window.partitionBy("StockCode").orderBy("invoice_date").rowsBetween(-7, -1)
w30 = Window.partitionBy("StockCode").orderBy("invoice_date").rowsBetween(-30, -1)

daily = daily.withColumn("qty_lag_1",          F.lag("daily_qty", 1).over(w))
daily = daily.withColumn("qty_lag_7",          F.lag("daily_qty", 7).over(w))
daily = daily.withColumn("qty_lag_30",         F.lag("daily_qty", 30).over(w))
daily = daily.withColumn("qty_rolling_avg_7",  F.round(F.avg("daily_qty").over(w7), 2))
daily = daily.withColumn("qty_rolling_avg_30", F.round(F.avg("daily_qty").over(w30), 2))
daily = daily.withColumn("qty_rolling_std_7",  F.round(F.stddev("daily_qty").over(w7), 2))
daily = daily.dropna(subset=["qty_lag_30"])

# Customer features
customer_stats = df.groupBy("customer_id").agg(
    F.sum("revenue").alias("customer_total_spend"),
    F.countDistinct("Invoice").alias("customer_total_orders"),
    F.min("invoice_date").alias("customer_first_purchase"),
    F.max("invoice_date").alias("customer_last_purchase")
)

# Save as Parquet
os.makedirs(f"{FEATURES_PATH}/ml_features",      exist_ok=True)
os.makedirs(f"{FEATURES_PATH}/customer_features", exist_ok=True)

daily.write.mode("overwrite").parquet(f"{FEATURES_PATH}/ml_features")
customer_stats.write.mode("overwrite").parquet(f"{FEATURES_PATH}/customer_features")

print(f"Features saved: {daily.count():,} rows, {len(daily.columns)} columns")
spark.stop()
