from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window

def create_spark_session():
    return (SparkSession.builder
        .appName("WindTurbineAnomalyDetection")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.driver.memory", "4g")
        .getOrCreate())

def ingest_scada(spark, path: str):
    df = (spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(path))
    
    # Standardise column names
    df = df.toDF(*[c.strip().lower().replace(" ", "_").replace("/", "_") for c in df.columns])
    
    # Parse timestamp
    df = df.withColumn("timestamp", F.to_timestamp("date_time", "dd MM yyyy HH:mm"))
    
    return df

def feature_engineering(df):
    window_1h = Window.orderBy("timestamp").rowsBetween(-6, 0)   # 10-min intervals → 6 = 1hr
    window_24h = Window.orderBy("timestamp").rowsBetween(-144, 0)

    df = (df
        # Rolling stats
        .withColumn("power_rolling_mean_1h",  F.avg("lv_activepower_kw").over(window_1h))
        .withColumn("power_rolling_std_1h",   F.stddev("lv_activepower_kw").over(window_1h))
        .withColumn("wind_rolling_mean_24h",  F.avg("wind_speed_m_s").over(window_24h))
        
        # Derived features
        .withColumn("power_deviation",
            F.col("lv_activepower_kw") - F.col("theoretical_power_curve_kwh"))
        .withColumn("capacity_factor",
            F.col("lv_activepower_kw") / F.lit(3600.0))  # adjust to rated capacity
        
        # Time features
        .withColumn("hour",        F.hour("timestamp"))
        .withColumn("day_of_week", F.dayofweek("timestamp"))
        .withColumn("month",       F.month("timestamp"))
        
        # Lag features (detect sudden changes)
        .withColumn("power_lag_1",  F.lag("lv_activepower_kw", 1).over(Window.orderBy("timestamp")))
        .withColumn("power_delta",  F.col("lv_activepower_kw") - F.col("power_lag_1"))
        
        .dropna()
    )
    return df

def label_anomalies(df):
    """
    Rule-based labels for ground truth (no labels in raw SCADA).
    Use domain knowledge:
      - Power output << theoretical given wind speed = fault
      - Negative power = fault  
      - Power > rated at low wind = sensor fault
    """
    df = df.withColumn("anomaly", 
        F.when(
            (F.col("power_deviation") < -500) |   # >500kW below theoretical
            (F.col("lv_activepower_kw") < -10) |  # negative power
            (F.col("power_delta").cast("double") < -1000),  # sudden drop
            F.lit(1)
        ).otherwise(F.lit(0))
    )
    return df