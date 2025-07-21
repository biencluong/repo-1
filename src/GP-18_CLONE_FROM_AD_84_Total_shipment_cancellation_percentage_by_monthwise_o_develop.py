# PySpark script: Databricks month-wise shipment aggregation with cancellation analytics and client name
# Purpose: Calculate month-wise total and cancelled shipments per client, plus cancellation rate, with client names
# Author: Bien Luong
# Date: 2025-07-21
# Description: This script reads 'purgo_playground.shipments' and 'purgo_playground.clients', 
# aggregates shipments by client, year, and month, computes total shipments, cancelled shipments,
# and cancellation percentage (for cancellation_flag='Yes'), joins client name, handles schema consistency,
# data quality, and null scenarios per upstream requirements.

# ----------------------------------------------------------
# Imports (pyspark.sql.functions/types, all pip: pyspark)
# ----------------------------------------------------------
import pyspark.sql.functions as F                  
from pyspark.sql.types import (                    
    StringType, IntegerType, LongType, DoubleType, DateType
)

# ------------------------------------------------------------------
# Section: Read base tables from Unity Catalog (no explicit schema)
# ------------------------------------------------------------------
shipments_df = spark.table("purgo_playground.shipments")
clients_df = spark.table("purgo_playground.clients")

# --------------------------------------------------------------
# Section: Helper for shipment date validation/error detection
# --------------------------------------------------------------
def assert_valid_shipment_dates(shipments_df):
    """
    Checks for invalid shipment_date formats in the shipments DataFrame.
    Args:
        shipments_df (pyspark.sql.DataFrame): the input shipments DataFrame
    Raises:
        Exception with diagnostic shipment_id(s) on any invalid date parse.
    """
    # Try casting shipment_date to date and find rows where this fails
    bad_rows = (
        shipments_df
        .withColumn("shipment_date_cast", F.col("shipment_date").cast(DateType()))
        .filter(F.col("shipment_date").isNotNull() & F.col("shipment_date_cast").isNull())
        .select("shipment_id", "shipment_date")
    )
    invalid = bad_rows.collect()
    if invalid:
        msg = "; ".join(f"shipment_id {r['shipment_id']} (shipment_date={r['shipment_date']})" for r in invalid)
        raise Exception(f"Invalid date format in shipment_date for {msg}")

# -------------------------------------------------------------------
# Section: Main aggregation logic (all transformation in one function)
# -------------------------------------------------------------------
def compute_monthly_cancellation_analytics(shipments_df, clients_df):
    """
    Aggregates shipments month-wise per client with cancellation metrics and client name.
    Args:
        shipments_df (pyspark.sql.DataFrame): shipments source DataFrame
        clients_df (pyspark.sql.DataFrame): clients source DataFrame
    Returns:
        pyspark.sql.DataFrame: Output DataFrame with columns:
          client_name (StringType, nullable), year (IntegerType, not null),
          month (IntegerType, not null), total_shipments (LongType, not null),
          cancelled_shipments (LongType, not null), cancellation_percentage (DoubleType, not null)
    """
    # Validate shipment_date column and error on any invalid formats
    assert_valid_shipment_dates(shipments_df)

    # Filter only shipments with valid date (should be all after above check)
    clean_shipments_df = shipments_df.withColumn(
        "shipment_date", F.col("shipment_date").cast(DateType())
    )

    # Extract year and month (mandatory), compute aggregates
    agg_df = (
        clean_shipments_df
        .withColumn("year", F.year("shipment_date").cast(IntegerType()))
        .withColumn("month", F.month("shipment_date").cast(IntegerType()))
        .groupBy("client_id", "year", "month")
        .agg(
            F.count(F.lit(1)).cast(LongType()).alias("total_shipments"),
            F.sum(
                F.when(F.col("cancellation_flag") == F.lit("Yes"), F.lit(1)).otherwise(F.lit(0))
            ).cast(LongType()).alias("cancelled_shipments")
        )
        .withColumn(
            "cancellation_percentage",
            F.when(
                F.col("total_shipments") > 0,
                F.round(F.col("cancelled_shipments") * F.lit(100.0) / F.col("total_shipments"), 1)
            ).otherwise(F.lit(0.0))
            .cast(DoubleType())
        )
    )

    # Left join with clients for client_name, ensuring blank if client_id not matched
    result_df = (
        agg_df
        .join(clients_df.select("client_id", "client_name"), on="client_id", how="left")
        .select(
            F.col("client_name").cast(StringType()),
            F.col("year").cast(IntegerType()),
            F.col("month").cast(IntegerType()),
            F.col("total_shipments").cast(LongType()),
            F.col("cancelled_shipments").cast(LongType()),
            F.col("cancellation_percentage").cast(DoubleType())
        )
    )
    return result_df

# -------------------------------------------------------------------
# Section: Compute final analytics DataFrame and display the result
# -------------------------------------------------------------------
analytics_df = compute_monthly_cancellation_analytics(shipments_df, clients_df)

# Show the final output DataFrame (for Databricks notebooks use .display; here .show for script context)
analytics_df.show(truncate=False)
# End of script
