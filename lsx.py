import polars as pl
import glob,os
from pathlib import Path
from datetime import datetime
import pytz
from datetime import time
from datetime import date
import pandas as pd
import json
#import pyarrow
import numpy as np
#from ftplib import FTP


##MAIN

#Step 1: all files in last directory
files = glob.glob(r'lsxtradesyesterday_*.csv')
# Step 2: Read all CSVs into Polars DataFrames

df_list = []
for f in files:
    try:
        # ignore_errors=True helps skip rows with corrupted quoting
        df_list.append(pl.read_csv(f, infer_schema_length=100,separator=";",decimal_comma=True, ignore_errors=True))
    except Exception as e:
        print(f"Warning: could not read {f}: {e}")

#delete name column from all frames
for index,dlist in enumerate(df_list):
        # Rename orderId to TVTIC if it exists to ensure consistency across files
        if 'orderId' in df_list[index].columns:
            df_list[index] = df_list[index].rename({'orderId': 'TVTIC'})

        if len(df_list[index].columns) == 11:
            df_list[index]=df_list[index].drop('displayName')

# Step 3: Concatenate all data into one DataFrame
print('start concatenation....')
df = pl.concat(df_list, how="diagonal")
print('...end concatenation')
# Step 4: Column conversions

# -- Convert 'tradeTime' and 'publishedTime' to pl.Datetime
# Parse the tradeTime column to datetime and convert to Berlin time
df = df.with_columns(
     pl.col("tradeTime")
     .str.replace("Z", "")
     .str.replace("T"," ")
     .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S%.f", strict=False)
    .dt.replace_time_zone("UTC")
    .dt.convert_time_zone("Europe/Berlin")
    .alias("tradeTime"),

    pl.col("publishedTime")
     .str.replace("Z", "")
     .str.replace("T"," ")
     .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S%.f", strict=False)
    .dt.replace_time_zone("UTC")
    .dt.convert_time_zone("Europe/Berlin")
    .alias("publishedTime")
    
)
# -- Cast 'price' to float and 'size' to int
df = df.with_columns([
    pl.col("price").cast(pl.Float64, strict=False),
    pl.col("size").cast(pl.Int64, strict=False),
])
df = df.drop_nulls(["price", "size", "tradeTime"])

# -- Clean string columns
str_columns = ['isin', 'displayName', 'quotation', 'currency', 'TVTIC', 'mic', 'flags']
for col in str_columns:
    if col in df.columns:
        df = df.with_columns(pl.col(col).cast(pl.Utf8).fill_null("").str.strip_chars())
 
print('Done with structure cleanup')

df_sorted=df.sort('publishedTime')
print('all sorted, now make it unique based on TVTIC...')
df=df_sorted.unique(subset=['TVTIC'],keep='last')

print('Grouping transactions by ISIN and price within 1 second...')
# Consolidate TVTIC, group CANC/AMND transactions to deduct/add amounts.
df = df.with_columns([
    pl.col("tradeTime").dt.truncate("1s").alias("trade_sec")
])

# For cancellations, size is negated to subtract from the group
df = df.with_columns(
    pl.when(pl.col("flags").str.contains("CANC"))
    .then(pl.col("size") * -1)
    .otherwise(pl.col("size"))
    .alias("size")
)

# Merge transactions for the same isin and same price within 1 second
grouped = df.group_by(["isin", "trade_sec", "price"]).agg([
    pl.col("size").sum().alias("total_size"),
    pl.col("tradeTime").first().alias("tradeTime"),
    pl.col("publishedTime").first().alias("publishedTime"),
    pl.col("TVTIC").first().alias("TVTIC"),
    pl.col("flags").first().alias("flags"),
    pl.col("mic").first().alias("mic"),
    pl.col("currency").first().alias("currency"),
    pl.col("quotation").first().alias("quotation")
])

# Remove groups where total amount of shares ended up <= 0 (e.g., fully cancelled)
grouped = grouped.filter(pl.col("total_size") > 0)

print("Filtering for ISINs with >= 10 transactions per day and >= 5000 Euro volume...")
grouped = grouped.with_columns([
    pl.col("trade_sec").dt.truncate("1d").alias("trade_day"),
    (pl.col("total_size") * pl.col("price")).alias("volume")
])

# Find all ISINs with >= 10 transactions per day and a volume of >= 5000 Euro on ANY day in the files
daily_stats = grouped.group_by(["isin", "trade_day"]).agg([
    pl.col("total_size").len().alias("tx_count"),
    pl.col("volume").sum().alias("daily_volume")
])

valid_isins = daily_stats.filter(
    (pl.col("tx_count") >= 10) &
    (pl.col("daily_volume") >= 5000)
).select("isin").unique()

# Filter the aggregated dataset based on valid ISINs
final_df = grouped.join(valid_isins, on="isin", how="inner")

# Drop the temporary grouping columns
final_df = final_df.drop(["trade_sec", "trade_day", "volume"])

print(f"Done. Writing {final_df.shape[0]} rows to parquet.")
final_df.write_parquet('raw_output.parquet')
       