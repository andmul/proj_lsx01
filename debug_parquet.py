import polars as pl

df = pl.read_parquet("consolidated_transactions.parquet")
print(f"Total Rows in Parquet: {df.height}")
print(f"Total Unique ISINs: {df['isin'].n_unique()}")

df = df.with_columns([
    pl.col("tradeTime").dt.hour().alias("hour"),
    pl.col("tradeTime").dt.minute().alias("minute")
])

morning_ticks = df.filter((pl.col("hour") == 7) | ((pl.col("hour") == 8) & (pl.col("minute") <= 45)))
afternoon_ticks = df.filter((pl.col("hour") >= 9) & (pl.col("hour") <= 17))

print(f"Morning Ticks (07:00-08:45): {morning_ticks.height}")
print(f"Morning ISINs: {morning_ticks['isin'].n_unique()}")

print(f"Afternoon Ticks (09:00-17:00): {afternoon_ticks.height}")
print(f"Afternoon ISINs: {afternoon_ticks['isin'].n_unique()}")
