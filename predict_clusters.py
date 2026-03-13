import polars as pl
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import warnings

# ==============================================================================
# 1. TRAIN BASELINE MODEL (N=18) ON HISTORICAL DATA
# ==============================================================================
historical_file = "consolidated_transactions.parquet"
print(f"Loading historical training data from {historical_file}...")

if not os.path.exists(historical_file):
    print(f"Error: {historical_file} not found. Cannot compute cluster centroids without baseline data.")
    import sys
    sys.exit(1)

# Extract identical baseline features used in the original unsupervised script
df_hist = (
    pl.scan_parquet(historical_file)
    .with_columns([
        pl.col("tradeTime").dt.truncate("1d").alias("trade_day"),
        pl.col("tradeTime").dt.hour().alias("hour"),
        pl.col("tradeTime").dt.minute().alias("minute"),
        (pl.col("size") * pl.col("price")).alias("volume")
    ])
)

morning_hist = df_hist.filter(
    ((pl.col("hour") == 7) | ((pl.col("hour") == 8) & (pl.col("minute") < 45)))
)

morning_agg = (
    morning_hist
    .sort(["isin", "tradeTime"])
    .group_by(["isin", "trade_day"])
    .agg([
        pl.len().alias("tick_count"),
        pl.col("volume").sum().alias("price_volume"),
        pl.col("price").last().alias("price_845")
    ])
    .filter(pl.col("tick_count") >= 25)
    .drop_nulls()
    .collect()
)

# Standardize and train
features = ["price_volume", "tick_count", "price_845"]
X_hist_raw = morning_agg.select(features).to_numpy()

scaler = StandardScaler()
X_hist_scaled = scaler.fit_transform(X_hist_raw)

print("Fitting KMeans (n_clusters=18)...")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    kmeans = KMeans(n_clusters=18, random_state=42, n_init='auto')
    kmeans.fit(X_hist_scaled)

# ==============================================================================
# 2. LOAD AND PROCESS NEW TRADES (trades.csv / trade.csv)
# ==============================================================================
if os.path.exists("trade.csv"):
    new_trades_file = "trade.csv"
elif os.path.exists("trades.csv"):
    new_trades_file = "trades.csv"
else:
    print("\nWarning: trade.csv not found. Waiting for user to provide the file.")
    import sys
    sys.exit(1)

print(f"\nProcessing new trades from {new_trades_file}...")

# Read the CSV as simply as possible as per user request
df_new = pl.read_csv(new_trades_file, separator=";")

# Rename orderId if present
if 'orderId' in df_new.columns:
    df_new = df_new.rename({'orderId': 'TVTIC'})

# Convert timestamps and numbers
df_new = df_new.with_columns(
    pl.col("tradeTime")
    .str.replace("Z", "")
    .str.replace("T"," ")
    .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S%.f", strict=False)
    .dt.replace_time_zone("UTC")
    .dt.convert_time_zone("Europe/Berlin")
    .alias("tradeTime")
)

# Cast price and size dynamically catching pure floats vs strings depending on read_csv inference
if df_new["price"].dtype == pl.Utf8 or df_new["price"].dtype == pl.String:
    df_new = df_new.with_columns(pl.col("price").str.replace(",", "."))

df_new = df_new.with_columns([
    pl.col("price").cast(pl.Float64, strict=False),
    pl.col("size").cast(pl.Int64, strict=False)
]).drop_nulls(["price", "size", "tradeTime"])

# Extract the morning time boundary natively
df_new = df_new.with_columns([
    pl.col("tradeTime").dt.hour().alias("hour"),
    pl.col("tradeTime").dt.minute().alias("minute"),
    (pl.col("size") * pl.col("price")).alias("volume")
])

# Filter explicitly between 07:00 and 08:45
df_new = df_new.filter(
    ((pl.col("hour") == 7) | ((pl.col("hour") == 8) & (pl.col("minute") < 45)))
)

# ==============================================================================
# 3. GENERATE PREDICTIONS
# ==============================================================================
# "consider only isins with >10 ticks within the 7-8:45 time window"
new_agg = (
    df_new
    .sort(["isin", "tradeTime"])
    .group_by("isin")
    .agg([
        pl.len().alias("tick_count"),
        pl.col("volume").sum().alias("price_volume"),
        pl.col("price").last().alias("price_845")
    ])
    .filter(pl.col("tick_count") > 10)
    .drop_nulls()
)

if new_agg.shape[0] == 0:
    print("No ISINs in the new file met the >10 ticks criteria.")
else:
    # Scale using the historical model's exact spatial standard deviations
    X_new_raw = new_agg.select(features).to_numpy()
    X_new_scaled = scaler.transform(X_new_raw)

    # Predict
    predicted_clusters = kmeans.predict(X_new_scaled)

    # Attach and output
    predictions_df = new_agg.with_columns([
        pl.Series("Predicted_Cluster", predicted_clusters)
    ])

    # Sort logically so the user can group their orders by cluster or tick depth
    predictions_df = predictions_df.sort(["Predicted_Cluster", "tick_count"], descending=[False, True])

    print("\n" + "="*60)
    print("NEW TRADES PREDICTION OUTPUT (N=18)")
    print("="*60)
    print(f"{'ISIN':<15} | {'Tick Count':<12} | {'Assigned Cluster':<16}")
    print("-" * 50)

    for row in predictions_df.iter_rows(named=True):
        print(f"{row['isin']:<15} | {row['tick_count']:<12} | C_{row['Predicted_Cluster']:<14}")

    predictions_df.write_csv("trades_predicted_clusters.csv")
    print("\nSuccessfully saved predictions to trades_predicted_clusters.csv")
