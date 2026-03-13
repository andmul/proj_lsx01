import polars as pl
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import warnings

# ==============================================================================
# 1. TRAIN BASELINE MODEL (N=30) ON HISTORICAL DATA
# ==============================================================================
historical_file = "consolidated_transactions.parquet"
print(f"Loading historical training data from {historical_file}...")

if not os.path.exists(historical_file):
    print(f"Error: {historical_file} not found. Cannot compute cluster centroids without baseline data.")
    import sys
    sys.exit(1)

def extract_pattern_features(df_input):
    """
    Extracts time-series shape and trajectory patterns across 15-minute buckets
    for the morning trading session (07:00 to 08:44).
    """
    df = df_input.with_columns([
        pl.col("tradeTime").dt.truncate("1d").alias("trade_day"),
        (((pl.col("tradeTime").dt.hour() - 7) * 60 + pl.col("tradeTime").dt.minute()) // 15).cast(pl.Int32).alias("bucket")
    ])

    # Filter exactly between 07:00 and 08:44 (buckets 0 to 6)
    df = df.filter((pl.col("bucket") >= 0) & (pl.col("bucket") <= 6))

    # Base aggregation per bucket
    agg = df.group_by(["isin", "trade_day", "bucket"]).agg([
        pl.col("price").last().alias("price"),
        (pl.col("size") * pl.col("price")).sum().alias("volume"),
        pl.len().alias("tick_count")
    ])

    # Iterative pivoting safely handling completely empty time buckets
    base_frame = agg.select(["isin", "trade_day"]).unique()
    for b in range(7):
        b_df = agg.filter(pl.col("bucket") == b).select([
            "isin", "trade_day",
            pl.col("price").alias(f"price_{b}"),
            pl.col("volume").alias(f"volume_{b}"),
            pl.col("tick_count").alias(f"tick_count_{b}")
        ])
        base_frame = base_frame.join(b_df, on=["isin", "trade_day"], how="left")

    pivoted = base_frame

    # Forward fill prices (carry forward last known price if no trades in a 15min block)
    for b in range(1, 7):
        pivoted = pivoted.with_columns([
            pl.coalesce([f"price_{b}", f"price_{b-1}"]).alias(f"price_{b}")
        ])

    # Backward fill to backstop missing early buckets
    for b in range(5, -1, -1):
        pivoted = pivoted.with_columns([
            pl.coalesce([f"price_{b}", f"price_{b+1}"]).alias(f"price_{b}")
        ])

    pivoted = pivoted.fill_null(0.0)

    # Filter constraints to ensure healthy denominators
    pivoted = pivoted.filter((pl.col("price_0") > 0))

    # Compute normalized pattern features:
    # 1. Price trajectory (% returns vs 07:00 bucket)
    returns = [((pl.col(f"price_{b}") - pl.col("price_0")) / pl.col("price_0")).alias(f"ret_{b}") for b in range(1, 7)]

    # 2. Volume shape (% of total morning volume in each 15-minute chunk)
    total_vol = pl.sum_horizontal([f"volume_{b}" for b in range(7)])
    vol_pct = [(pl.col(f"volume_{b}") / total_vol).fill_nan(0.0).alias(f"vol_pct_{b}") for b in range(7)]

    # 3. Tick shape (% of total morning ticks in each 15-minute chunk)
    total_ticks = pl.sum_horizontal([f"tick_count_{b}" for b in range(7)])
    tick_pct = [(pl.col(f"tick_count_{b}") / total_ticks).fill_nan(0.0).alias(f"tick_pct_{b}") for b in range(7)]

    final_df = pivoted.with_columns(returns + vol_pct + tick_pct + [total_vol.alias("total_volume"), total_ticks.alias("total_ticks")])
    return final_df

# ==============================================================================
# 1. TRAIN BASELINE MODEL (N=18) ON HISTORICAL DATA
# ==============================================================================
historical_file = "consolidated_transactions.parquet"
print(f"Loading historical training data from {historical_file}...")

if not os.path.exists(historical_file):
    print(f"Error: {historical_file} not found. Cannot compute cluster centroids without baseline data.")
    import sys
    sys.exit(1)

# Extract patterns and normalize vectors
df_hist = pl.scan_parquet(historical_file)
features_df = extract_pattern_features(df_hist).collect()

# We only consider assets with enough morning ticks to form a coherent structural pattern
morning_agg = features_df.filter(pl.col("total_ticks") >= 25)

# 20 dimensions: 6 returns + 7 volume % + 7 tick %
features = [f"ret_{b}" for b in range(1, 7)] + [f"vol_pct_{b}" for b in range(7)] + [f"tick_pct_{b}" for b in range(7)]

X_hist_raw = morning_agg.select(features).to_numpy()

scaler = StandardScaler()
X_hist_scaled = scaler.fit_transform(X_hist_raw)

print("Fitting KMeans (n_clusters=30)...")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # Optimal N=30 from the unsupervised_clustering.py evaluation
    kmeans = KMeans(n_clusters=30, random_state=42, n_init='auto')
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

# ==============================================================================
# 3. GENERATE PREDICTIONS
# ==============================================================================
# "consider only isins with >10 ticks within the 7-8:45 time window"
new_agg = extract_pattern_features(df_new)
new_agg = new_agg.filter(pl.col("total_ticks") > 10)

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
    predictions_df = predictions_df.sort(["Predicted_Cluster", "total_ticks"], descending=[False, True])

    print("\n" + "="*80)
    print("NEW TRADES PREDICTION OUTPUT (N=30 - SHAPE CLUSTERING)")
    print("="*80)
    print(f"{'ISIN':<15} | {'Total Ticks':<12} | {'Morning Return (%)':<18} | {'Assigned Cluster':<16}")
    print("-" * 75)

    for row in predictions_df.iter_rows(named=True):
        ret_pct = row['ret_6'] * 100
        print(f"{row['isin']:<15} | {row['total_ticks']:<12.0f} | {ret_pct:>17.2f}% | C_{row['Predicted_Cluster']:<14}")

    predictions_df.write_csv("trades_predicted_clusters.csv")
    print("\nSuccessfully saved predictions to trades_predicted_clusters.csv")
