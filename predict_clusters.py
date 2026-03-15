import polars as pl
try:
    from tslearn.clustering import TimeSeriesKMeans
    from tslearn.utils import to_time_series_dataset
    from tslearn.preprocessing import TimeSeriesScalerMinMax
except ImportError:
    print("Error: The 'tslearn' library is required to perform un-aggregated tick-by-tick clustering.")
    print("Please run: pip install tslearn")
    import sys
    sys.exit(1)

import numpy as np
import os
import warnings

def extract_raw_tick_series(df_input):
    """
    Extracts the un-aggregated, raw tick-by-tick arrays of Price and Cumulative Volume
    for the morning trading session (07:00 to 08:44).
    """
    df = df_input.with_columns([
        pl.col("tradeTime").dt.truncate("1d").alias("trade_day"),
        pl.col("tradeTime").dt.hour().alias("hour"),
        pl.col("tradeTime").dt.minute().alias("minute")
    ])

    # Filter exactly between 07:00 and 08:44
    morning_df = df.filter(
        ((pl.col("hour") == 7) | ((pl.col("hour") == 8) & (pl.col("minute") < 45)))
    )

    # Extract the exact array of raw prices and calculate raw tick volume
    agg = (
        morning_df
        .sort(["isin", "tradeTime"])
        .group_by(["isin", "trade_day"])
        .agg([
            pl.len().alias("total_ticks"),
            pl.col("price").alias("price_series"),
            (pl.col("size") * pl.col("price")).alias("tick_volume_series")
        ])
    )
    return agg


# ==============================================================================
# 1. TRAIN BASELINE MODEL (N=5) ON HISTORICAL DATA
# ==============================================================================
historical_file = "consolidated_transactions.parquet"
print(f"Loading historical training data from {historical_file}...")

if not os.path.exists(historical_file):
    print(f"Error: {historical_file} not found. Cannot compute cluster centroids without baseline data.")
    import sys
    sys.exit(1)

# Extract patterns and normalize vectors
df_hist = pl.scan_parquet(historical_file)
features_df = extract_raw_tick_series(df_hist).collect()

# We only consider assets with enough morning ticks to form a coherent structural pattern
morning_agg = features_df.filter(pl.col("total_ticks") >= 25)

# Build sequence dataset for historical fitting
time_series_data = []
for row in morning_agg.iter_rows(named=True):
    price = np.array(row["price_series"], dtype=float)
    tick_vol = np.array(row["tick_volume_series"], dtype=float)
    cum_vol = np.cumsum(tick_vol)
    ts = np.column_stack((price, cum_vol))
    time_series_data.append(ts)

X_raw = to_time_series_dataset(time_series_data)

# Extract max_length to correctly pad incoming daily files downstream
max_historical_len = X_raw.shape[1]

# Fit Scaler
scaler = TimeSeriesScalerMinMax()
X_hist_scaled = scaler.fit_transform(X_raw)

# Fix zero division (flatlines) while maintaining DTW structural NaNs
for i in range(X_hist_scaled.shape[0]):
    for d in range(X_hist_scaled.shape[2]):
        valid_idx = ~np.isnan(X_hist_scaled[i, :, d])
        if np.any(valid_idx) and np.all(np.isnan(X_hist_scaled[i, valid_idx, d])):
            X_hist_scaled[i, valid_idx, d] = 0.0

print("Fitting Time-Series KMeans (n_clusters=18, metric=dtw)...")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # Optimal N=18 as explicitly stated by the user
    kmeans = TimeSeriesKMeans(
        n_clusters=18,
        metric="dtw",
        max_iter=5,
        random_state=42,
        n_jobs=-1
    )
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
new_agg = extract_raw_tick_series(df_new)
new_agg = new_agg.filter(pl.col("total_ticks") > 10)

if new_agg.shape[0] == 0:
    print("No ISINs in the new file met the >10 ticks criteria.")
else:
    # Convert new raw ticks to DTW padded space
    new_time_series = []
    for row in new_agg.iter_rows(named=True):
        price = np.array(row["price_series"], dtype=float)
        tick_vol = np.array(row["tick_volume_series"], dtype=float)
        cum_vol = np.cumsum(tick_vol)
        ts = np.column_stack((price, cum_vol))
        new_time_series.append(ts)

    # We must construct a padded dataset, but strictly bounded to max_historical_len
    # to fit into the exact dimensional space the DTW KMeans expects.
    X_new_raw = to_time_series_dataset(new_time_series)

    # Truncate or pad manually to match historical max_len
    current_len = X_new_raw.shape[1]
    if current_len > max_historical_len:
        X_new_raw = X_new_raw[:, :max_historical_len, :]
    elif current_len < max_historical_len:
        pad_width = max_historical_len - current_len
        # Pad with NaNs along the time axis to respect DTW boundaries
        X_new_raw = np.pad(X_new_raw, ((0,0), (0, pad_width), (0,0)), mode='constant', constant_values=np.nan)

    # Scale independently using the DTW MinMax scaler
    X_new_scaled = scaler.transform(X_new_raw)

    # Handle perfect flatlines safely
    for i in range(X_new_scaled.shape[0]):
        for d in range(X_new_scaled.shape[2]):
            valid_idx = ~np.isnan(X_new_scaled[i, :, d])
            if np.any(valid_idx) and np.all(np.isnan(X_new_scaled[i, valid_idx, d])):
                X_new_scaled[i, valid_idx, d] = 0.0

    # Predict
    predicted_clusters = kmeans.predict(X_new_scaled)

    # Attach and output
    predictions_df = new_agg.with_columns([
        pl.Series("Predicted_Cluster", predicted_clusters),
        (((pl.col("price_series").list.last() - pl.col("price_series").list.first()) / pl.col("price_series").list.first()) * 100).alias("morning_return_pct")
    ])

    # Sort logically so the user can group their orders by cluster or tick depth
    predictions_df = predictions_df.sort(["Predicted_Cluster", "total_ticks"], descending=[False, True])

    print("\n" + "="*80)
    print("NEW TRADES PREDICTION OUTPUT (N=18 - DTW TIME SERIES CLUSTERING)")
    print("="*80)
    print(f"{'ISIN':<15} | {'Total Ticks':<12} | {'Morning Return (%)':<18} | {'Assigned Cluster':<16}")
    print("-" * 75)

    for row in predictions_df.iter_rows(named=True):
        ret_pct = row['morning_return_pct']
        print(f"{row['isin']:<15} | {row['total_ticks']:<12.0f} | {ret_pct:>17.2f}% | C_{row['Predicted_Cluster']:<14}")

    predictions_df.write_csv("trades_predicted_clusters.csv")
    print("\nSuccessfully saved predictions to trades_predicted_clusters.csv")
