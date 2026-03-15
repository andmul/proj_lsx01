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
import sys

file_path = "consolidated_transactions.parquet"
if not os.path.exists(file_path):
    print(f"Error: {file_path} not found. Please run process_all_lsx.py first.")
    sys.exit(1)

print("Loading and querying dataset via Polars LazyFrame...")

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
            pl.col("price").last().alias("price_845"), # Save the very last price before 08:45 for target calcs
            pl.col("price").alias("price_series"),
            (pl.col("size") * pl.col("price")).alias("tick_volume_series"),
            (pl.col("size") * pl.col("price")).sum().alias("total_volume")
        ])
    )
    return agg

# Load parquet for feature extraction
df_base = pl.scan_parquet(file_path)

# Setup the basic day/hour tags for the market window join later
df = df_base.with_columns([
    pl.col("tradeTime").dt.truncate("1d").alias("trade_day"),
    pl.col("tradeTime").dt.hour().alias("hour")
])

# ==============================================================================
# 1. EARLY MORNING WINDOW (07:00:00 to 08:44:59) - RAW TICK EXTRACTION
# ==============================================================================
morning_agg = extract_raw_tick_series(df_base)

# Filter all stocks with at least 25 raw ticks
morning_agg = morning_agg.filter(pl.col("total_ticks") >= 25)

# ==============================================================================
# 2. REGULAR MARKET WINDOW (09:00:00 to 17:00:00)
# ==============================================================================
market_df = df.filter(
    (pl.col("hour") >= 9) & (pl.col("hour") < 17)
)

# Get the highest price and exact time during regular market hours per ISIN per Day
market_highs = (
    market_df
    .sort(["isin", "trade_day", "price"], descending=[False, False, True])
    .group_by(["isin", "trade_day"])
    .agg([
        pl.col("price").first().alias("day_high"),
        pl.col("tradeTime").first().alias("time_of_high")
    ])
)

# Get the lowest price and exact time during regular market hours per ISIN per Day
market_lows = (
    market_df
    .sort(["isin", "trade_day", "price"], descending=[False, False, False])
    .group_by(["isin", "trade_day"])
    .agg([
        pl.col("price").first().alias("day_low"),
        pl.col("tradeTime").first().alias("time_of_low")
    ])
)

market_agg = market_highs.join(market_lows, on=["isin", "trade_day"], how="inner")

# ==============================================================================
# 3. MERGE AND CALCULATE METRICS
# ==============================================================================
print("Executing complex time-window aggregations...")
final_df = (
    morning_agg.join(market_agg, on=["isin", "trade_day"], how="inner")
    .with_columns([
        # Average percentage change of the Highest price between 9 and 5 compared to the 8:45 price
        (((pl.col("day_high") - pl.col("price_845")) / pl.col("price_845")) * 100).alias("pct_change_max")
    ])
    .drop_nulls()
    .collect()
)

total_obs = final_df.shape[0]
print(f"Data prepared! Extracted {total_obs} daily stock instances meeting the >= 25 early trades criteria.")

if total_obs == 0:
    print("Not enough data to perform clustering. Exiting.")
    sys.exit(1)

# Prepare features for clustering
# Unroll the extracted Polars lists into numpy arrays
time_series_data = []
print("Structuring variable-length raw tick trajectories...")
for row in final_df.iter_rows(named=True):
    # Zip the raw price series and the cumulative tick volume to represent the evolving shape
    price = np.array(row["price_series"], dtype=float)
    tick_vol = np.array(row["tick_volume_series"], dtype=float)
    cum_vol = np.cumsum(tick_vol)

    # 2D array [number_of_ticks, features=2]
    ts = np.column_stack((price, cum_vol))
    time_series_data.append(ts)

# Pad to the maximum sequence length to create a strictly typed array for distance calcs
X_raw = to_time_series_dataset(time_series_data)

# Scale them individually per sequence! (So €5 flatlines aren't treated as distant from €100 flatlines)
# DTW will evaluate purely on the local shape of the curve
scaler = TimeSeriesScalerMinMax()
X_scaled = scaler.fit_transform(X_raw)

# Note: TimeSeriesScalerMinMax leaves the tslearn padding (NaNs) intact.
# For DTW, we leave the NaNs alone! `tslearn`'s DTW natively ignores trailing NaNs in distance calculations
# so it mathematically computes the actual variable length trajectories perfectly.

# Fix the flatline scaling issue (max == min) for valid data points without touching the structural NaNs
for i in range(X_scaled.shape[0]):
    for d in range(X_scaled.shape[2]):
        valid_idx = ~np.isnan(X_scaled[i, :, d])
        if np.any(valid_idx) and np.all(np.isnan(X_scaled[i, valid_idx, d])):
            # It was a flatline that became NaN during division.
            # We set the valid sequence length to 0.0 (the baseline)
            X_scaled[i, valid_idx, d] = 0.0

print(f"Dataset compiled. Shape: [instances: {X_scaled.shape[0]}, max_ticks: {X_scaled.shape[1]}, dims: {X_scaled.shape[2]}]")

print("\n" + "="*80)
print("UNSUPERVISED TIME SERIES K-MEANS CLUSTERING ANALYSIS (N = 3 to 50)")
print("Based on Un-aggregated Raw Tick-by-Tick Sequences (DTW)")
print("="*80)

# Track the absolute best cluster structure discovered during the full loop
best_global_advancer_ratio = -1
best_global_cluster_df = None
best_n_val = 0

# Suppress warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    # User requested 3 to 50 explicitly
    for n_clusters in range(3, 51):
        # Fit the unsupervised time-series model using DTW
        kmeans = TimeSeriesKMeans(
            n_clusters=n_clusters,
            metric="dtw",
            max_iter=5,
            random_state=42,
            n_jobs=-1  # Parallelize to handle the DTW scaling
        )
        cluster_labels = kmeans.fit_predict(X_scaled)

        # Attach cluster labels back to the dataframe
        clustered_df = final_df.with_columns([
            pl.Series("cluster", cluster_labels)
        ])

        # Calculate stats per cluster
        cluster_stats = (
            clustered_df
            .with_columns([
                # Advancer = positive change, Decliner = negative or zero
                (pl.col("pct_change_max") > 0).cast(pl.Int32).alias("is_advancer"),
                # Reconstruct morning return from first and last price in the series array
                (((pl.col("price_series").list.last() - pl.col("price_series").list.first()) / pl.col("price_series").list.first()) * 100).alias("morning_return_pct")
            ])
            .group_by("cluster")
            .agg([
                pl.len().alias("count"),
                pl.col("pct_change_max").mean().alias("avg_max_pct_change"),
                (pl.col("is_advancer").sum() / pl.len() * 100).alias("advancer_ratio_pct"),

                # Show the cluster's average underlying profile
                pl.col("total_volume").mean().alias("avg_volume"),
                pl.col("total_ticks").mean().alias("avg_ticks"),
                pl.col("morning_return_pct").mean().alias("avg_morning_return")
            ])
            .sort("advancer_ratio_pct", descending=True)
        )

        print(f"\n--- N_CLUSTERS = {n_clusters} ---")

        # Identify patterns
        print(f"{'Cluster':<8} | {'Total Stocks':<13} | {'Avg Max Spike %':<16} | {'Advancer Ratio %':<17} | Profile (Volume / Ticks / Morning Ret%)")
        print("-" * 115)

        for row in cluster_stats.iter_rows(named=True):
            # Formatting
            c_id = row['cluster']
            cnt = row['count']
            avg_spike = f"{row['avg_max_pct_change']:.2f}%"
            adv_ratio = f"{row['advancer_ratio_pct']:.2f}%"

            # Profile string
            prof_vol = f"€{row['avg_volume']/1000:.0f}k" if row['avg_volume'] > 1000 else f"€{row['avg_volume']:.0f}"
            prof_ticks = f"{row['avg_ticks']:.0f}t"
            prof_ret = f"{row['avg_morning_return']:.2f}%"
            profile = f"{prof_vol} / {prof_ticks} / {prof_ret}"

            # Highlight extreme edges
            marker = ""
            if row['advancer_ratio_pct'] >= 65:
                marker = "<-- HIGH ADVANCER BIAS"
            elif row['advancer_ratio_pct'] <= 35:
                marker = "<-- HIGH DECLINER BIAS"

            print(f"C_{c_id:<6} | {cnt:<13} | {avg_spike:<16} | {adv_ratio:<17} | {profile} {marker}")

            # Track the absolute best cluster cohort (requiring at least 5 instances to avoid 100% win-rates on a single stock)
            if cnt >= 5 and row['advancer_ratio_pct'] > best_global_advancer_ratio:
                best_global_advancer_ratio = row['advancer_ratio_pct']
                best_n_val = n_clusters

                # Isolate the raw underlying instances that fell into this specific top cluster
                best_global_cluster_df = clustered_df.filter(pl.col("cluster") == c_id)

print("\n" + "="*80)
print(f"WINNER: N={best_n_val} | Highest Advancer Ratio: {best_global_advancer_ratio:.2f}%")
print("="*80)

if best_global_cluster_df is not None:
    export_df = best_global_cluster_df.select([
        "isin",
        "trade_day",
        "day_low",
        "time_of_low",
        "day_high",
        "time_of_high",
        "pct_change_max"
    ])

    out_file = "best_advancer_cluster.csv"
    export_df.write_csv(out_file)
    print(f"Exported the underlying {export_df.shape[0]} trades to {out_file}.")
