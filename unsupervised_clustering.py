import polars as pl
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import warnings
import sys

file_path = "consolidated_transactions.parquet"
if not os.path.exists(file_path):
    print(f"Error: {file_path} not found. Please run process_all_lsx.py first.")
    sys.exit(1)

print("Loading and querying dataset via Polars LazyFrame...")

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

# Load parquet for feature extraction
df_base = pl.scan_parquet(file_path)

# Extract identical baseline features used in predict_clusters.py
df = df_base.with_columns([
    pl.col("tradeTime").dt.truncate("1d").alias("trade_day"),
    pl.col("tradeTime").dt.hour().alias("hour")
])

# ==============================================================================
# 1. EARLY MORNING WINDOW (07:00:00 to 08:44:59)
# ==============================================================================
# Aggregate early morning features per ISIN per Day
morning_agg = extract_pattern_features(df_base)

# Filter all stocks with at least 25 trades between 7 am and 8:45 am
morning_agg = morning_agg.filter(pl.col("total_ticks") >= 25)

# For evaluating the target pct_change later, we need the final 08:45 price which is 'price_6' in our pattern df
morning_agg = morning_agg.rename({"price_6": "price_845"})

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
# We cluster based on the 20-dimensional time-series shape and trajectory patterns
features = [f"ret_{b}" for b in range(1, 7)] + [f"vol_pct_{b}" for b in range(7)] + [f"tick_pct_{b}" for b in range(7)]
X_raw = final_df.select(features).to_numpy()

# Standardize the features so KMeans evaluates pattern variances equally
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

print("\n" + "="*80)
print("UNSUPERVISED K-MEANS CLUSTERING ANALYSIS (N = 3 to 50)")
print("Based on Morning Trajectory Patterns (15-min buckets)")
print("="*80)

# Track the absolute best cluster structure discovered during the full loop
best_global_advancer_ratio = -1
best_global_cluster_df = None
best_n_val = 0

# Suppress sklearn memory leak warnings on Windows
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    for n_clusters in range(3, 51):
        # Fit the unsupervised model
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
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
                (pl.col("pct_change_max") > 0).cast(pl.Int32).alias("is_advancer")
            ])
            .group_by("cluster")
            .agg([
                pl.len().alias("count"),
                pl.col("pct_change_max").mean().alias("avg_max_pct_change"),
                (pl.col("is_advancer").sum() / pl.len() * 100).alias("advancer_ratio_pct"),

                # Show the cluster's average underlying profile
                pl.col("total_volume").mean().alias("avg_volume"),
                pl.col("total_ticks").mean().alias("avg_ticks"),
                pl.col("ret_6").mean().alias("avg_morning_return")
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
            prof_ret = f"{(row['avg_morning_return'] * 100):.2f}%"
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
