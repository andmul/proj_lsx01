import polars as pl
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import warnings

def main():
    file_path = "consolidated_transactions.parquet"
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found. Please run process_all_lsx.py first.")
        return

    print("Loading and querying dataset via Polars LazyFrame...")

    # Load and extract necessary time features
    df = (
        pl.scan_parquet(file_path)
        .with_columns([
            pl.col("tradeTime").dt.truncate("1d").alias("trade_day"),
            pl.col("tradeTime").dt.hour().alias("hour"),
            pl.col("tradeTime").dt.minute().alias("minute"),
            (pl.col("size") * pl.col("price")).alias("volume")
        ])
    )

    # ==============================================================================
    # 1. EARLY MORNING WINDOW (07:00:00 to 08:44:59)
    # ==============================================================================
    # Filter trades strictly between 7am and 8:45am
    morning_df = df.filter(
        ((pl.col("hour") == 7) | ((pl.col("hour") == 8) & (pl.col("minute") < 45)))
    )

    # Aggregate early morning features per ISIN per Day
    morning_agg = (
        morning_df
        .sort(["isin", "tradeTime"])
        .group_by(["isin", "trade_day"])
        .agg([
            pl.len().alias("tick_count"),
            pl.col("volume").sum().alias("price_volume"),
            pl.col("price").last().alias("price_845") # The final price hit exactly before 8:45
        ])
        # Filter all stocks with at least 25 trades between 7 am and 8:45 am
        .filter(pl.col("tick_count") >= 25)
    )

    # ==============================================================================
    # 2. REGULAR MARKET WINDOW (09:00:00 to 17:00:00)
    # ==============================================================================
    market_df = df.filter(
        (pl.col("hour") >= 9) & (pl.col("hour") < 17)
    )

    # Get the highest price during regular market hours per ISIN per Day
    market_agg = (
        market_df
        .group_by(["isin", "trade_day"])
        .agg([
            pl.col("price").max().alias("max_price_9_17")
        ])
    )

    # ==============================================================================
    # 3. MERGE AND CALCULATE METRICS
    # ==============================================================================
    print("Executing complex time-window aggregations...")
    final_df = (
        morning_agg.join(market_agg, on=["isin", "trade_day"], how="inner")
        .with_columns([
            # Average percentage change of the Highest price between 9 and 5 compared to the 8:45 price
            (((pl.col("max_price_9_17") - pl.col("price_845")) / pl.col("price_845")) * 100).alias("pct_change_max")
        ])
        .drop_nulls()
        .collect()
    )

    total_obs = final_df.shape[0]
    print(f"Data prepared! Extracted {total_obs} daily stock instances meeting the >= 25 early trades criteria.")

    if total_obs == 0:
        print("Not enough data to perform clustering. Exiting.")
        return

    # Prepare features for clustering
    # We cluster based on: price volume, number of transactions, and price
    features = ["price_volume", "tick_count", "price_845"]
    X_raw = final_df.select(features).to_numpy()

    # Standardize the features so KMeans isn't heavily biased by massive volume numbers
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    print("\n" + "="*80)
    print("UNSUPERVISED K-MEANS CLUSTERING ANALYSIS (N = 3 to 12)")
    print("="*80)

    # Suppress sklearn memory leak warnings on Windows
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for n_clusters in range(3, 13):
            # Fit the unsupervised model
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            cluster_labels = kmeans.fit_predict(X_scaled)

            # Attach cluster labels back to the dataframe
            # Overwrite the 'cluster' column each iteration
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

                    # Let's show the cluster's average underlying profile to understand *what* it is
                    pl.col("price_volume").mean().alias("avg_volume"),
                    pl.col("tick_count").mean().alias("avg_ticks"),
                    pl.col("price_845").mean().alias("avg_price")
                ])
                .sort("advancer_ratio_pct", descending=True)
            )

            print(f"\n--- N_CLUSTERS = {n_clusters} ---")

            # Identify patterns: Is there a cluster heavily skewed to advancers or decliners?
            # A completely random market would show ~50% advancers.
            print(f"{'Cluster':<8} | {'Total Stocks':<13} | {'Avg Max Spike %':<16} | {'Advancer Ratio %':<17} | Profile (Volume / Ticks / Price)")
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
                prof_prc = f"€{row['avg_price']:.2f}"
                profile = f"{prof_vol} / {prof_ticks} / {prof_prc}"

                # Highlight extreme edges
                marker = ""
                if row['advancer_ratio_pct'] >= 65:
                    marker = "<-- HIGH ADVANCER BIAS"
                elif row['advancer_ratio_pct'] <= 35:
                    marker = "<-- HIGH DECLINER BIAS"

                print(f"C_{c_id:<6} | {cnt:<13} | {avg_spike:<16} | {adv_ratio:<17} | {profile} {marker}")

if __name__ == "__main__":
    main()
