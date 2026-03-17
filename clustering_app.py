import streamlit as st
import polars as pl
import numpy as np
import warnings
import os
import plotly.graph_objects as go
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.utils import to_time_series_dataset

# ==============================================================================
# APP CONFIG
# ==============================================================================
st.set_page_config(page_title="Tick-by-Tick Cluster Explorer", layout="wide")
st.title("Intraday Time-Series Clustering (Raw Ticks)")
st.markdown("Compare dynamic variable-length tick clustering algorithms without sampling or aggregations.")

# ==============================================================================
# SIDEBAR
# ==============================================================================
st.sidebar.header("1. Data Extraction Pipeline")
start_hour = st.sidebar.number_input("Morning Start Hour", min_value=0, max_value=23, value=7)
start_minute = st.sidebar.number_input("Morning Start Minute", min_value=0, max_value=59, value=30)
end_hour = st.sidebar.number_input("Morning End Hour", min_value=0, max_value=23, value=9)
end_minute = st.sidebar.number_input("Morning End Minute", min_value=0, max_value=59, value=0)

min_ticks = st.sidebar.slider("Minimum Ticks per Morning", min_value=10, max_value=200, value=25)
max_samples = st.sidebar.slider("Max ISINs (Prevents DTW Freezing)", min_value=50, max_value=1000, value=200)

st.sidebar.header("2. Algorithm Configuration")
clustering_metric = st.sidebar.selectbox("Distance Metric", ["dtw", "softdtw", "euclidean"])
k_start = st.sidebar.number_input("Grid Search K Min", min_value=2, max_value=20, value=3)
k_end = st.sidebar.number_input("Grid Search K Max", min_value=3, max_value=50, value=8)

if st.sidebar.button("Run Extraction & Clustering"):
    file_path = "consolidated_transactions.parquet"
    if not os.path.exists(file_path):
        st.error(f"Cannot find {file_path}. Please generate it first.")
        st.stop()

    with st.spinner("Extracting dynamic tick arrays from Polars..."):
        # ==========================================================================
        # EXTRACT RAW TICK ARRAYS (NO BUCKETING OR RESAMPLING)
        # ==========================================================================
        # Fallback for older Polars versions that do not have `scan_parquet` natively
        if hasattr(pl, "scan_parquet"):
            df = pl.scan_parquet(file_path)
        else:
            df = pl.read_parquet(file_path).lazy()

        df = df.with_columns([
            pl.col("tradeTime").dt.truncate("1d").alias("trade_day"),
            pl.col("tradeTime").dt.hour().alias("hour"),
            pl.col("tradeTime").dt.minute().alias("minute")
        ])

        # We need integer logic for minute boundaries
        df = df.with_columns( (pl.col("hour") * 60 + pl.col("minute")).alias("time_val") )
        start_val = start_hour * 60 + start_minute
        end_val = end_hour * 60 + end_minute

        morning_df = df.filter((pl.col("time_val") >= start_val) & (pl.col("time_val") < end_val))

        # Raw sequence array aggregation
        agg = (
            morning_df
            .sort(["isin", "tradeTime"])
            .group_by(["isin", "trade_day"])
            .agg([
                pl.len().alias("total_ticks"),
                pl.col("price").last().alias("price_cutoff"),
                pl.col("price").alias("price_series"),
                (pl.col("size") * pl.col("price")).alias("tick_volume_series")
            ])
            .filter(pl.col("total_ticks") >= min_ticks)
        ).collect()

        # Market targets (e.g. 09:00 to 17:00)
        market_df = df.filter((pl.col("hour") >= end_hour) & (pl.col("hour") < 17)).collect()
        market_highs = market_df.group_by(["isin", "trade_day"]).agg(pl.col("price").max().alias("day_high"))

        final_df = agg.join(market_highs, on=["isin", "trade_day"], how="inner")
        final_df = final_df.with_columns([
            (((pl.col("day_high") - pl.col("price_cutoff")) / pl.col("price_cutoff")) * 100).alias("pct_change_max")
        ])

        if final_df.height == 0:
            st.error("No instances found matching your criteria. Try widening the time window or lowering Minimum Ticks.")
            st.stop()

        if final_df.height > max_samples:
            st.warning(f"Data truncated to {max_samples} instances to prevent DTW CPU starvation.")
            final_df = final_df.head(max_samples)

        st.success(f"Extracted {final_df.height} raw trajectory arrays.")

    with st.spinner("Compiling variable-length sequences to normalized shape vectors..."):
        time_series_data = []
        for row in final_df.iter_rows(named=True):
            price = np.array(row["price_series"], dtype=float)
            tick_vol = np.array(row["tick_volume_series"], dtype=float)
            cum_vol = np.cumsum(tick_vol)
            ts = np.column_stack((price, cum_vol))
            time_series_data.append(ts)

        X_raw = to_time_series_dataset(time_series_data)

        # Shape normalization logic (MinMax)
        scaler = TimeSeriesScalerMinMax()
        X_scaled = scaler.fit_transform(X_raw)

        # Handle zero-division NaNs gracefully leaving padding NaNs intact
        for i in range(X_scaled.shape[0]):
            for d in range(X_scaled.shape[2]):
                valid_idx = ~np.isnan(X_scaled[i, :, d])
                if np.any(valid_idx) and np.all(np.isnan(X_scaled[i, valid_idx, d])):
                    X_scaled[i, valid_idx, d] = 0.0

    st.write(f"Sequence Matrix Compiled: `Instances={X_scaled.shape[0]} | Max Ticks={X_scaled.shape[1]} | Dims={X_scaled.shape[2]}`")

    # ==========================================================================
    # GRID SEARCH CLUSTERING
    # ==========================================================================
    st.subheader(f"Grid Search Results (Metric: {clustering_metric})")

    best_ratio = -1
    best_k = 0
    best_df = None
    best_model = None

    progress = st.progress(0)
    total_steps = k_end - k_start + 1
    step = 0

    results = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k in range(k_start, k_end + 1):
            model = TimeSeriesKMeans(n_clusters=k, metric=clustering_metric, max_iter=5, random_state=42, n_jobs=-1)
            labels = model.fit_predict(X_scaled)

            c_df = final_df.with_columns([
                pl.Series("cluster", labels),
                (pl.col("pct_change_max") > 0).cast(pl.Int32).alias("is_advancer")
            ])

            stats = c_df.group_by("cluster").agg([
                pl.len().alias("count"),
                (pl.col("is_advancer").sum() / pl.len() * 100).alias("advancer_ratio")
            ]).sort("advancer_ratio", descending=True)

            top_ratio = stats["advancer_ratio"][0]
            top_cluster = stats["cluster"][0]
            top_count = stats["count"][0]

            results.append({"K": k, "Top Advancer Ratio": top_ratio, "Instances": top_count})

            if top_count >= 5 and top_ratio > best_ratio:
                best_ratio = top_ratio
                best_k = k
                best_df = c_df
                best_model = model

            step += 1
            progress.progress(step / total_steps)

    st.table(pl.DataFrame(results).to_pandas())

    if best_df is not None:
        st.success(f"**Best Configuration:** K={best_k} produced a cluster with {best_ratio:.2f}% advancers.")

        # Plot the winning centroids (the underlying shapes discovered)
        st.subheader("Discovered Centroid Shapes (Normalized Price Trajectory)")
        fig = go.Figure()

        centroids = best_model.cluster_centers_
        # Plot price dim 0 for each cluster
        for c_idx in range(best_k):
            # Centroids might contain trailing NaNs due to padding averaging
            c_shape = centroids[c_idx, :, 0]
            valid_shape = c_shape[~np.isnan(c_shape)]
            fig.add_trace(go.Scatter(y=valid_shape, mode='lines', name=f"Cluster {c_idx}"))

        fig.update_layout(xaxis_title="Normalized Tick Timeline", yaxis_title="Normalized Price [0,1]")
        st.plotly_chart(fig, use_container_width=True)
