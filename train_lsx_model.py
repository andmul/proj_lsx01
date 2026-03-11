import polars as pl
import lightgbm as lgb
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Train LightGBM model on LSX Parquet data")
    parser.add_argument("--file", default="consolidated_transactions.parquet", help="Path to input parquet file")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: {args.file} not found. Please run process_all_lsx.py first.")
        return

    print(f"Loading and processing {args.file}...")

    # Output directory for individual ISIN models
    models_dir = "lsx_models"
    os.makedirs(models_dir, exist_ok=True)

    # 1. Use Lazy Execution (scan_parquet instead of read_parquet)
    # We sort by ISIN and tradeTime globally to allow Polars to compute rolling features in native Rust speed
    query = (
        pl.scan_parquet(args.file)
        .sort(['isin', 'tradeTime'])

        # Calculate volume dynamically
        .with_columns([
            (pl.col('size') * pl.col('price')).alias('volume')
        ])

        # 2. Create Time-Aware and Tick-Aware Features per ISIN
        .with_columns([
            pl.col('tradeTime').diff().dt.total_milliseconds().over('isin').alias('delta_t_ms'),

            # Predict the percentage return to the next tick
            ((pl.col('price').shift(-1).over('isin') - pl.col('price')) / pl.col('price')).alias('target_return'),

            pl.col('volume').rolling_sum(window_size=10).over('isin').alias('vol_sum_10tick'),
            pl.col('volume').rolling_sum(window_size=50).over('isin').alias('vol_sum_50tick'),

            pl.col('price').rolling_std(window_size=20).over('isin').alias('price_std_20tick')
        ])
    )

    # 3. Execute the optimized query into memory
    print("Executing query and materializing rolling features into RAM...")
    df = query.collect()

    # Drop rows where rolling windows or future-shifts created nulls
    df = df.drop_nulls(['target_return', 'delta_t_ms', 'vol_sum_50tick', 'price_std_20tick'])

    unique_isins = df.select("isin").unique().to_series().to_list()
    print(f"Dataset prepared. Total valid rows: {df.shape[0]} across {len(unique_isins)} ISINs.")

    if df.shape[0] == 0:
        print("Not enough data to train (the dataset might be too small to satisfy the 50-tick rolling window constraint).")
        return

    # We do not need the ISIN itself as a feature anymore since we train a dedicated model per ISIN
    features = ['price', 'volume', 'delta_t_ms', 'vol_sum_10tick', 'vol_sum_50tick', 'price_std_20tick']

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'verbose': -1  # Suppress internal C++ warnings
    }

    print(f"Starting dedicated LightGBM training loop for {len(unique_isins)} models...")

    # 4. Iterate over ISINs and train independent models
    success_count = 0
    for i, stock_isin in enumerate(unique_isins):
        # Filter dataframe for just this ISIN natively in Polars
        isin_df = df.filter(pl.col("isin") == stock_isin)

        # If an ISIN doesn't have enough rows to learn anything meaningful (e.g., minimum 100 rows), skip it
        if isin_df.shape[0] < 100:
            print(f"[{i+1}/{len(unique_isins)}] Skipping {stock_isin} - Insufficient data ({isin_df.shape[0]} rows)")
            continue

        X = isin_df.select(features).to_numpy()
        y = isin_df.select('target_return').to_numpy().ravel()

        train_data = lgb.Dataset(X, label=y)

        try:
            model = lgb.train(params, train_data, num_boost_round=100)
            model_file = os.path.join(models_dir, f"model_{stock_isin}.txt")
            model.save_model(model_file)
            success_count += 1
            if success_count % 50 == 0:
                print(f"[{i+1}/{len(unique_isins)}] Trained and saved model for {stock_isin} ({isin_df.shape[0]} ticks)")
        except Exception as e:
            print(f"[{i+1}/{len(unique_isins)}] Error training {stock_isin}: {e}")

    print(f"\nTraining pipeline complete! Successfully generated {success_count} independent ISIN models in the '{models_dir}/' directory.")

if __name__ == "__main__":
    main()
