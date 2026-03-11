"""
=============================================================================
PROJECT: 0.5% Forward-Looking Breakout Classifier
=============================================================================
DATA: ~40M rows of irregular transactional tick data (Parquet).
HARDWARE LIMIT: 32GB RAM (Out-of-core processing required).

ARCHITECTURE NOTES FOR JULES:
1. Memory Management: Uses Polars `scan_parquet` for lazy execution. Do not
   materialize the dataset into RAM until the final `collect()` after the join.
2. Target Engineering: We are predicting a binary outcome: 1 if the max price
   in the *future* 2 hours is >= 1.005 * max price of the *past* 2 hours.
3. Shift-and-Join: To prevent data leakage, the future rolling window is
   calculated, its timestamps are shifted backward by exactly 2 hours, and
   then it is merged to the past features using a forward `join_asof`.
4. Validation: Strict chronological split (80/20). Shuffling is explicitly
   disabled to prevent look-ahead bias.
5. Model: LightGBM (Objective: Binary).
=============================================================================
"""

import polars as pl
import lightgbm as lgb
import matplotlib.pyplot as plt
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Train 0.5% Breakout Classifier on LSX Parquet data")
    parser.add_argument("--file", default="consolidated_transactions.parquet", help="Path to input parquet file")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: {args.file} not found. Please run process_all_lsx.py first.")
        return

    print(f"Loading and processing {args.file}...")

    models_dir = "lsx_breakout_models"
    os.makedirs(models_dir, exist_ok=True)

    # ==============================================================================
    # 1. LOAD DATA
    # ==============================================================================
    # Lazy load and ensure the timeline is strictly chronological PER ISIN
    df = (
        pl.scan_parquet(args.file)
        .sort(['isin', 'tradeTime'])
        # Re-calculate volume since it was omitted from the raw consolidated dump
        .with_columns([
            (pl.col('size') * pl.col('price')).alias('volume')
        ])
    )

    # ==============================================================================
    # 2. FEATURE ENGINEERING (The "Past 2 Hours")
    # ==============================================================================
    # Calculate what the market looked like in the 2 hours exactly prior to each tick, strictly ISOLATED BY ISIN
    past_features = df.rolling(index_column="tradeTime", period="2h", by="isin").agg([
        pl.col("price").last().alias("current_price"),
        pl.col("price").max().alias("past_2h_max_price"),
        pl.col("volume").sum().alias("past_2h_volume"),
        pl.len().alias("past_2h_tick_count")
    ])

    # ==============================================================================
    # 3. TARGET ENGINEERING (The "Future 2 Hours")
    # ==============================================================================
    # Calculate the max price in the upcoming 2 hours per ISIN, then shift the timestamps
    # backward so they align with the "current" transaction.
    future_target = (
        df.select(['isin', 'tradeTime', 'price'])
        .rolling(index_column="tradeTime", period="2h", by="isin")
        .agg([
            pl.col("price").max().alias("future_2h_max_price")
        ])
        .with_columns([
            (pl.col("tradeTime") - pl.duration(hours=2)).alias("tradeTime")
        ])
        # Re-sort is required by Polars after modifying the index_column before a join_asof
        .sort(['isin', 'tradeTime'])
    )

    # ==============================================================================
    # 4. MERGE & CREATE BINARY LABELS (The "0.5% Breakout")
    # ==============================================================================
    # Join past and future without data leakage matching EXACTLY on ISIN, apply the 0.5% rule, and load into RAM
    print("Executing complex timeframe aggregations and materializing into RAM...")
    final_df = (
        past_features.join_asof(
            future_target,
            on="tradeTime",
            by="isin",
            strategy="forward"
        )
        .with_columns([
            # Target = 1 if the future max price is at least 0.5% higher than the past max price
            (pl.col("future_2h_max_price") >= (pl.col("past_2h_max_price") * 1.005)).cast(pl.Int32).alias("target_label")
        ])
        .drop_nulls()
        .collect() # Executes the lazy graph
    )

    print(f"Dataset prepared. Total valid rows: {final_df.height}")

    if final_df.height == 0:
        print("Not enough data to train. Exiting.")
        return

    # ==============================================================================
    # 5. CHRONOLOGICAL TRAIN/TEST SPLIT
    # ==============================================================================
    unique_isins = final_df.select("isin").unique().to_series().to_list()
    print(f"Starting dedicated LightGBM training loop for {len(unique_isins)} models...")

    success_count = 0
    for i, stock_isin in enumerate(unique_isins):
        # Filter dataframe for just this ISIN natively in Polars
        isin_df = final_df.filter(pl.col("isin") == stock_isin)

        # If an ISIN doesn't have enough rows to learn anything meaningful, skip it
        if isin_df.shape[0] < 100:
            continue

        total_rows = isin_df.height
        train_size = int(total_rows * 0.8)

        train_df = isin_df.head(train_size)
        test_df = isin_df.tail(total_rows - train_size)

        features = ['current_price', 'past_2h_max_price', 'past_2h_volume', 'past_2h_tick_count']
        target = 'target_label'

        X_train = train_df.select(features).to_numpy()
        y_train = train_df.select(target).to_numpy().ravel()

        X_test = test_df.select(features).to_numpy()
        y_test = test_df.select(target).to_numpy().ravel()

        # ==============================================================================
        # 6. MODEL TRAINING (Binary Classification)
        # ==============================================================================
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=features)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data, feature_name=features)

        params = {
            'objective': 'binary',
            'metric': ['binary_logloss', 'auc'],
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'verbose': -1
        }

        try:
            model = lgb.train(
                params,
                train_data,
                valid_sets=[train_data, test_data],
                num_boost_round=200,
                callbacks=[lgb.early_stopping(stopping_rounds=10)]
            )
            model_file = os.path.join(models_dir, f"model_{stock_isin}.txt")
            model.save_model(model_file)
            success_count += 1
            if success_count % 50 == 0:
                print(f"[{i+1}/{len(unique_isins)}] Trained and saved classifier for {stock_isin} ({isin_df.shape[0]} ticks)")
        except Exception as e:
            print(f"[{i+1}/{len(unique_isins)}] Error training {stock_isin}: {e}")

    print(f"\nTraining pipeline complete! Successfully generated {success_count} independent ISIN models in the '{models_dir}/' directory.")

if __name__ == "__main__":
    main()
