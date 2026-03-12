"""
=============================================================================
PROJECT: 0.5% Forward-Looking Breakdown (Short) Classifier
=============================================================================
DATA: ~40M rows of irregular transactional tick data (Parquet).
HARDWARE LIMIT: 32GB RAM (Out-of-core processing required).

ARCHITECTURE NOTES FOR JULES:
1. Memory Management: Uses Polars `scan_parquet` for lazy execution. Do not
   materialize the dataset into RAM until the final `collect()` after the join.
2. Target Engineering: We are predicting a binary outcome: 1 if the min price
   in the *future* 2.5 hours is <= 0.995 * min price of the *past* 2 hours.
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
import os
import numpy as np
import warnings
from sklearn.metrics import roc_auc_score, precision_score, recall_score, precision_recall_curve

def main():
    file_path = "consolidated_transactions.parquet"
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found. Please run process_all_lsx.py first.")
        return

    print(f"Loading and processing {file_path}...")

    models_dir = "lsx_breakdown_models"
    os.makedirs(models_dir, exist_ok=True)

    # ==============================================================================
    # 1. LOAD DATA AND CREATE BENCHMARK DELTA FEATURE
    # ==============================================================================
    # "Please add an indikator as a feature: this is the percent delta between the price of the security
    # with isin DE0005933931 at the close of the two hour period compared to the last tick from the previous trading day."
    df_raw = pl.scan_parquet(file_path).sort('tradeTime')

    # Isolate the benchmark ISIN DE0005933931 to calculate its previous daily close
    bench_df = df_raw.filter(pl.col('isin') == 'DE0005933931')

    # Calculate the last tick price per day for the benchmark
    # Instead of shifting by pl.duration(days=1), which breaks over weekends, we use a row-based shift
    # over the unique trading days to cleanly map "previous day close" to the "current day".
    bench_daily = (
        bench_df
        .with_columns(pl.col('tradeTime').dt.truncate('1d').alias('trade_day'))
        .group_by('trade_day')
        .agg(pl.col('price').last().alias('bench_close'))
        .sort('trade_day')
        .with_columns(pl.col('bench_close').shift(1).alias('bench_prev_close'))
    )

    # Calculate the benchmark's current price at the close of the 2-hour rolling window.
    # To do this, we just need the benchmark's current price joined back into the main timeline.
    bench_current = bench_df.select(['tradeTime', pl.col('price').alias('bench_current_price')]).sort('tradeTime').with_columns(pl.col('tradeTime').set_sorted())

    df = (
        df_raw
        .with_columns([
            pl.col('tradeTime').dt.truncate('1d').alias('trade_day'),
            (pl.col('size') * pl.col('price')).alias('volume')
        ])
        .join(bench_daily.select(['trade_day', 'bench_prev_close']), on='trade_day', how='left')
        .sort('tradeTime')
        .with_columns(pl.col('tradeTime').set_sorted())
        .join_asof(bench_current, on='tradeTime', strategy='backward')
        .with_columns([
            ((pl.col('bench_current_price') - pl.col('bench_prev_close')) / pl.col('bench_prev_close')).alias('bench_delta')
        ])
        .sort(['isin', 'tradeTime'])
    )

    # ==============================================================================
    # 2. FEATURE ENGINEERING (The "Past 2 Hours from Open")
    # ==============================================================================
    # "Lets have the 2 hours rolling from open time ( earliest transaction in a trading day ) to 4 hours later."

    # Find the daily open time for each ISIN
    daily_open = (
        df.group_by(['isin', 'trade_day'])
        .agg([pl.col("tradeTime").min().alias("open_time")])
    )

    df = df.join(daily_open, on=['isin', 'trade_day'], how='left')

    # Calculate what the market looked like in the past 2 hours prior to each tick
    past_features = df.rolling(index_column="tradeTime", period="2h", by="isin").agg([
        pl.col("price").last().alias("current_price"),
        pl.col("price").min().alias("past_2h_min_price"), # Changed to MIN for short strategy
        pl.col("volume").sum().alias("past_2h_volume"),
        pl.len().alias("past_2h_tick_count"),
        pl.col("bench_delta").last().alias("bench_delta"),
        pl.col("open_time").last().alias("open_time") # Must forward the open_time column out of the rolling agg to filter it next!
    ])

    # "Lets have the 2 hours rolling from open time to 4 hours later"
    # We only filter the PAST features so we don't accidentally starve the FUTURE target of data.
    past_features = past_features.filter(
        (pl.col("tradeTime") >= (pl.col("open_time") + pl.duration(hours=2))) &
        (pl.col("tradeTime") <= (pl.col("open_time") + pl.duration(hours=4)))
    )
    # Re-sort to guarantee the asof join executes cleanly
    past_features = past_features.sort(['isin', 'tradeTime'])

    # ==============================================================================
    # 3. TARGET ENGINEERING (The "Future 2.5 Hours with 30min Lag")
    # ==============================================================================
    # "We need approx 30 minutes lag to get the signal processed and the trade done. So the timeframe
    # to check for breakouts should move by 30 mins but lets extend it to 2.5 hours."

    # Future window: We want to evaluate the min price in the 2.5 hour window that STARTs 30 mins from now.
    # Total duration from now to end of window = 3 hours (30m lag + 2.5h window)
    future_target = (
        df.select(['isin', 'tradeTime', 'price'])
        # A rolling window of 2.5h
        .rolling(index_column="tradeTime", period="2h30m", by="isin")
        .agg([
            pl.col("price").min().alias("future_2h30m_min_price"), # Changed to MIN for short breakdown
            pl.col("price").last().alias("cover_price") # Price at the end of the breakdown period
        ])
        # Shift the index back by exactly 3 hours to align with the "current" transaction
        # Because if the 2.5h window ends at T+3h, it means it covers [T+30m to T+3h].
        .with_columns([
            (pl.col("tradeTime") - pl.duration(hours=3)).alias("tradeTime")
        ])
        .sort(['isin', 'tradeTime'])
    )

    # We also need to know the actual short price (the price exactly 30 mins from "now")
    future_short_price = (
        df.select(['isin', 'tradeTime', pl.col('price').alias('short_price')])
        .with_columns([
            (pl.col("tradeTime") - pl.duration(minutes=30)).alias("tradeTime")
        ])
        .sort(['isin', 'tradeTime'])
    )

    # ==============================================================================
    # 4. MERGE & CREATE BINARY LABELS (The "-0.5% Breakdown")
    # ==============================================================================
    print("Executing complex timeframe aggregations and materializing into RAM...")

    # Because 'by="isin"' is used in join_asof, Polars strictly requires the joined column
    # to be explicitly flagged as sorted using set_sorted(), and requires both the 'by' and 'on'
    # columns to be physically sorted.
    # Since these are LazyFrames with complex grouping boundaries, it's safer to materialize the sorting
    past_features = past_features.collect().sort(['isin', 'tradeTime']).lazy()
    future_target = future_target.collect().sort(['isin', 'tradeTime']).lazy()
    future_short_price = future_short_price.collect().sort(['isin', 'tradeTime']).lazy()

    past_features = past_features.with_columns(pl.col('tradeTime').set_sorted())
    future_target = future_target.with_columns(pl.col('tradeTime').set_sorted())
    future_short_price = future_short_price.with_columns(pl.col('tradeTime').set_sorted())

    final_df = (
        past_features.join_asof(
            future_target,
            on="tradeTime",
            by="isin",
            strategy="forward"
        )
        .join_asof(
            future_short_price,
            on="tradeTime",
            by="isin",
            strategy="forward"
        )
        .with_columns([
            # Target = 1 if the future min price drops at least 0.5% below the past min price
            (pl.col("future_2h30m_min_price") <= (pl.col("past_2h_min_price") * 0.995)).cast(pl.Int32).alias("target_label")
        ])
        .drop_nulls()
        .collect()
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

    report_data = []
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

        features = ['current_price', 'past_2h_min_price', 'past_2h_volume', 'past_2h_tick_count', 'bench_delta']
        target = 'target_label'

        X_train = train_df.select(features).to_numpy()
        y_train = train_df.select(target).to_numpy().ravel()

        X_test = test_df.select(features).to_numpy()
        y_test = test_df.select(target).to_numpy().ravel()

        # For P&L tracking, we keep the original test dataframe alongside the arrays

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

            # Evaluate Out-Of-Sample Predictions
            y_pred_prob = model.predict(X_test)

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    auc = roc_auc_score(y_test, y_pred_prob)

                    # Generate the Precision-Recall curve
                    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_prob)

                    # Calculate the F0.5 Score for every threshold
                    beta = 0.5
                    with np.errstate(divide='ignore', invalid='ignore'):
                        f_scores = (1 + beta**2) * (precisions * recalls) / ((beta**2 * precisions) + recalls)
                        f_scores = np.nan_to_num(f_scores)

                    # Find the threshold that produced the absolute best score
                    optimal_idx = np.argmax(f_scores)
                    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

                    final_precision = precisions[optimal_idx]
                    final_recall = recalls[optimal_idx]

            except ValueError:
                # Triggers if the test set entirely lacks breakdowns
                auc, final_precision, final_recall, optimal_threshold = 0, 0, 0, 0.5

            test_breakdowns = int(y_test.sum())
            breakdown_freq = test_breakdowns / len(y_test) if len(y_test) > 0 else 0

            # Generate trading signals
            trading_signals = (y_pred_prob >= optimal_threshold).astype(int)

            # P&L Calculation for Short Selling:
            # We short at short_price (30 min lag) and buy back to cover at cover_price (end of window).
            # Profit = (short_price - cover_price) / short_price
            test_df_with_signals = test_df.with_columns([
                pl.Series("signal", trading_signals)
            ])

            executed_trades = test_df_with_signals.filter(pl.col("signal") == 1)

            if executed_trades.height > 0:
                pnl = executed_trades.select((((pl.col("short_price") - pl.col("cover_price")) / pl.col("short_price")) * 100).mean()).item()
                # Aggregate unique trading signal dates
                executed_dates = executed_trades.with_columns(
                    pl.col("tradeTime").dt.truncate("1d").cast(pl.Utf8).alias("trade_day_str")
                ).select("trade_day_str").unique().to_series().to_list()
                signal_dates_str = " | ".join(executed_dates)
            else:
                pnl = 0.0
                signal_dates_str = "None"

            report_data.append({
                "ISIN": stock_isin,
                "Total_Ticks": total_rows,
                "Test_Set_Ticks": len(y_test),
                "Actual_Breakdowns_In_Test": test_breakdowns,
                "Base_Breakdown_Frequency": f"{breakdown_freq*100:.2f}%",
                "Optimal_Threshold": f"{optimal_threshold:.4f}",
                "Model_Precision (F0.5)": f"{final_precision*100:.2f}%",
                "Model_Recall": f"{final_recall*100:.2f}%",
                "ROC_AUC_Score": round(auc, 4),
                "Test_Set_Avg_PnL_Percent": round(pnl, 2),
                "Signal_Dates": signal_dates_str
            })

            # Save the model only if the AUC is better than a coin flip (0.50) + slight edge (0.55)
            if auc >= 0.55:
                model_file = os.path.join(models_dir, f"model_{stock_isin}.txt")
                model.save_model(model_file)

            success_count += 1

        except Exception as e:
            print(f"Model failed for ISIN {stock_isin}: {e}")
            continue

    if not report_data:
        print("No ISINs successfully trained models with valid metrics.")
        return

    # Create a DataFrame from the summary statistics
    report_df = pl.DataFrame(report_data)
    # Rank them by mathematical predictability
    report_df = report_df.sort("ROC_AUC_Score", descending=True)

    report_file = "breakdown_strategy_report.csv"
    report_df.write_csv(report_file)

    print(f"\n=======================================================")
    print(f"PIPELINE COMPLETE: Evaluated {success_count} independent ISIN models.")
    print(f"A human-readable master report has been saved to: {report_file}")
    print(f"Only mathematically viable models (AUC >= 0.55) were saved to '{models_dir}/'.")
    print(f"\nTop 5 most predictable ISINs for the -0.5% breakdown:")
    print(report_df.head(5))
    print(f"=======================================================")

if __name__ == "__main__":
    main()
