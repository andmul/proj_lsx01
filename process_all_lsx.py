import polars as pl
import pytz
import glob
import os
from datetime import datetime, date, timedelta
import holidays
import argparse
import time

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

def log_mem(stage=""):
    """Helper to print current memory usage"""
    if HAS_PSUTIL:
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / (1024 * 1024)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {stage} | Memory: {mem_mb:.2f} MB")
    else:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {stage}")

def extract_names(directory):
    files = glob.glob(os.path.join(directory, 'lsxtradesyesterday_*.csv'))
    if not files: return

    print("Extracting displayName mapping...")
    mapping_dfs = []

    for f in files:
        try:
            # We only need the header to check if displayName is present
            with open(f, 'r', encoding='utf-8') as file:
                header = file.readline()

            if 'displayName' in header:
                # Use quote_char=None to prevent unescaped quotes from crashing the parser
                df = pl.read_csv(
                    f, separator=";", decimal_comma=True,
                    ignore_errors=True, quote_char=None, truncate_ragged_lines=True
                )

                # Strip out the literal quotes that are now part of the string values
                for col in df.columns:
                    if df[col].dtype == pl.Utf8:
                        df = df.with_columns(pl.col(col).str.strip_chars('"'))

                if 'displayName' in df.columns and 'isin' in df.columns:
                    df = df.select(["isin", "displayName"]).drop_nulls().unique()
                    mapping_dfs.append(df)

        except pl.exceptions.NoDataError:
            pass # skip empty files silently
        except Exception as e:
            print(f"Error reading {f} for names: {e}")

    if mapping_dfs:
        final_mapping = pl.concat(mapping_dfs).unique(subset=["isin"], keep="last")
        output_file = "isin_display_names.parquet"
        final_mapping.write_parquet(output_file)
        print(f"Saved {final_mapping.shape[0]} names to {output_file}")
    else:
        print("No files with displayName found.")

def get_trading_days(start_d, end_d):
    """Calculate the number of valid German trading days in the range"""
    # Create a list of all years spanning the entire range to accurately calculate internal holidays
    de_holidays = holidays.DE(years=list(range(start_d.year, end_d.year + 1)))
    days = 0
    curr = start_d
    while curr <= end_d:
        if curr.weekday() < 5 and curr not in de_holidays:
            days += 1
        curr += timedelta(days=1)
    return days

def filter_isins(df, start_date_str, end_date_str):
    print("Filtering ISINs based on activity...")
    start_d = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    end_d = datetime.strptime(end_date_str, "%Y-%m-%d").date()

    total_trading_days = get_trading_days(start_d, end_d)
    print(f"Total valid trading days in period ({start_date_str} to {end_date_str}): {total_trading_days}")

    df = df.with_columns([
        pl.col("tradeTime").dt.truncate("1d").alias("trade_day"),
        (pl.col("size") * pl.col("price")).alias("volume")
    ])

    daily_vol = df.group_by(["isin", "trade_day"]).agg(pl.col("volume").sum().alias("daily_vol"))
    active_days = daily_vol.filter(pl.col("daily_vol") >= 5000)

    isin_active_counts = active_days.group_by("isin").agg(pl.col("trade_day").len().alias("active_days_count"))

    required_active_days = total_trading_days - 25
    if required_active_days <= 0:
        required_active_days = 0

    print(f"ISINs must have at least {required_active_days} active days.")

    valid_isins = isin_active_counts.filter(pl.col("active_days_count") >= required_active_days)

    final_df = df.join(valid_isins.select("isin"), on="isin", how="inner")
    final_df = final_df.drop(["trade_day", "volume"])
    return final_df

import shutil

def process_transactions(directory, start_date_str, end_date_str):
    files = glob.glob(os.path.join(directory, 'lsxtradesyesterday_*.csv'))
    if not files: return None

    # We use a Map-Reduce approach to avoid Out-Of-Memory (OOM) errors.
    # 1. Map: Convert each massive, dirty CSV into a strictly-typed, clean temporary Parquet file.
    # This prevents storing dozens of gigabytes of raw DataFrames in RAM.
    temp_dir = os.path.join(directory, "temp_lsx_cache")
    os.makedirs(temp_dir, exist_ok=True)

    start_d_obj = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    end_d_obj = datetime.strptime(end_date_str, "%Y-%m-%d").date()

    # Filter files based on filename prior to loading them to save disk I/O and processing time.
    target_files = []
    for f in files:
        filename = os.path.basename(f)
        parts = filename.split('_')
        if len(parts) > 1:
            date_str = parts[1].split('.')[0]
            try:
                # The file represents the previous trading day, but checking the filename date
                # against the bounds +/- 5 days is a safe loose filter before strictly filtering internally
                file_date = datetime.strptime(date_str, "%Y%m%d").date()
                if start_d_obj - timedelta(days=5) <= file_date <= end_d_obj + timedelta(days=5):
                    target_files.append(f)
            except ValueError:
                # If we can't parse the filename, include it just in case
                target_files.append(f)

    log_mem(f"Starting map phase: Processing {len(target_files)} relevant files (out of {len(files)} total) into temporary cache...")

    for i, f in enumerate(target_files):
        if i > 0 and i % 50 == 0:
            log_mem(f"  ...processed {i}/{len(target_files)} files")

        try:
            # Eagerly load the file using robust quoting constraints to bypass unescaped delimiters
            df = pl.read_csv(f, separator=";", decimal_comma=True, ignore_errors=True, quote_char=None, truncate_ragged_lines=True)

            # Strip literal quotes out of the strings
            for col in df.columns:
                if df[col].dtype == pl.Utf8:
                    df = df.with_columns(pl.col(col).str.strip_chars('"'))

            if 'orderId' in df.columns:
                df = df.rename({'orderId': 'TVTIC'})
            if 'displayName' in df.columns:
                df = df.drop('displayName')

            # Perform initial conversions so parquet has strict types
            df = df.with_columns(
                 pl.col("tradeTime")
                 .str.replace("Z", "")
                 .str.replace("T"," ")
                 .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S%.f", strict=False)
                .dt.replace_time_zone("UTC")
                .dt.convert_time_zone("Europe/Berlin")
                .alias("tradeTime"),

                pl.col("publishedTime")
                 .str.replace("Z", "")
                 .str.replace("T"," ")
                 .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S%.f", strict=False)
                .dt.replace_time_zone("UTC")
                .dt.convert_time_zone("Europe/Berlin")
                .alias("publishedTime")
            )

            # Since polars might have parsed price with commas literally as string (because quote_char=None bypasses decimal_comma sometimes)
            df = df.with_columns([
                pl.col("price").str.replace(",", ".").cast(pl.Float64, strict=False),
                pl.col("size").cast(pl.Int64, strict=False),
            ])
            df = df.drop_nulls(["price", "size", "tradeTime"])

            # Run the unique constraint early per file to lower storage size
            df = df.sort('publishedTime').unique(subset=['TVTIC'], keep='last')

            out_path = os.path.join(temp_dir, f"chunk_{i}.parquet")
            df.write_parquet(out_path)

        except pl.exceptions.NoDataError:
            pass # Skip silently
        except Exception as e:
            print(f"Error caching {f}: {e}")

    log_mem("Finished map phase.")

    # 2. Reduce: Stream the perfectly uniform Parquet directory natively out-of-core
    log_mem("Starting reduce phase: executing out-of-core lazy evaluation on cached parquet files...")

    # We use scan_parquet which safely streams the files natively without loading them into RAM
    df = pl.scan_parquet(os.path.join(temp_dir, "*.parquet"))

    # Group by identical timestamp to aggregate split executions and amendments/cancellations
    # NOTE: We skip global `.sort().unique()` here because we already deduplicated
    # strictly per file during the Map phase, and doing a global out-of-core sort
    # over 1000 files frequently triggers OS segfaults (0xC0000005) in the Polars engine.
    df = df.with_columns([
        pl.col("tradeTime").dt.truncate("1s").alias("trade_sec")
    ])

    df = df.with_columns(
        pl.when(pl.col("flags").str.contains("CANC"))
        .then(pl.col("size") * -1)
        .otherwise(pl.col("size"))
        .alias("size")
    )

    grouped = df.group_by(["isin", "trade_sec", "price"]).agg([
        pl.col("size").sum().alias("total_size"),
        pl.col("tradeTime").first().alias("tradeTime"),
        pl.col("publishedTime").first().alias("publishedTime"),
        pl.col("TVTIC").first().alias("TVTIC"),
        pl.col("mic").first().alias("mic"),
        pl.col("currency").first().alias("currency"),
        pl.col("quotation").first().alias("quotation")
    ])

    # Delete all flagged amnd/canc after this is done by defaulting the flag
    grouped = grouped.with_columns(pl.lit("ALGO;").alias("flags"))

    grouped = grouped.filter(pl.col("total_size") > 0)
    grouped = grouped.rename({"total_size": "size"})
    grouped = grouped.drop(["trade_sec"])

    # Apply time boundary filter
    start_d = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_d = datetime.strptime(end_date_str, "%Y-%m-%d")

    berlin_tz = pytz.timezone("Europe/Berlin")
    start_dt = berlin_tz.localize(start_d)

    import datetime as dt
    end_dt = berlin_tz.localize(end_d + dt.timedelta(days=1))

    grouped = grouped.filter((pl.col("tradeTime") >= start_dt) & (pl.col("tradeTime") < end_dt))

    # In extremely large environments, `.collect()` can still cause OS-level kills when doing complex
    # join aggregations. We'll sink the highly compressed intermediate grouped dataset to disk first
    # to free memory, then filter the ISINs and sink to the master file.

    intermediate_file = os.path.join(temp_dir, "intermediate_grouped.parquet")
    log_mem("Sinking intermediate reduced dataframe to disk... (This may take a while)")
    try:
        grouped.sink_parquet(intermediate_file)
        log_mem("Successfully wrote intermediate dataset.")

        # Load the newly compressed intermediate dataset back in as a LazyFrame
        compressed_lf = pl.scan_parquet(intermediate_file)

        # Apply the final ISIN filter logic natively
        final_lf = filter_isins(compressed_lf, start_date_str, end_date_str)

        master_file = "consolidated_transactions.parquet"
        log_mem(f"Streaming final dataset directly to {master_file}...")
        final_lf.sink_parquet(master_file)

        log_mem("Master file successfully created!")

    except Exception as e:
        print(f"\nCRITICAL ERROR during execution: {e}")
        print("The script crashed before finishing. Chunks were left behind for debugging.")
        raise e
    finally:
        # ALWAYS clean up temporary cache, even if script fails mid-way
        log_mem("Cleaning up temporary chunk files...")
        shutil.rmtree(temp_dir, ignore_errors=True)

    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=r"d:\lsx", help="Input directory")
    parser.add_argument("--start", default="2023-08-17", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=str(date.today()), help="End date YYYY-MM-DD")
    # use parse_known_args to prevent SystemExit in Jupyter Notebook environments
    args, unknown = parser.parse_known_args()

    log_mem(f"Starting LSX processing pipeline. Checking directory: {args.dir}")

    # 1. Extract mapping
    extract_names(args.dir)

    # 2. Process data
    success = process_transactions(args.dir, args.start, args.end)

    if success:
        log_mem("Pipeline finished successfully! Check for 'consolidated_transactions.parquet' in your directory.")

if __name__ == "__main__":
    main()
