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
            # We use scan_csv to lazily read the massive file directly from disk without loading it fully into RAM.
            # However, because of malformed quotes in the LSX feeds, we must still use robust parsing limits.
            lf = pl.scan_csv(f, separator=";", decimal_comma=True, ignore_errors=True, quote_char=None, truncate_ragged_lines=True)

            # Since quote_char=None brings in quoted headers like '"tradeTime"', we must eagerly rename them
            # if necessary, or just extract the names cleanly. Eager fetch of 1 row is essentially free.
            header_df = pl.read_csv(f, separator=";", ignore_errors=True, quote_char=None, truncate_ragged_lines=True, n_rows=1)
            new_columns = [c.replace("\ufeff", "").strip().strip('"').strip("'") for c in header_df.columns]

            # Set up the lazy evaluation tree
            lf = lf.rename(dict(zip(header_df.columns, new_columns)))

            if 'orderId' in new_columns:
                lf = lf.rename({'orderId': 'TVTIC'})
            if 'displayName' in new_columns:
                lf = lf.drop('displayName')

            if 'orderId' in new_columns:
                lf = lf.rename({'orderId': 'TVTIC'})
            if 'displayName' in new_columns:
                lf = lf.drop('displayName')

            schema = lf.collect_schema()

            # Check the schema to see if tradeTime is already parsed natively or a string
            time_dtype = schema.get("tradeTime")
            if time_dtype == pl.String or time_dtype == pl.Utf8:
                # Perform string conversion
                lf = lf.with_columns([
                    pl.col("tradeTime")
                     .str.strip_chars('"')
                     .str.replace("Z", "")
                     .str.replace("T"," ")
                     .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S%.f", strict=False)
                    .dt.replace_time_zone("UTC")
                    .dt.convert_time_zone("Europe/Berlin")
                ])
            else:
                lf = lf.with_columns(pl.col("tradeTime").dt.replace_time_zone("UTC").dt.convert_time_zone("Europe/Berlin"))

            pub_dtype = schema.get("publishedTime")
            if "publishedTime" in schema.names():
                pub_dtype = schema.get("publishedTime")
                if pub_dtype == pl.String or pub_dtype == pl.Utf8:
                    lf = lf.with_columns([
                        pl.col("publishedTime")
                         .str.strip_chars('"')
                         .str.replace("Z", "")
                         .str.replace("T"," ")
                         .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S%.f", strict=False)
                        .dt.replace_time_zone("UTC")
                        .dt.convert_time_zone("Europe/Berlin")
                    ])
                else:
                    lf = lf.with_columns(pl.col("publishedTime").dt.replace_time_zone("UTC").dt.convert_time_zone("Europe/Berlin"))

            # Extract hour and minute natively inside the lazy frame
            lf = lf.with_columns([
                pl.col("tradeTime").dt.hour().alias("hour"),
                pl.col("tradeTime").dt.minute().alias("minute"),
                pl.col("tradeTime").dt.truncate("1d").alias("trade_day")
            ])

            # Strip literal quotes out of strings based on schema check
            # We must verify the column STILL exists in lf.columns before stripping
            for col_name, dtype in schema.items():
                if col_name not in ["tradeTime", "publishedTime", "hour", "minute"] and col_name in new_columns:
                    if dtype == pl.String or dtype == pl.Utf8:
                        lf = lf.with_columns(pl.col(col_name).str.strip_chars('"').alias(col_name))

            # Parse numerics based on schema check
            price_dtype = schema.get("price")
            if price_dtype == pl.String or price_dtype == pl.Utf8:
                lf = lf.with_columns([
                    pl.col("price").str.replace(",", ".").cast(pl.Float64, strict=False)
                ])
            else:
                lf = lf.with_columns([
                    pl.col("price").cast(pl.Float64, strict=False)
                ])

            lf = lf.with_columns(pl.col("size").cast(pl.Int64, strict=False))

            lf = lf.drop_nulls(["price", "size", "tradeTime"])

            # ISIN FILTER: Only keep ISIN/day combinations with >= 20 ticks between 7:30 and 9:00
            # We want to identify them lazily, then join against the FULL daily dataset so we don't
            # drop the afternoon data the machine learning scripts require.

            valid_isins_lf = (
                lf.filter(
                    ((pl.col("hour") == 7) & (pl.col("minute") >= 30)) |
                    ((pl.col("hour") == 8)) |
                    ((pl.col("hour") == 9) & (pl.col("minute") == 0))
                )
                .group_by(["isin", "trade_day"])
                .agg(pl.len().alias("morning_ticks"))
                .filter(pl.col("morning_ticks") >= 20)
                .select(["isin", "trade_day"])
            )

            # To actually join the full-day stream against the valid morning ISINs WITHOUT losing the afternoon data,
            # we must recreate the original lazyframe stream before we filtered it.
            # The current `lf` object was already filtered to `07:30 to 09:00` on line 191!

            # Recreate the raw stream for the final join
            raw_lf = pl.scan_csv(f, separator=";", decimal_comma=True, ignore_errors=True, quote_char=None, truncate_ragged_lines=True)
            raw_lf = raw_lf.rename(dict(zip(header_df.columns, new_columns)))

            if 'orderId' in new_columns:
                raw_lf = raw_lf.rename({'orderId': 'TVTIC'})
            if 'displayName' in new_columns:
                raw_lf = raw_lf.drop('displayName')

            raw_lf = raw_lf.with_columns([
                pl.col("tradeTime")
                 .str.strip_chars('"')
                 .str.replace("Z", "")
                 .str.replace("T"," ")
                 .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S%.f", strict=False)
                .dt.replace_time_zone("UTC")
                .dt.convert_time_zone("Europe/Berlin"),

                pl.col("publishedTime")
                 .str.strip_chars('"')
                 .str.replace("Z", "")
                 .str.replace("T"," ")
                 .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S%.f", strict=False)
                .dt.replace_time_zone("UTC")
                .dt.convert_time_zone("Europe/Berlin")
            ])

            for col_name, dtype in schema.items():
                if col_name not in ["tradeTime", "publishedTime", "hour", "minute"] and col_name in new_columns:
                    if dtype == pl.String or dtype == pl.Utf8:
                        raw_lf = raw_lf.with_columns(pl.col(col_name).str.strip_chars('"').alias(col_name))

            if price_dtype == pl.String or price_dtype == pl.Utf8:
                raw_lf = raw_lf.with_columns([pl.col("price").str.replace(",", ".").cast(pl.Float64, strict=False)])
            else:
                raw_lf = raw_lf.with_columns([pl.col("price").cast(pl.Float64, strict=False)])

            raw_lf = raw_lf.with_columns([
                pl.col("size").cast(pl.Int64, strict=False),
                pl.col("tradeTime").dt.truncate("1d").alias("trade_day")
            ])
            raw_lf = raw_lf.drop_nulls(["price", "size", "tradeTime"])

            # Join the raw full-day stream against the valid morning ISINs
            # This massively reduces RAM usage by dropping inactive stocks entirely,
            # while keeping the full-day trajectory for the active ones.
            lf_final = raw_lf.join(valid_isins_lf, on=["isin", "trade_day"], how="inner")

            # Execute the constrained stream
            df = lf_final.collect()

            if df.height == 0:
                print(f"File {f} had no ISINs matching the morning tick criteria. Skipping caching.")
                continue

            # Run the unique constraint early per file to lower storage size
            # Sometimes publishedTime doesn't parse well, so fallback to tradeTime for sorting
            sort_col = "publishedTime" if "publishedTime" in df.columns and df["publishedTime"].null_count() == 0 else "tradeTime"
            df = df.sort(sort_col).unique(subset=['TVTIC'], keep='last')

            # Cleanup temporary columns before writing parquet
            df = df.drop(["hour", "minute", "trade_day"])

            out_path = os.path.join(temp_dir, f"chunk_{i}.parquet")
            df.write_parquet(out_path)
            log_mem(f"  -> Dumped chunk_{i}.parquet (Rows: {df.height})")

        except pl.exceptions.NoDataError:
            pass # Skip silently
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error caching {f}: {e}")

    log_mem("Finished map phase.")

    # Check if ANY chunks were written
    chunk_files = glob.glob(os.path.join(temp_dir, "*.parquet"))
    if not chunk_files:
        print("Warning: No files contained valid morning trades matching the ISIN filters.")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return False

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
