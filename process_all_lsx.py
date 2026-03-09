import polars as pl
import pytz
import glob
import os
from datetime import datetime, date, timedelta
import holidays
import argparse

def extract_names(directory):
    files = glob.glob(os.path.join(directory, 'lsxtradesyesterday_*.csv'))
    if not files: return

    print("Extracting displayName mapping...")
    mapping_dfs = []

    for f in files:
        try:
            with open(f, 'r', encoding='utf-8') as file:
                header = file.readline()

            if 'displayName' in header:
                df = pl.read_csv(f, separator=";", decimal_comma=True, ignore_errors=True)
                if 'displayName' in df.columns and 'isin' in df.columns:
                    df = df.select(["isin", "displayName"]).drop_nulls().unique()
                    mapping_dfs.append(df)
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
    de_holidays = holidays.DE(years=[start_d.year, end_d.year])
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
    print(f"Found {valid_isins.shape[0]} valid ISINs.")

    final_df = df.join(valid_isins.select("isin"), on="isin", how="inner")
    final_df = final_df.drop(["trade_day", "volume"])
    return final_df

def process_transactions(directory, start_date_str, end_date_str):
    files = glob.glob(os.path.join(directory, 'lsxtradesyesterday_*.csv'))
    if not files: return None

    # Filter files strictly if needed, but since we already downloaded the specific ones, we can parse all in the dir
    print(f"Loading {len(files)} files for consolidation...")

    df_list = []
    for f in files:
        try:
            df = pl.read_csv(f, separator=";", decimal_comma=True, ignore_errors=True)
            if 'orderId' in df.columns:
                df = df.rename({'orderId': 'TVTIC'})
            if 'displayName' in df.columns:
                df = df.drop('displayName')
            df_list.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    df = pl.concat(df_list, how="diagonal")
    print("Total raw rows:", df.shape[0])

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

    df = df.with_columns([
        pl.col("price").cast(pl.Float64, strict=False),
        pl.col("size").cast(pl.Int64, strict=False),
    ])
    df = df.drop_nulls(["price", "size", "tradeTime"])

    df = df.sort('publishedTime').unique(subset=['TVTIC'], keep='last')
    print("Unique TVTIC rows:", df.shape[0])

    # Group by identical timestamp to aggregate split executions and amendments/cancellations
    df = df.with_columns([
        pl.col("tradeTime").dt.truncate("1s").alias("trade_sec")
    ])

    df = df.with_columns(
        pl.when(pl.col("flags").str.contains("CANC"))
        .then(pl.col("size") * -1)
        .otherwise(pl.col("size"))
        .alias("size")
    )

    print("Consolidating AMND/CANC...")
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

    print(f"Consolidated rows: {grouped.shape[0]}")

    # Apply ISIN filter within the timeframe
    # But wait, first we need to make sure we only filter rows IN the timeframe.
    # If the user has other files in d:\lsx, we must filter tradeTime before counting!
    start_d = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_d = datetime.strptime(end_date_str, "%Y-%m-%d")

    # Convert to timezone aware datetime to filter
    start_dt = start_d.replace(tzinfo=pytz.timezone("Europe/Berlin"))
    # Add 1 day to end_date to include the whole day
    import datetime as dt
    end_dt = (end_d + dt.timedelta(days=1)).replace(tzinfo=pytz.timezone("Europe/Berlin"))

    grouped = grouped.filter((pl.col("tradeTime") >= start_dt) & (pl.col("tradeTime") < end_dt))
    print(f"Rows strictly within {start_date_str} to {end_date_str}: {grouped.shape[0]}")

    final_df = filter_isins(grouped, start_date_str, end_date_str)
    return final_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=r"d:\lsx", help="Input directory")
    parser.add_argument("--start", default="2023-08-17", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=str(date.today()), help="End date YYYY-MM-DD")
    args = parser.parse_args()

    print(f"Checking directory: {args.dir}")

    # 1. Extract mapping
    extract_names(args.dir)

    # 2. Process data
    df = process_transactions(args.dir, args.start, args.end)

    if df is not None:
        # 4. Save to parquet with fastparquet engine/pyarrow for timezone safety
        out_file = "consolidated_transactions.parquet"
        print(f"Saving final dataset ({df.shape[0]} rows) to {out_file}...")
        df.write_parquet(out_file, use_pyarrow=True)
        print("Done!")

if __name__ == "__main__":
    main()
