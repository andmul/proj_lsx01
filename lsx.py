import polars as pl
import glob,os
from pathlib import Path
from datetime import datetime
import pytz
from datetime import time
from datetime import date
import pandas as pd
import json
#import pyarrow
import numpy as np
import paramiko
import argparse
import stat

# SFTP Configuration
SFTP_HOST = os.environ.get("LSX_FTP_HOST", "ngcobalt350.manitu.net")
SFTP_PORT = int(os.environ.get("LSX_FTP_PORT", 23))
SFTP_USER = os.environ.get("LSX_FTP_USER")
SFTP_PASS = os.environ.get("LSX_FTP_PASS")

def download_files(start_date=None, end_date=None, download_dir="."):
    """
    Connects via SFTP, lists files matching `lsxtradesyesterday_*.csv`,
    filters them by the provided timeframe, and downloads them.
    Dates should be strings in format YYYYMMDD.
    """
    if not SFTP_USER or not SFTP_PASS:
        print("Error: SFTP credentials not provided. Set LSX_FTP_USER and LSX_FTP_PASS environment variables.")
        return []

    transport = paramiko.Transport((SFTP_HOST, SFTP_PORT))
    transport.connect(username=SFTP_USER, password=SFTP_PASS)
    sftp = paramiko.SFTPClient.from_transport(transport)

    print("Fetching file list from server...")
    all_files = sftp.listdir()
    target_files = [f for f in all_files if f.startswith("lsxtradesyesterday_") and f.endswith(".csv")]
    target_files.sort()

    if not target_files:
        print("No files found matching the pattern.")
        sftp.close()
        transport.close()
        return []

    import csv

    def get_date_from_sftp_file(sftp_client, filename):
        """Reads the first 10 rows of the file on the SFTP server to determine its true date."""
        try:
            with sftp_client.open(filename, 'r') as remote_f:
                head = []
                for _ in range(12):  # Header + 10 rows + 1 extra
                    line = remote_f.readline()
                    if not line: break
                    head.append(line)

            reader = csv.DictReader(head, delimiter=';')
            dates_found = set()
            rows_read = 0
            for row in reader:
                if rows_read >= 10:
                    break
                if 'tradeTime' in row and row['tradeTime']:
                    # Extract '2024-04-03' and convert to '20240403'
                    date_part = row['tradeTime'][:10].replace('-', '')
                    dates_found.add(date_part)
                rows_read += 1

            if len(dates_found) == 0:
                print(f"Error: Could not find any valid tradeTime in the first 10 rows of {filename}")
                return None
            elif len(dates_found) > 1:
                print(f"Error: Multiple dates found in the first 10 rows of {filename}. Dates found: {dates_found}")
                return None
            else:
                return list(dates_found)[0]

        except Exception as e:
            print(f"Error reading file {filename} from SFTP: {e}")
            return None

    # Determine chronological bounds by reading content of first and last file
    first_date = get_date_from_sftp_file(sftp, target_files[0])
    last_date = get_date_from_sftp_file(sftp, target_files[-1])

    print(f"Server data (based on content) ranges from {first_date} to {last_date}")

    downloaded_paths = []
    print(f"Filtering {len(target_files)} files by timeframe...")
    for f in target_files:
        # Instead of scanning every single file on the server (which could take a long time over SFTP for 1000+ files),
        # we check the local cache first. If it's cached, we parse the local file date.
        # But if it's not cached, we read the remote header.
        local_path = os.path.join(download_dir, f)

        file_date = None
        if os.path.exists(local_path):
            # Parse from local
            try:
                with open(local_path, 'r', encoding='utf-8') as local_f:
                    head = [next(local_f) for _ in range(12)]
                reader = csv.DictReader(head, delimiter=';')
                dates_found = set()
                rows = 0
                for row in reader:
                    if rows >= 10: break
                    if 'tradeTime' in row and row['tradeTime']:
                        dates_found.add(row['tradeTime'][:10].replace('-', ''))
                    rows += 1
                if len(dates_found) == 1:
                    file_date = list(dates_found)[0]
            except StopIteration:
                pass

        if not file_date:
            file_date = get_date_from_sftp_file(sftp, f)

        if not file_date:
            continue

        # Check against timeframe
        if start_date and file_date < start_date:
            continue
        if end_date and file_date > end_date:
            continue

        downloaded_paths.append(local_path)
        if os.path.exists(local_path):
            print(f"File {f} ({file_date}) already exists locally, skipping download.")
            continue

        print(f"Downloading {f} ({file_date})...")
        sftp.get(f, local_path)

    sftp.close()
    transport.close()
    return downloaded_paths

##MAIN

parser = argparse.ArgumentParser(description="Process LSX trade files.")
parser.add_argument("--start", help="Start date (YYYYMMDD)", default=None)
parser.add_argument("--end", help="End date (YYYYMMDD)", default=None)
parser.add_argument("--skip-download", action="store_true", help="Skip SFTP download and process local files only.")
args = parser.parse_args()

# Step 1: get files
if not args.skip_download:
    files = download_files(args.start, args.end)
else:
    files = glob.glob(r'lsxtradesyesterday_*.csv')
# Step 2: Read all CSVs into Polars DataFrames

df_list = []
for f in files:
    try:
        # ignore_errors=True helps skip rows with corrupted quoting
        df_list.append(pl.read_csv(f, infer_schema_length=100,separator=";",decimal_comma=True, ignore_errors=True))
    except Exception as e:
        print(f"Warning: could not read {f}: {e}")

if not df_list:
    print("No files to process. Exiting.")
    exit(0)

#delete name column from all frames
for index,dlist in enumerate(df_list):
        # Rename orderId to TVTIC if it exists to ensure consistency across files
        if 'orderId' in df_list[index].columns:
            df_list[index] = df_list[index].rename({'orderId': 'TVTIC'})

        if len(df_list[index].columns) == 11:
            df_list[index]=df_list[index].drop('displayName')

# Step 3: Concatenate all data into one DataFrame
print('start concatenation....')
df = pl.concat(df_list, how="diagonal")
print('...end concatenation')
# Step 4: Column conversions

# -- Convert 'tradeTime' and 'publishedTime' to pl.Datetime
# Parse the tradeTime column to datetime and convert to Berlin time
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
# -- Cast 'price' to float and 'size' to int
df = df.with_columns([
    pl.col("price").cast(pl.Float64, strict=False),
    pl.col("size").cast(pl.Int64, strict=False),
])
df = df.drop_nulls(["price", "size", "tradeTime"])

# -- Clean string columns
str_columns = ['isin', 'displayName', 'quotation', 'currency', 'TVTIC', 'mic', 'flags']
for col in str_columns:
    if col in df.columns:
        df = df.with_columns(pl.col(col).cast(pl.Utf8).fill_null("").str.strip_chars())
 
print('Done with structure cleanup')

df_sorted=df.sort('publishedTime')
print('all sorted, now make it unique based on TVTIC...')
df=df_sorted.unique(subset=['TVTIC'],keep='last')

print('Grouping transactions by ISIN and price within 1 second...')
# Consolidate TVTIC, group CANC/AMND transactions to deduct/add amounts.
df = df.with_columns([
    pl.col("tradeTime").dt.truncate("1s").alias("trade_sec")
])

# For cancellations, size is negated to subtract from the group
# Note regarding AMND flags: As requested, AMND sizes are treated identically to
# original trades. They simply inject the stated replacement 'size' into the timeframe aggregate
# so that the ultimate SUM(size) reaches the mathematically desired target.
df = df.with_columns(
    pl.when(pl.col("flags").str.contains("CANC"))
    .then(pl.col("size") * -1)
    .otherwise(pl.col("size"))
    .alias("size")
)

# Merge transactions for the same isin and same price within 1 second
grouped = df.group_by(["isin", "trade_sec", "price"]).agg([
    pl.col("size").sum().alias("total_size"),
    pl.col("tradeTime").first().alias("tradeTime"),
    pl.col("publishedTime").first().alias("publishedTime"),
    pl.col("TVTIC").first().alias("TVTIC"),
    pl.col("flags").first().alias("flags"),
    pl.col("mic").first().alias("mic"),
    pl.col("currency").first().alias("currency"),
    pl.col("quotation").first().alias("quotation")
])

# Remove groups where total amount of shares ended up <= 0 (e.g., fully cancelled)
grouped = grouped.filter(pl.col("total_size") > 0)

print("Filtering for ISINs with >= 10 transactions per day and >= 5000 Euro volume...")
grouped = grouped.with_columns([
    pl.col("trade_sec").dt.truncate("1d").alias("trade_day"),
    (pl.col("total_size") * pl.col("price")).alias("volume")
])

# Find all ISINs with >= 10 transactions per day and a volume of >= 5000 Euro on ANY day in the files
daily_stats = grouped.group_by(["isin", "trade_day"]).agg([
    pl.col("total_size").len().alias("tx_count"),
    pl.col("volume").sum().alias("daily_volume")
])

valid_isins = daily_stats.filter(
    (pl.col("tx_count") >= 10) &
    (pl.col("daily_volume") >= 5000)
).select("isin").unique()

# Filter the aggregated dataset based on valid ISINs
final_df = grouped.join(valid_isins, on="isin", how="inner")

# Drop the temporary grouping columns
final_df = final_df.drop(["trade_sec", "trade_day", "volume"])

print(f"Done. Writing {final_df.shape[0]} rows to parquet.")
final_df.write_parquet('raw_output.parquet')
       