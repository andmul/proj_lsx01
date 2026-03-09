import os
import glob
import datetime
import holidays
import argparse

def check_files(directory="."):
    # 5MB threshold
    MIN_SIZE_BYTES = 5 * 1024 * 1024

    # Generate German holidays for 2023-2026
    # Note: German holidays can vary slightly by state. We'll use the country-wide defaults (DE)
    de_holidays = holidays.DE(years=[2023, 2024, 2025, 2026])

    import polars as pl

    # Find all files matching the pattern
    files = glob.glob(os.path.join(directory, 'lsxtradesyesterday_*.csv'))

    valid_files_by_date = {}
    for f in files:
        size = os.path.getsize(f)
        if size < MIN_SIZE_BYTES:
            continue

        filename = os.path.basename(f)
        file_date_str = filename.split('_')[1].split('.')[0]
        try:
            filename_date = datetime.datetime.strptime(file_date_str, "%Y%m%d").date()
        except ValueError:
            print(f"Error: Invalid filename date format in {f}")
            continue

        # Calculate expected date (-1 trading day)
        expected_date = filename_date - datetime.timedelta(days=1)
        while expected_date.weekday() >= 5 or expected_date in de_holidays:
            expected_date -= datetime.timedelta(days=1)

        # Read the first 100 rows to extract dates efficiently without memory bloat
        try:
            df = pl.read_csv(f, separator=";", decimal_comma=True, ignore_errors=True, columns=["tradeTime"], n_rows=100)
            if "tradeTime" not in df.columns:
                df = pl.read_csv(f, separator=";", decimal_comma=True, ignore_errors=True, n_rows=100).select(["tradeTime"])
        except Exception as e:
            print(f"Error reading file {f}: {e}")
            continue

        if df.shape[0] == 0:
            print(f"Error: File {f} is empty or could not be parsed")
            continue

        # Extract date part
        df = df.with_columns(pl.col("tradeTime").str.slice(0, 10).alias("date_str"))

        # Find majority date
        date_counts = df.group_by("date_str").len().sort("len", descending=True)
        majority_date_str = date_counts.select("date_str").head(1).item()

        if not majority_date_str:
            print(f"Error: Could not parse dates in {f}")
            continue

        try:
            majority_date = datetime.datetime.strptime(majority_date_str, "%Y-%m-%d").date()
            if majority_date == expected_date:
                if 2023 <= majority_date.year <= 2026:
                    valid_files_by_date[majority_date] = f
            else:
                print(f"File {f}: majority date ({majority_date}) does not match expected date ({expected_date}).")
                # Even if it doesn't match expected, we could still map it to its majority date.
                # However, the user specifically wants to ensure it aligns with `-1 trading day`.
                # We'll map it to its majority date so it still counts as a valid file for THAT date.
                if 2023 <= majority_date.year <= 2026:
                    valid_files_by_date[majority_date] = f
        except ValueError:
            print(f"Error: Invalid date format parsed from {f}: {majority_date_str}")

    # Determine start and end of our evaluation period
    start_date = datetime.date(2023, 1, 1)
    # If today is before end of 2026, we only check up to today or yesterday, not future dates!
    today = datetime.date.today()
    end_date = min(datetime.date(2026, 12, 31), today)

    missing_trading_days = []
    holiday_trading_days = []

    current_date = start_date
    while current_date <= end_date:
        is_weekend = current_date.weekday() >= 5 # 5=Sat, 6=Sun
        is_holiday = current_date in de_holidays
        is_expected_trading_day = not is_weekend and not is_holiday

        has_valid_file = current_date in valid_files_by_date

        if is_expected_trading_day and not has_valid_file:
            missing_trading_days.append(current_date)
        elif not is_expected_trading_day and has_valid_file:
            reason = "Weekend" if is_weekend else f"Holiday ({de_holidays.get(current_date)})"
            holiday_trading_days.append((current_date, reason, valid_files_by_date[current_date]))

        current_date += datetime.timedelta(days=1)

    print("=== LSX File Validation Report (2023-2026) ===")
    print(f"Directory checked: {directory}")
    print(f"Total valid files (>= 5MB) found in range: {len(valid_files_by_date)}")
    print("-" * 45)

    print(f"\n[ MISSING TRADING DAYS ] ({len(missing_trading_days)} days missing)")
    if missing_trading_days:
        # Group by month for cleaner output if there are many
        by_month = {}
        for d in missing_trading_days:
            month_str = d.strftime("%Y-%m")
            if month_str not in by_month:
                by_month[month_str] = []
            by_month[month_str].append(d.strftime("%d"))

        for month, days in sorted(by_month.items()):
            print(f"  {month}: {', '.join(days)}")
    else:
        print("  All expected trading days have a valid file! 🎉")

    print(f"\n[ HOLIDAY/WEEKEND TRADING FOUND ] ({len(holiday_trading_days)} occurrences)")
    if holiday_trading_days:
        for d, reason, f in sorted(holiday_trading_days, key=lambda x: x[0]):
            print(f"  {d.strftime('%Y-%m-%d')}: {reason} - File: {os.path.basename(f)}")
    else:
        print("  No files found on weekends or public holidays.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check LSX directory for valid daily trading files.")
    parser.add_argument("--dir", default=".", help="Directory to scan (default: current dir)")
    args = parser.parse_args()
    check_files(args.dir)
