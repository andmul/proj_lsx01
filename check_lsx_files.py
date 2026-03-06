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

    # Find all files matching the pattern
    files = glob.glob(os.path.join(directory, 'lsxtradesyesterday_*.csv'))

    valid_files_by_date = {}
    for f in files:
        size = os.path.getsize(f)
        if size < MIN_SIZE_BYTES:
            continue

        # We need to extract date from the first 10 data rows of the CSV
        # We use standard library to avoid overhead of loading pandas/polars just for headers
        import csv
        dates_found = set()
        try:
            with open(f, 'r', encoding='utf-8') as csvfile:
                # Use a reader that ignores strict quotes in case of errors
                reader = csv.DictReader(csvfile, delimiter=';')

                rows_read = 0
                for row in reader:
                    if rows_read >= 10:
                        break
                    if 'tradeTime' in row and row['tradeTime']:
                        # Example tradeTime: 2024-04-03T05:30:01.109000Z
                        # Extract the date part: YYYY-MM-DD
                        date_part = row['tradeTime'][:10]
                        dates_found.add(date_part)
                    rows_read += 1

        except Exception as e:
            print(f"Error reading file {f}: {e}")
            continue

        if len(dates_found) == 0:
            print(f"Error: Could not find any valid tradeTime in the first 10 rows of {f}")
        elif len(dates_found) > 1:
            print(f"Error: Multiple dates found in the first 10 rows of {f}. Dates found: {dates_found}")
        else:
            date_str = list(dates_found)[0]
            try:
                file_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
                if 2023 <= file_date.year <= 2026:
                    valid_files_by_date[file_date] = f
            except ValueError:
                print(f"Error: Invalid date format parsed from {f}: {date_str}")

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
