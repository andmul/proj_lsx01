import os
import glob
import datetime
import holidays
import polars as pl
import argparse

def get_previous_trading_day(d, de_holidays):
    """Returns the most recent trading day strictly before the given date."""
    prev_d = d - datetime.timedelta(days=1)
    while prev_d.weekday() >= 5 or prev_d in de_holidays:
        prev_d -= datetime.timedelta(days=1)
    return prev_d

def analyze_file_dates(directory="."):
    # Initialize German holidays (cover a wide range)
    de_holidays = holidays.DE(years=list(range(2020, 2030)))

    files = glob.glob(os.path.join(directory, 'lsxtradesyesterday_*.csv'))
    if not files:
        print(f"No files found matching 'lsxtradesyesterday_*.csv' in {directory}")
        return

    print(f"{'Filename':<35} | {'Min Date':<10} | {'Max Date':<10} | {'Max Date First Row':<18} | {'Filename Date':<13} | {'Expected Date (-1 TRD)':<22} | {'Matches Expected?'}")
    print("-" * 145)

    files.sort()

    total_checked = 0
    total_issues = 0
    total_perfect = 0

    for f in files:
        filename = os.path.basename(f)
        total_checked += 1
        has_issue = False
        issue_msg = ""

        # 1. Parse filename date
        file_date_str = filename.split('_')[1].split('.')[0]
        try:
            file_date = datetime.datetime.strptime(file_date_str, "%Y%m%d").date()
        except ValueError:
            has_issue = True
            issue_msg = "Invalid filename date format"

        if not has_issue:
            # Calculate the expected date (-1 trading day)
            expected_date = get_previous_trading_day(file_date, de_holidays)

            # 2. Read file to find actual min/max dates and position
            try:
                # We just need tradeTime, ignoring errors for malformed lines
                df = pl.read_csv(f, separator=";", decimal_comma=True, ignore_errors=True, columns=["tradeTime"])
            except Exception as e:
                try:
                    # If column extraction fails because of schema or header issues, load all and select
                    df = pl.read_csv(f, separator=";", decimal_comma=True, ignore_errors=True)
                    if "tradeTime" not in df.columns:
                        has_issue = True
                        issue_msg = "'tradeTime' column not found"
                    else:
                        df = df.select(["tradeTime"])
                except Exception as e:
                    has_issue = True
                    issue_msg = f"Failed to read: {e}"

        if not has_issue:
            if df.shape[0] == 0:
                has_issue = True
                issue_msg = "File is empty or could not be parsed"

        if not has_issue:
            # Extract just the date part YYYY-MM-DD (first 10 chars)
            df = df.with_columns(pl.col("tradeTime").str.slice(0, 10).alias("date_str"))

            # Find min and max
            # Polars handles string min/max chronologically if formatted YYYY-MM-DD
            min_date_str = df.select(pl.col("date_str").min()).item()
            max_date_str = df.select(pl.col("date_str").max()).item()

            if not min_date_str or not max_date_str:
                has_issue = True
                issue_msg = "Could not parse dates"

        if not has_issue:
            # Find the first occurrence (row index) of the max date
            # Create an index column to find the exact row position
            df = df.with_row_index("row_num")
            first_occurrence = df.filter(pl.col("date_str") == max_date_str).select("row_num").head(1).item()

            # Add 1 because rows are usually 1-indexed for humans (or 2 if counting CSV header)
            # We will state "Row X (0-indexed data row)"
            row_pos_display = f"Row {first_occurrence} (data)"

            # Check if matches expected
            try:
                max_date_obj = datetime.datetime.strptime(max_date_str, "%Y-%m-%d").date()
                matches = (max_date_obj == expected_date)
                if not matches:
                    has_issue = True
                matches_str = "YES" if matches else "NO"
            except ValueError:
                has_issue = True
                matches_str = "ERROR"

        if has_issue:
            total_issues += 1
            if issue_msg:
                print(f"{filename:<35} | {issue_msg}")
            else:
                print(f"{filename:<35} | {min_date_str:<10} | {max_date_str:<10} | {row_pos_display:<18} | {file_date_str:<13} | {str(expected_date):<22} | {matches_str}")
        else:
            total_perfect += 1

    print("-" * 145)
    print(f"Summary: Checked {total_checked} files | Perfect: {total_perfect} | Issues Found: {total_issues}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze min/max dates within LSX CSV files.")
    parser.add_argument("--dir", default=".", help="Directory containing the CSV files to scan.")
    args = parser.parse_args()

    analyze_file_dates(args.dir)
