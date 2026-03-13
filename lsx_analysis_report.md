# Analysis Report: LSX Trades Data

## 1. Overview and Files Found
- I successfully connected to the FTP server (`ngcobalt350.manitu.net:23`) and discovered 437 files beginning with `lsxtrades`.
- The files are typically massive, ranging from ~30MB to ~120MB per day, representing tick-level trading data.
- Two sample files were downloaded for detailed analysis:
  - `lsxtradesyesterday_20250427.csv` (~54MB)
  - `lsxtradesyesterday_20251024.csv` (~61MB)

## 2. Data Schema and Characteristics
The files are semicolon-separated (`separator=";"`) and contain the following columns:
1. `isin`: The unique asset identifier.
2. `tradeTime`: Timestamp of the trade in UTC (e.g., `2025-04-25T05:30:00.436000Z`).
3. `quotation`: Type of quote (e.g., `MONE`).
4. `price`: Trade execution price formatted with commas instead of periods (e.g., `167,5000`).
5. `currency`: Currency of the quote (e.g., `EUR`).
6. `size`: Volume/number of shares traded.
7. `TVTIC`: Transaction identification number (e.g., `HAMLUS0231351067202504250530005533868A0000002`).
8. `mic`: Market Identifier Code (e.g., `HAML;HAMN`).
9. `flags`: Execution or correction flags (e.g., `ALGO;`, `CANC;`, `AMND;`).
10. `publishedTime`: Secondary timestamp when the trade was published.

## 3. Findings regarding Transaction Numbers and Corrections
You correctly noted that the transaction number (`TVTIC` column) is **not unique**. This occurs because subsequent rows with the same transaction number (`TVTIC`) represent corrections or updates to the prior trade.
- **Flags indicate corrections:**
  - Standard trades typically have flags like `ALGO;`.
  - Corrections are indicated by flags like `CANC;` (Cancellation) or `AMND;` (Amendment).
- **Handling pattern:** To process this accurately, we have to group rows by `TVTIC`. If a `CANC;` flag appears, we should negate the original size or discard the prior tick with the same `TVTIC`.

## 4. How to Proceed with Massive Datasets
Since processing 437 files (each up to 120MB) directly would cause Out-Of-Memory (OOM) errors, the best approach is an **Out-of-Core Map-Reduce Pipeline** (which is what we have been building):
1. **Map (Chunk Processing):** Use `polars.scan_csv()` or `polars.read_csv()` to lazily parse individual files one by one. Handle the `price` column by replacing commas with dots and casting to float. Apply the time-filtering logic (e.g., focusing only on morning ticks 07:00-08:45 Berlin time or filtering specific highly liquid ISINs).
2. **Handle Corrections:** In this step, group by `TVTIC` and aggregate the `size` (by negating sizes for `CANC;` records) so we get the *net* execution size for a given transaction.
3. **Reduce:** Save the filtered and corrected results of each file into highly-compressed `.parquet` chunks.
4. **Final Aggregation:** Run a final pass over the `.parquet` files using `polars.scan_parquet().sink_parquet()` to aggregate everything into a single consolidated dataset for ML modeling.

Our current scripts (`process_all_lsx.py` and `predict_clusters.py`) have been configured exactly this way, using streaming and out-of-core evaluation to prevent memory issues.
