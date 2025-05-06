
# TAQ Correlation Matrix Generator

This script connects to the WRDS database to pull high-frequency TAQ (Trade and Quote) data for S&P 500 stocks, calculates correlation matrices over specified time intervals, and saves them as CSV files.

---

## Features

- Pulls NBBO (best bid/ask) TAQ data via WRDS
- Supports both intra-day (e.g., `1h`, `4h`) and multi-day (e.g., `2d`, `5d`) intervals
- Resamples price data using a specified frequency (e.g., `250ms`, `5min`)
- Computes correlation matrices between selected tickers
- Outputs each matrix as a separate `.csv` file

---

## Usage

From the command line:
```bash
python3 correlation_matrix_exporter.py <start_date> <end_date> <interval> <sampling_frequency>
```
### Example:
```bash
python3 correlation_matrix_exporter.py 2017-12-19 2017-12-22 4h 1min
```
### Arguments:
- `start_date`: Start of the date range (format: `YYYY-MM-DD`)
- `end_date`: End of the date range (format: `YYYY-MM-DD`)
- `interval`: Time period for each correlation matrix (e.g., `4h`, `1d`, `3d`)
- `sampling_frequency`: Resample frequency for price data (e.g., `1min`, `5min`, `30s`)

---

## Output

Each correlation matrix is saved as a CSV file, e.g.:
```bash
correlation_matrix_20171219_093000_to_20171219_133000.csv
```

Each row and column represents a PERMNO (stock ID). Values are Pearson correlation coefficients of resampled midquote prices.

---

## Requirements

- Python 3.7+
- WRDS credentials
- Packages:
  - `wrds`
  - `pandas`
  - `numpy`
  - `tqdm`

