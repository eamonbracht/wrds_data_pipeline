import pandas as pd
import numpy as np
import datetime
import os
import wrds
import gc
import sys


# Connect to WRDS
db = wrds.Connection()

# Load S&P500 permno universe
sp500ccm = pd.read_csv("sp500ccm_filtered.csv.gz", compression='gzip', usecols=["permno", "ticker", "start", "ending", "date"])

# Convert date columns to datetime
sp500ccm['start'] = pd.to_datetime(sp500ccm['start'])
sp500ccm['ending'] = pd.to_datetime(sp500ccm['ending'])

# Define TAQ pull + resample function
def get_taq_data(permnos, day, frequency='1s', chunk_size=100, start_time=None, end_time=None):
    if start_time is None:
        start_time = datetime.time(9, 30)
    if end_time is None:
        end_time = datetime.time(16, 0)

    all_chunks = []
    for i in range(0, len(permnos), chunk_size):
        chunk = permnos[i:i + chunk_size]

        lookup = (
            sp500ccm[
                (sp500ccm['permno'].isin(chunk)) &
                (sp500ccm['date'] == day.strftime('%Y-%m-%d'))
            ]
            .drop_duplicates('permno')[['permno', 'ticker']]
            .dropna()
        )

        if lookup.empty:
            continue

        # Split into sym_root and sym_suffix
        lookup['sym_root'] = lookup['ticker'].str.split('.').str[0]
        lookup['sym_suffix'] = lookup['ticker'].str.split('.').str[1].fillna('').astype(str)

        # Build WHERE clause
        where_clauses = []
        for root, suffix in lookup[['sym_root', 'sym_suffix']].drop_duplicates().values:
            if suffix == '':
                clause = f"(sym_root = '{root}' AND sym_suffix IS NULL)"
            else:
                clause = f"(sym_root = '{root}' AND sym_suffix = '{suffix}')"
            where_clauses.append(clause)
        where_clause_str = " OR ".join(where_clauses)


        # SQL query
        query = f"""
            SELECT date, time_m, sym_root, sym_suffix, best_bid, best_ask
            FROM taqmsec.complete_nbbo_{day.strftime('%Y%m%d')}
            WHERE ({where_clause_str})
            AND time_m BETWEEN %(start_time)s AND %(end_time)s
        """
        params = {
            'start_time': start_time.strftime('%H:%M:%S'),
            'end_time': end_time.strftime('%H:%M:%S')
        }

        try:
            print("start query")
            data = db.raw_sql(query, params=params)
            print("end query")
            data['price'] = (data['best_bid'] + data['best_ask']) / 2
            data['datetime'] = pd.to_datetime(data['date'].astype(str) + ' ' + data['time_m'].astype(str), errors='coerce')
            data = data.dropna(subset=['datetime'])

            # Normalize suffix and merge permno
            data['sym_suffix'] = data['sym_suffix'].fillna('').astype(str)
            lookup['sym_suffix'] = lookup['sym_suffix'].fillna('').astype(str)
            data = data.merge(lookup, on=['sym_root', 'sym_suffix'], how='left')

            data = data[['datetime', 'permno', 'price']]
            data = data.groupby(['datetime', 'permno'], as_index=False).mean()

            resampled_list = []
            for permno in data['permno'].dropna().unique():
                perm_df = data[data['permno'] == permno].copy()
                perm_df.set_index('datetime', inplace=True)
                perm_resampled = perm_df['price'].resample(frequency).ffill().rename(str(permno))
                resampled_list.append(perm_resampled)

            if resampled_list:
                resampled = pd.concat(resampled_list, axis=1)
                all_chunks.append(resampled)

            del data, resampled_list, lookup
            gc.collect()

        except Exception as e:
            print(f"Error pulling chunk {i}-{i+chunk_size} on {day}: {e}")
            continue

    if all_chunks:
        final_df = pd.concat(all_chunks, axis=1)
        final_df = final_df.between_time("09:30:00", "16:00:00")
        return final_df
    else:
        return pd.DataFrame()

# === MAIN LOOP ===
start_date = pd.Timestamp(sys.argv[1])
end_date = pd.Timestamp(sys.argv[2])

current = start_date
while current <= end_date:
    if current.weekday() >= 5:  # Skip weekends
        current += pd.Timedelta(days=1)
        continue

    print(f"Processing {current.strftime('%Y-%m-%d')}...")
    permnos = sp500ccm[(sp500ccm['start'] <= current) & (sp500ccm['ending'] >= current)]['permno'].unique()

    if len(permnos) == 0:
        print(f"No permnos for {current.strftime('%Y-%m-%d')}")
        current += pd.Timedelta(days=1)
        continue

    df = get_taq_data(permnos, current)

    if df.empty:
        print(f"No data for {current.strftime('%Y-%m-%d')}")
        current += pd.Timedelta(days=1)
        continue

    year = current.year
    month = current.month
    output_dir = f"data/{year}/{month:02d}"
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"taq_resampled_{current.strftime('%Y-%m-%d')}.csv.gz")
    df.to_csv(output_file, compression='gzip')
    print(f"Saved {output_file}")
    del df
    gc.collect()

    current += pd.Timedelta(days=1)

print("All done.")