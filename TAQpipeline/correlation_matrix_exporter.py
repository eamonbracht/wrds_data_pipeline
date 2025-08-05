import wrds
import pandas as pd
import numpy as np
import tqdm
import datetime
import sys

# Connect to WRDS
db = wrds.Connection(wrds_username='akulshar')

# Pull S&P 500 definition and daily returns
sp500 = db.raw_sql("""
    SELECT a.*, b.date, b.ret
    FROM crsp.dsp500list AS a
    JOIN crsp.dsf AS b ON a.permno = b.permno
    WHERE b.date BETWEEN a.start AND a.ending
      AND b.date BETWEEN '01/01/2004' AND '12/31/2020'
    ORDER BY b.date;
""", date_cols=['start', 'ending', 'date'])
print("sp500 done")

# Pull firm metadata
mse = db.raw_sql("""
    SELECT comnam, cusip, ncusip, namedt, nameendt, 
           permno, shrcd, exchcd, hsiccd, ticker, tsymbol
    FROM crsp.msenames;
""", date_cols=['namedt', 'nameendt'])
print("mse done")

# Filter firm metadata by valid date range
sp500 = (
    sp500
    .merge(mse, how='left', on='permno')
    .query('date >= namedt and date <= nameendt')
)

# Pull linking table for Compustat-CRSP linking
ccm = db.raw_sql("""
    SELECT gvkey, liid AS iid, lpermno AS permno, 
           linktype, linkprim, linkdt, linkenddt
    FROM crsp.ccmxpf_linktable
    WHERE SUBSTR(linktype, 1, 1) = 'L'
      AND linkprim IN ('C', 'P');
""", date_cols=['linkdt', 'linkenddt'])
print("ccm done")
ccm['linkenddt'] = ccm['linkenddt'].fillna(pd.to_datetime('today'))

sp500ccm = (
    sp500
    .merge(ccm, how='left', on='permno')
    .query('date >= linkdt and date <= linkenddt')
    .drop(columns=['namedt', 'nameendt', 'linktype', 'linkprim', 'linkdt', 'linkenddt'])
)

def get_taq_data(permnos, day, frequency='1h', start_time=None, end_time=None):
    if start_time is None:
        start_time = datetime.time(9, 30)
    if end_time is None:
        end_time = datetime.time(16, 0)

    lookup = (
        sp500ccm[
            (sp500ccm['permno'].isin(permnos)) &
            (sp500ccm['start'] <= day) &
            (sp500ccm['ending'] >= day)
        ]
        .drop_duplicates('permno')
        [['permno', 'ticker']]
        .dropna()
    )

    lookup['sym_root'] = lookup['ticker'].str.split('.').str[0]
    lookup['sym_suffix'] = lookup['ticker'].str.split('.').str[1].fillna('').astype(str)

    if lookup.empty:
        return pd.DataFrame()

    where_clauses = []
    for root, suffix in lookup[['sym_root', 'sym_suffix']].drop_duplicates().values:
        if suffix == '':
            clause = f"(sym_root = '{root}' AND sym_suffix IS NULL)"
        else:
            clause = f"(sym_root = '{root}' AND sym_suffix = '{suffix}')"
        where_clauses.append(clause)
    where_clause_str = " OR ".join(where_clauses)

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
        data = db.raw_sql(query, params=params)
        data['price'] = (data['best_bid'] + data['best_ask']) / 2
        data['datetime'] = pd.to_datetime(data['date'].astype(str) + ' ' + data['time_m'].astype(str), errors='coerce')
        data = data.dropna(subset=['datetime'])

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

        if not resampled_list:
            return pd.DataFrame()

        resampled = pd.concat(resampled_list, axis=1)
        resampled = resampled.between_time("09:30:00", "16:00:00")
        return resampled

    except Exception:
        return pd.DataFrame()

def create_correlation_matrices(start_date, end_date, interval, sampling_frequency):
    correlation_matrices = []
    interval_timedelta = pd.to_timedelta(interval)
    market_open = datetime.time(9, 30)
    market_close = datetime.time(16, 0)
    business_days = pd.bdate_range(start=start_date, end=end_date)

    if interval_timedelta < datetime.timedelta(days=1):
        with tqdm.tqdm(desc="Intra-day intervals") as pbar:
            for day in business_days:
                try:
                    permnos = sp500ccm[
                        (sp500ccm['start'] <= day) & (sp500ccm['ending'] >= day)
                    ]['permno'].unique()
                    permnos = sorted(permnos)[:3]
                    if not permnos:
                        continue

                    current_dt = datetime.datetime.combine(day, market_open)
                    end_dt = datetime.datetime.combine(day, market_close)

                    while current_dt < end_dt:
                        next_dt = current_dt + interval_timedelta
                        if next_dt.time() > market_close:
                            next_dt = datetime.datetime.combine(day, market_close)

                        data = get_taq_data(
                            permnos=list(permnos),
                            day=day,
                            frequency=sampling_frequency,
                            start_time=current_dt.time(),
                            end_time=next_dt.time()
                        )

                        if not data.empty:
                            corr = data.corr()
                            correlation_matrices.append((day, current_dt.time(), next_dt.time(), corr))

                        current_dt = next_dt
                        pbar.update(1)

                except Exception:
                    continue

    else:
        interval_days = interval_timedelta.days
        with tqdm.tqdm(desc="Multi-day intervals") as pbar:
            current_index = 0
            while current_index + interval_days <= len(business_days):
                chunk_days = business_days[current_index:current_index + interval_days]
                current_index += interval_days

                all_data = []
                tickers_by_day = []

                for day in chunk_days:
                    try:
                        permnos = sp500ccm[
                            (sp500ccm['start'] <= day) & (sp500ccm['ending'] >= day)
                        ]['permno'].unique()
                        permnos = sorted(permnos)[:3]
                        if not permnos:
                            continue

                        data = get_taq_data(
                            permnos=list(permnos),
                            day=day,
                            frequency=sampling_frequency
                        )

                        if not data.empty:
                            all_data.append(data)
                            tickers_by_day.append(set(data.columns))

                    except Exception:
                        continue

                if all_data and tickers_by_day:
                    common_tickers = set.intersection(*tickers_by_day)
                    filtered = [df[list(common_tickers)] for df in all_data]
                    combined = pd.concat(filtered)
                    correlation_matrices.append((chunk_days[0], chunk_days[-1], None, combined.corr()))

                pbar.update(1)

    return correlation_matrices

def main():
    if len(sys.argv) <= 4:
        print("Usage: python script.py <start_date> <end_date> <sampling_frequency>")
        sys.exit(1)

    start_date = sys.argv[1]
    end_date = sys.argv[2]
    interval = sys.argv[3]
    frequency = sys.argv[4]

    matrices = create_correlation_matrices(start_date, end_date, interval=interval, sampling_frequency=frequency)

    for i, (start, end, extra, matrix) in enumerate(matrices):
        if end:
            fname = f"correlation_matrix_{start.strftime('%Y%m%d_%H%M%S')}_to_{end.strftime('%Y%m%d_%H%M%S')}.csv"
        elif extra:
            fname = f"correlation_matrix_{start.strftime('%Y%m%d_%H%M%S')}_{extra.strftime('%H%M%S')}.csv"
        else:
            fname = f"correlation_matrix_{start.strftime('%Y%m%d_%H%M%S')}.csv"
        
        matrix.to_csv(fname)

if __name__ == '__main__':
    main()
