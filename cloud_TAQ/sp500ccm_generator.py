import wrds
import pandas as pd

db = wrds.Connection()

# Pull S&P 500 definition and daily returns
sp500 = db.raw_sql("""
    SELECT a.*, b.date, b.ret
    FROM crsp.dsp500list AS a
    JOIN crsp.dsf AS b ON a.permno = b.permno
    WHERE b.date BETWEEN a.start AND a.ending
      AND b.date BETWEEN '01/01/2004' AND '05/31/2025'
    ORDER BY b.date;
""", date_cols=['start', 'ending', 'date'])

# Pull firm metadata
mse = db.raw_sql("""
    SELECT comnam, cusip, ncusip, namedt, nameendt, 
           permno, shrcd, exchcd, hsiccd, ticker, tsymbol
    FROM crsp.msenames;
""", date_cols=['namedt', 'nameendt'])

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

# Ensure linkenddt is filled
ccm['linkenddt'] = ccm['linkenddt'].fillna(pd.to_datetime('today'))

# Merge with link table and filter by valid link date
sp500ccm = (
    sp500
    .merge(ccm, how='left', on='permno')
    .query('date >= linkdt and date <= linkenddt')
    .drop(columns=['namedt', 'nameendt', 'linktype', 'linkprim', 'linkdt', 'linkenddt'])
)

sp500ccm.to_csv("sp500ccm_filtered.csv.gz", compression = 'gzip', index=False)
