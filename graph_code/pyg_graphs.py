import os, sys
import glob
from pathlib import Path
import torch
import pandas as pd
from torch_geometric.data import Data, Dataset
import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from torch_geometric.data.makedirs import makedirs
from datetime import datetime, timedelta
import numpy as np
import wrds

# cols = ['sale_nwc', 'rd_sale', 'adv_sale', 'staff_sale', 'accrual', 'ptb', 'PEG_trailing']
cols = ['bm', 'evm', 'pe_op_basic', 'pe_op_dil', 'pe_exi', 'pe_inc', 'ps',
       'pcf', 'dpr', 'npm', 'opmbd', 'opmad', 'gpm', 'ptpm', 'cfm', 'roa',
       'roe', 'roce', 'efftax', 'aftret_eq', 'aftret_invcapx', 'aftret_equity',
       'pretret_noa', 'pretret_earnat', 'equity_invcap', 'debt_invcap',
       'totdebt_invcap', 'capital_ratio', 'int_debt', 'int_totdebt', 'cash_lt',
       'invt_act', 'rect_act', 'debt_at', 'debt_ebitda', 'short_debt',
       'curr_debt', 'lt_debt', 'profit_lct', 'ocf_lct', 'cash_debt', 'fcf_ocf',
       'lt_ppent', 'dltt_be', 'debt_assets', 'debt_capital', 'de_ratio',
       'intcov', 'intcov_ratio', 'cash_ratio', 'quick_ratio', 'curr_ratio',
       'cash_conversion', 'inv_turn', 'at_turn', 'rect_turn', 'pay_turn',
       'sale_invcap', 'sale_equity', 'sale_nwc', 'rd_sale', 'adv_sale',
       'staff_sale', 'accrual', 'ptb', 
    #    'Energy','Materials', 'Industrials', 'Consumer Discretionary', 'Consumer Staples', 
    #    'Health Care', 'Financials', 'Information Technology', 'Teleommunication Services', 'Utilities', 'Financials'
       ]

def format_date(date):
    return f'{date[:4]}-{date[4:6]}-{date[6:]}'

class StockMarketDataset(Dataset):
    def __init__(self, root, exp, intra=True, transform=None, pre_transform=None):
        self.exp = exp
        self.intra = intra
        super().__init__(root, transform, pre_transform)
    # @property
    # def processed_dir(self):
    #     return os.path.join(self.root, 'processed_gics')

    def _process(self):
        if not os.path.isdir(self.processed_dir):
            makedirs(self.processed_dir)
        if self.intra:
            if len(os.listdir(self.processed_dir))> 20000:
                return
            self.process_intra()
        else:
            return
            # self.process
    @property
    def raw_file_names(self):
        """
        The raw file names are the directories (dates) under the root directory.
        """
        names = [d for d in os.listdir(self.raw_dir) if os.path.isdir(os.path.join(self.raw_dir, d))]
        names.sort()
        return names

    @property
    def processed_file_names(self):
        """
        The processed file names are Data objects saved as .pt files for each day.
        """
        return [f"data_{i}.pt" for i in range(len(self.raw_file_names))] # daily data
        # names = [d for d in os.listdir(self.processed_dir) if os.path.isdir(os.path.join(self.processed_dir, d))]
        ##########
        # intraday 
        # names = os.listdir(self.processed_dir)
        # names.sort(key = lambda x: int(x[:-3].split('_')[-1]))
        # return names
        ##########

    def process(self):
        """
        Processes the raw data into PyG Data objects.
        """
        price = pd.read_csv('/N/u/ebracht/Quartz/Documents/thesis/data/processed/spy_daily.csv', index_col = 0)
        price.date = pd.to_datetime(price.date)
        for i, day_folder in tqdm.tqdm(enumerate(self.raw_file_names)):
            folder_path = os.path.join(self.raw_dir, day_folder)
            # print(day_folder)
            date = format_date(day_folder)
            # Load adjacency matrix from parquet
            adj_path = os.path.join(folder_path, f"{self.exp}_pmfg.parquet")
            adj_df = pd.read_parquet(adj_path)
            tickers = adj_df.index.tolist()
            edges = [(tickers.index(src), tickers.index(dst)) for src, dst in adj_df[adj_df > 0].stack().index]
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_weight = torch.tensor(adj_df[adj_df > 0].stack().values, dtype=torch.float)
            
            # new stuff
            # sp500ccm link from wrds cloud code

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
                .drop(columns=['namedt', 'nameendt', 'linktype', 'linkprim', 'linkdt', 'linkenddt', 'ret', 'comnam', 'cusip', 'ncusip', 'shrcd', 'exchcd', 'hsiccd'])
            )

            cols = ['gvkey', 'public_date', 'bm', 'evm', 'pe_op_basic', 'pe_op_dil', 'pe_exi', 'pe_inc', 'ps',
            'pcf', 'dpr', 'npm', 'opmbd', 'opmad', 'gpm', 'ptpm', 'cfm', 'roa',
            'roe', 'roce', 'efftax', 'aftret_eq', 'aftret_invcapx', 'aftret_equity',
            'pretret_noa', 'pretret_earnat', 'equity_invcap', 'debt_invcap',
            'totdebt_invcap', 'capital_ratio', 'int_debt', 'int_totdebt', 'cash_lt',
            'invt_act', 'rect_act', 'debt_at', 'debt_ebitda', 'short_debt',
            'curr_debt', 'lt_debt', 'profit_lct', 'ocf_lct', 'cash_debt', 'fcf_ocf',
            'lt_ppent', 'dltt_be', 'debt_assets', 'debt_capital', 'de_ratio',
            'intcov', 'intcov_ratio', 'cash_ratio', 'quick_ratio', 'curr_ratio',
            'cash_conversion', 'inv_turn', 'at_turn', 'rect_turn', 'pay_turn',
            'sale_invcap', 'sale_equity', 'sale_nwc', 'rd_sale', 'adv_sale',
            'staff_sale', 'accrual', 'ptb', 
            ]
            cols_sql = ", ".join(cols)

            year_date = f"{int(date[:4]) - 1}{date[4:]}"
            day_date = (datetime.datetime.strptime(date, "%Y-%m-%d") - datetime.timedelta(days=1))
            
            features = []
            targets = []

            for ticker in tickers:
                #x
                filtered = sp500ccm[(sp500ccm.date == date) & (sp500ccm.ticker == ticker)]
                if filtered.empty:
                    print(f"No gvkey/permno found for ticker {ticker} on date {date}")
                    continue  # or handle differently
                gvkey = str(filtered.gvkey.iloc[0])
                compustat = db.raw_sql(f"""
                    SELECT {cols_sql}
                    FROM wrdsapps_finratio.firm_ratio
                    WHERE gvkey = '{gvkey}'
                    AND public_date BETWEEN '{year_date}' AND '{date}'
                    ORDER BY public_date;
                """)
                values = compustat.sort_values(by='public_date', ascending=False).head(1).drop(['gvkey', 'public_date'], axis=1).iloc[0].to_numpy()
                features.append([float(v) if pd.notnull(v) else float(0.0) for v in values])

                #y
                permno = str(filtered.permno.iloc[0])
                prices = db.raw_sql(f"""
                    SELECT date, permno, prc
                    FROM crsp_a_stock.dsf                     
                    WHERE permno = '{permno}'
                    AND date BETWEEN '{day_date}' AND '{date}'                                    
                """)
                prices = prices.sort_values(by = 'date')
                prev_price = prices.iloc[0]['prc']
                curr_price = prices.iloc[1]['prc']

                target = int(curr_price > prev_price)
                targets.append(target)

            x = torch.tensor(features, dtype=torch.float)
            y = torch.tensor(targets, dtype=torch.float)
            
            # Create Data object
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y, id = day_folder)

            if self.pre_transform:
                data = self.pre_transform(data)

            torch.save(data, os.path.join(self.processed_dir, f"data_{i}.pt"))

    def len(self):
        """
        Number of processed graphs in the dataset.
        """
        return len(self.processed_file_names)

    def get(self, idx):
        """
        Get a processed graph by index.
        """
        return torch.load(os.path.join(self.processed_dir, f"data_{idx}.pt"))
