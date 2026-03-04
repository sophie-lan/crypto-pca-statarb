from __future__ import annotations

import pandas as pd
import numpy as np

from typing import List, Sequence, Optional
from dataclasses import dataclass

"""
Read price data from CSV files and provide data loading functionality.
get_universe(t): return top40 tokens at time t.
get_price(token, start,end): return price data for a token between start and end dates.
"""

@dataclass
class DataLoaderConfig:
    prices_path: str  # Path to the CSV file containing price data
    universe_path: str  # Path to the CSV file containing universe data
    tz: str = "UTC"  # Timezone for datetime parsing

class DataLoader:
    def __init__(self, config: DataLoaderConfig):
        self.config = config

        self.prices_raw = self._load_prices_csv(config.prices_path)
        self.universe_raw = self._load_universe_csv(config.universe_path)

        self.common_index = self.prices_raw.index.intersection(self.universe_raw.index).sort_values()

        self.prices = self.prices_raw.loc[self.common_index]
        self.universe = self.universe_raw.loc[self.common_index]

    def _load_prices_csv(self, path: str) -> pd.DataFrame:
        """
        Read: coins_all_prices.csv
        Return:index = Timestamp, columns = token symbol, values = price (float)
        """
        df = pd.read_csv(path)
        df['timestamp'] = pd.to_datetime(df["startTime"], utc=True)
        df = df.set_index('timestamp').sort_index()
        prices_col = [col for col in df.columns if col not in ["startTime", "time"]]
        df = df[prices_col].astype(float)
        return df
    
    def _load_universe_csv(self, path: str) -> pd.DataFrame:
        """
        Read: coins_universe_150K_40.csv
        Return:index = Timestamp, columns = ['0', '1', ..., '39'], values = token symbol (str)
        """
        df = pd.read_csv(path)
        df['timestamp'] = pd.to_datetime(df["startTime"], utc=True)
        df = df.set_index('timestamp').sort_index()
        universe_col = [str(i) for i in range(40) if str(i) in df.columns]
        df = df[universe_col].astype(str)
        return df

    def get_universe(self, t: pd.Timestamp) -> List[str]:
        """
        Return the top40 tokens at time t.
        """
        if isinstance(t, str):
            t = pd.to_datetime(t, utc=True)
        row = self.universe.loc[t]
        raw_symbols = row.values.tolist()

        tokens: List[str] = []
        for sym in raw_symbols:
            if sym is None:
                continue
            sym_str = str(sym).strip()

            if sym_str == "" or sym_str.lower() == "nan":
                continue
            tokens.append(sym_str)
        tokens =[sym for sym in tokens if sym in self.prices.columns]

        return tokens

    def get_price_window(
            self,
            tokens: Sequence[str],
            start: pd.Timestamp | str,
            end: pd.Timestamp | str
            ) -> pd.DataFrame:
        """
        return price data for the given tokens between start and end dates
        index = Timestamp, columns = tokens
        """
        if isinstance(start, str):
            start = pd.to_datetime(start, utc=True)
        if isinstance(end, str):
            end = pd.to_datetime(end, utc=True)
         
        price_window = self.prices.loc[start:end, tokens]
        price_window = price_window.ffill().bfill()
        return price_window

    def get_return_window(
            self,
            tokens: Sequence[str],
            start: pd.Timestamp | str,
            end: pd.Timestamp | str
            ) -> pd.DataFrame:
        """
        return simple returns on the price window:
        R_t = (P_t - P_{t-1}) / P_{t-1}
        """
        price_window = self.get_price_window(tokens, start, end)
        return_window = price_window.pct_change()
        return return_window

    def get_full_price_df(self) -> pd.DataFrame:
        return self.prices.copy()

    def get_full_universe_df(self) -> pd.DataFrame:
        return self.universe.copy()