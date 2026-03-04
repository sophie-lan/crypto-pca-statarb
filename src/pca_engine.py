from __future__ import annotations

from typing import List, Sequence, Optional, Tuple
from dataclasses import dataclass

import pandas as pd
import numpy as np

from .data_loader import DataLoader

@dataclass
class PCAConfig:
    """
    window_size: M, rolling window size for PCA
    min_valid_ratio: minimum ratio of valid data points in the window to perform PCA
    """
    window_size: int = 240
    min_valid_ratio: float = 0.8

@dataclass
class PCAResult:
    """
    timestamp: pd.Timestamp
    tokens: token symbols involved in PCA
    eigenvalues: first two eigenvalues from PCA, descending order
    eigenvectors: first two eigenvectors from PCA, each corresponding to an eigenvalue
    eigenportfolios: portfolio weights for the first two eigenvectors
    factor_returns: time series of factor returns for the two principal components
    """
    timestamp: pd.Timestamp
    tokens: List[str]
    eigenvalues: np.ndarray  # shape (2,)
    eigenvectors: np.ndarray  # shape (2, N)
    eigenportfolios: np.ndarray  # shape (2, N)
    factor_returns: pd.DataFrame  # index = timestamps, columns = ['F1', 'F2']

class PCAEngine:
    """
    for each timestamp t:
    1. get the top40 tokens at time t using DataLoader.get_universe(t)
    2. get normalized price data for these tokens over the past M hours
    3. filter out tokens with insufficient data
    4. perform PCA on the normalized price data, extract first two principal components
    5. compute eigenportfolios and factor returns
    """
    def __init__(self, data_loader: DataLoader, config: Optional[PCAConfig] = None):
        self.data_loader = data_loader
        self.config = config or PCAConfig()
    
    def compute_at_time(self, t: pd.Timestamp | str) -> Optional[PCAResult]:
        """
        Compute PCA at a specific timestamp t, return PCAResult or None if insufficient data.
        """
        if isinstance(t, str):
            t = pd.to_datetime(t, utc=True)
        
        M = self.config.window_size

        # 1. time window: from t-M to t-1
        start = t - pd.Timedelta(hours=M)
        end = t - pd.Timedelta(hours=1)

        universe_tokens = self.data_loader.get_universe(t)
        price_window = self.data_loader.get_price_window(universe_tokens, start, end)
        price_window = price_window.mask(price_window <= 0)
        prices_filled = price_window.ffill().bfill()

        if prices_filled.shape[0] < M * self.config.min_valid_ratio:
            return None  # insufficient data in the time window

        # 3. normalize prices
        returns = np.log(prices_filled).diff()    
        returns = returns.iloc[1:]               

        # 3.1 filter by valid ratio
        valid_ratio = returns.notna().mean(axis=0)
        keep_cols = valid_ratio[valid_ratio >= self.config.min_valid_ratio].index.tolist()
        if len(keep_cols) < 2:
            return None
        returns = returns[keep_cols]

        # 3.2 drop std=0 
        std_returns = returns.std(axis=0)
        keep_cols_std = std_returns[std_returns > 0].index.tolist()
        if len(keep_cols_std) < 2:
            return None
        returns = returns[keep_cols_std]

        # 3.3 z-score standardization：Y = (R - mean) / std
        mean_returns = returns.mean(axis=0)
        std_returns = returns.std(axis=0)
        norm_return = (returns - mean_returns) / std_returns

        norm_return = norm_return.dropna(axis=1, how="any")
        if norm_return.shape[1] < 2:
            return None

        tokens_used = list(norm_return.columns)
        returns = returns[tokens_used]  

        # 4. corr matrix & PCA
        Sigma = norm_return.corr().values
        N = Sigma.shape[0]

        eigvals, eigvecs = np.linalg.eigh(Sigma)  
        sorted_indices = np.argsort(eigvals)[::-1]
        eigvals_sorted = eigvals[sorted_indices]
        eigvecs_sorted = eigvecs[:, sorted_indices]

        if N >= 2:
            lambda1, lambda2 = eigvals_sorted[:2]
            v1 = eigvecs_sorted[:, 0]
            v2 = eigvecs_sorted[:, 1]
        else:
            return None

        # 5. calculate eigenportfolios
        #    Q(j)_i = v(j)_i / sigma_bar_i
        sigma_bar = returns.std(axis=0).values          
        sigma_bar_safe = np.where(sigma_bar == 0, np.nan, sigma_bar)

        Q1 = v1 / sigma_bar_safe
        Q2 = v2 / sigma_bar_safe

        Q1 = np.nan_to_num(Q1, nan=0.0)
        Q2 = np.nan_to_num(Q2, nan=0.0)

        scale1 = np.sum(np.abs(Q1))
        if scale1 > 0:
            Q1 = Q1 / scale1
        scale2 = np.sum(np.abs(Q2))
        if scale2 > 0:
            Q2 = Q2 / scale2

        eigenvectors = np.column_stack([v1, v2])       # N x 2
        eigenportfolios = np.column_stack([Q1, Q2])    # N x 2

        # 6. calculate factor return：F_j = R * Q_j
        R_matrix = returns.values                      # T x N
        F1 = R_matrix @ Q1                             # T
        F2 = R_matrix @ Q2

        factor_returns = pd.DataFrame(
            data={"F1": F1, "F2": F2},
            index=returns.index,
        )

        result = PCAResult(
            timestamp=t,
            tokens=tokens_used,
            eigenvalues=np.array([lambda1, lambda2]),
            eigenvectors=eigenvectors,
            eigenportfolios=eigenportfolios,
            factor_returns=factor_returns,
        )
        return result
