from __future__ import annotations

from typing import List, Optional, Dict
from dataclasses import dataclass

import pandas as pd
import numpy as np

from src.data_loader import DataLoader
from src.pca_engine import PCAResult,PCAConfig

@dataclass
class ResidualConfig:
    """
    window_size: int = 240
    """
    window_size: int = 240
    min_valid_ratio: float = 0.8
    min_samples: int = 30
    ridge_lambda: float = 1e-5

@dataclass
class TokenRegressionResult:
    """
    store single token regression result
    token: token symbol
    beta0, beta1, beta2: coefficients
    residuals: time series of residuals
    """
    token: str
    beta0: float
    beta1: float
    beta2: float
    residuals: pd.Series  

@dataclass
class ResidualResult:
    """
    timestamp: current timestamp
    token_results:  key: token symbol, value: TokenRegressionResult
    """
    timestamp: pd.Timestamp
    token_results: Dict[str, TokenRegressionResult]

    def get_residual_series(self, token:str) -> pd.Series:
        
        return self.token_results[token].residuals
    
    def get_betas(self, token:str) -> np.ndarray:
        tr = self.token_results[token]
        return np.array([tr.beta0, tr.beta1, tr.beta2],dtype=float)

class ResidualModel:
    """
    linear regression of single token returns against first two PCA factor returns
    """

    def __init__(self, data_loader: DataLoader, config: Optional[ResidualConfig] = None):
        self.data_loader = data_loader
        self.config = config or ResidualConfig()
    
    def compute_at_time(
            self, 
            t: pd.Timestamp | str, 
            pca_result: PCAResult) -> Optional[ResidualResult]:
        if pca_result is None:
            return None
        if isinstance(t, str):
            t = pd.to_datetime(t, utc=True)
        
        M = self.config.window_size
        cfg = self.config
        start = t - pd.Timedelta(hours=M)
        end = t - pd.Timedelta(hours=1)

        tokens = pca_result.tokens
        factor_df = pca_result.factor_returns

        prices_window = self.data_loader.get_price_window(tokens, start, end)
        prices_window = prices_window.mask(prices_window <= 0)
        prices_window = prices_window.ffill().bfill()
        returns = np.log(prices_window).diff().iloc[1:]  # drop first NaN row

        common_index = returns.index.intersection(factor_df.index)
        returns = returns.loc[common_index]
        factor_df = factor_df.loc[common_index]

        if len(common_index) < cfg.min_samples:
            return None  # no overlapping data
        
        valid_ratio = returns.notna().mean(axis=0)
        keep_cols = valid_ratio[valid_ratio >= cfg.min_valid_ratio].index.tolist()
        if len(keep_cols) < 1:
            return None
        returns = returns[keep_cols]
        
        # construct design matrix
        X = np.column_stack(
            [
                np.ones(len(common_index)),           # intercept
                factor_df["F1"].values,
                factor_df["F2"].values,
            ]
        ) # shape (N, 3)

        token_results: Dict[str, TokenRegressionResult] = {}

        # perform regression for each token
        for token in keep_cols:
            y = returns[token].values  # shape (N,)

            mask = ~np.isnan(y)
            if np.sum(mask) < 5:
                continue  # not enough data points
            X_valid = X[mask]
            y_valid = y[mask]
            idx_clean = common_index[mask]

            ridge_lambda = cfg.ridge_lambda
            XT_X = X_valid.T @ X_valid
            reg_matrix = ridge_lambda * np.eye(XT_X.shape[0])
            beta = np.linalg.solve(XT_X + reg_matrix, X_valid.T @ y_valid)


            # compute residuals
            y_hat = X_valid @ beta
            eps = y_valid - y_hat

            eps = eps - eps.mean()

            residual_series = pd.Series(data=eps, index=idx_clean, name=token)

            token_results[token] = TokenRegressionResult(
                token=token,
                beta0=float(beta[0]),
                beta1=float(beta[1]),
                beta2=float(beta[2]),
                residuals=residual_series
            )
        if not token_results:
            return None  # no valid tokens
        return ResidualResult(timestamp=t, token_results=token_results)