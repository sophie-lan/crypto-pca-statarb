from __future__ import annotations

from typing import Optional
from dataclasses import dataclass

import pandas as pd
import numpy as np

@dataclass
class BacktestConfig:
    """
    hour_per_year
    risk_free_rate: 0 by default
    """
    hours_per_year: int = 8760
    risk_free_rate: float = 0.0

@dataclass
class BacktestResult:
    """
    Result of backtest:
    returns: hourly return rate of portfolio
    cum_returns: cumulative return curve
    sharpe ratio: annualized
    max_drawdown
    """
    returns: pd.Series
    cum_returns: pd.Series
    sharpe: float
    max_drawdown: float

class Backtester:
    """
    given current position and prices, calculate:
    hourly portfolio return
    sharpe ratio
    MDD
    """
    def __init__(self, config: Optional[BacktestConfig]=None):
        self.config = config or BacktestConfig()
    
    def run(
            self,
            positions: pd.DataFrame,
            prices: pd.DataFrame
    ) -> BacktestResult:
        prices = prices.sort_index()
        positions = positions.sort_index()

        common_index = prices.index.intersection(positions.index)
        prices = prices.loc[common_index]
        positions = positions.loc[common_index]

        returns_mat = prices.pct_change()
        returns_mat = returns_mat.fillna(0.0)

        shifted_pos = positions.shift(1).fillna(0.0)

        shifted_pos, returns_mat = shifted_pos.align(returns_mat, axis=1,join="outer",fill_value=0.0)

        pnl = (shifted_pos * returns_mat).sum(axis=1)
        notional = shifted_pos.abs().sum(axis=1)

        portfolio_ret = pnl.copy()
        nonzero_mask = notional > 0
        portfolio_ret[nonzero_mask] = pnl[nonzero_mask]/notional[nonzero_mask]
        portfolio_ret[~nonzero_mask] = 0.0

        cum_ret = (1.0+portfolio_ret).cumprod()

        sharpe = self._compute_sharpe(portfolio_ret)

        mdd = self._compute_mdd(cum_ret)

        return BacktestResult(
            returns=portfolio_ret,
            cum_returns=cum_ret,
            sharpe=sharpe,
            max_drawdown=mdd
        )
    
    def _compute_sharpe(self, r:pd.Series) -> float:
        """
        Sharpe = (E[R] - r_f) / std(R) * sqrt(hours_per_year)
        """
        rf = self.config.risk_free_rate
        excess = r - rf
        
        mu = excess.mean()
        sigma = excess.std(ddof=1)

        if sigma == 0 or np.isnan(sigma):
            return 0.0
        
        ann_factor = np.sqrt(self.config.hours_per_year)
        sharpe = float(mu / sigma * ann_factor)

        return sharpe

    def _compute_mdd(self, cum_ret: pd.Series) -> float:
        """
        MDD = min_t (cum_ret_t / running_max_t - 1)
        """
        running_max = cum_ret.cummax()
        drawdown = cum_ret/running_max - 1.0
        mdd = float(drawdown.min())
        return mdd

