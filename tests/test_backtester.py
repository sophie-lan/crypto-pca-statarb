"""
Tests for Backtester.

Validates Sharpe ratio and MDD calculations against known analytical results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtester import Backtester, BacktestConfig, BacktestResult


def _make_positions(data: dict, index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(data, index=index)


def _make_prices(data: dict, index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(data, index=index)


class TestBacktesterSharpe:
    def setup_method(self):
        self.bt = Backtester(BacktestConfig(hours_per_year=8760, risk_free_rate=0.0))

    def test_zero_return_gives_zero_sharpe(self):
        """Flat prices → zero returns → Sharpe = 0."""
        idx = pd.date_range("2021-01-01", periods=100, freq="h", tz="UTC")
        prices = _make_prices({"BTC": np.ones(100)}, idx)
        positions = _make_positions({"BTC": np.ones(100)}, idx)
        result = self.bt.run(positions, prices)
        assert result.sharpe == 0.0

    def test_constant_positive_return_sharpe(self):
        """Constant positive hourly return should yield a positive Sharpe ratio."""
        idx = pd.date_range("2021-01-01", periods=100, freq="h", tz="UTC")
        prices_arr = np.cumprod(np.ones(100) * 1.001)
        prices = _make_prices({"BTC": prices_arr}, idx)
        positions = _make_positions({"BTC": np.ones(100)}, idx)
        result = self.bt.run(positions, prices)
        # Position is shifted by 1, so first return is 0 and rest are ~0.001.
        # The series is not constant from the portfolio's perspective → std > 0 → Sharpe > 0.
        assert result.sharpe > 0

    def test_sharpe_positive_for_positive_mean_return(self):
        """Positive-mean returns should yield positive Sharpe."""
        rng = np.random.default_rng(42)
        n = 1000
        idx = pd.date_range("2021-01-01", periods=n, freq="h", tz="UTC")
        returns = 0.001 + rng.standard_normal(n) * 0.005  # positive mean
        prices_arr = np.cumprod(1 + returns)
        prices = _make_prices({"BTC": prices_arr}, idx)
        positions = _make_positions({"BTC": np.ones(n)}, idx)
        result = self.bt.run(positions, prices)
        assert result.sharpe > 0

    def test_sharpe_negative_for_negative_mean_return(self):
        rng = np.random.default_rng(99)
        n = 1000
        idx = pd.date_range("2021-01-01", periods=n, freq="h", tz="UTC")
        returns = -0.001 + rng.standard_normal(n) * 0.005
        prices_arr = np.cumprod(1 + np.clip(returns, -0.99, None))
        prices = _make_prices({"BTC": prices_arr}, idx)
        positions = _make_positions({"BTC": np.ones(n)}, idx)
        result = self.bt.run(positions, prices)
        assert result.sharpe < 0


class TestBacktesterMDD:
    def setup_method(self):
        self.bt = Backtester(BacktestConfig(hours_per_year=8760, risk_free_rate=0.0))

    def test_no_drawdown_on_monotone_growth(self):
        """Strictly increasing cumulative return → MDD = 0."""
        idx = pd.date_range("2021-01-01", periods=50, freq="h", tz="UTC")
        prices_arr = np.cumprod(np.ones(50) * 1.01)
        prices = _make_prices({"BTC": prices_arr}, idx)
        positions = _make_positions({"BTC": np.ones(50)}, idx)
        result = self.bt.run(positions, prices)
        assert result.max_drawdown >= -1e-10  # allow tiny float error

    def test_mdd_is_negative(self):
        """Any drawdown should produce a negative MDD."""
        idx = pd.date_range("2021-01-01", periods=6, freq="h", tz="UTC")
        # prices: up then down
        prices = _make_prices({"BTC": [100, 110, 120, 100, 90, 95]}, idx)
        positions = _make_positions({"BTC": np.ones(6)}, idx)
        result = self.bt.run(positions, prices)
        assert result.max_drawdown < 0

    def test_mdd_known_value(self):
        """
        Manual example:
          prices:    100 → 200 → 100
          returns:   +100%, -50%
          cum_ret:   1 → 2 → 1
          drawdown:  0 → 0 → -0.5
          MDD = -0.5
        """
        idx = pd.date_range("2021-01-01", periods=3, freq="h", tz="UTC")
        prices = _make_prices({"BTC": [100.0, 200.0, 100.0]}, idx)
        positions = _make_positions({"BTC": [1.0, 1.0, 1.0]}, idx)
        result = self.bt.run(positions, prices)
        assert abs(result.max_drawdown - (-0.5)) < 1e-6

    def test_zero_position_gives_flat_curve(self):
        """All-zero positions → portfolio return = 0 every period."""
        idx = pd.date_range("2021-01-01", periods=50, freq="h", tz="UTC")
        prices_arr = np.cumprod(1 + np.random.randn(50) * 0.01)
        prices = _make_prices({"BTC": prices_arr}, idx)
        positions = _make_positions({"BTC": np.zeros(50)}, idx)
        result = self.bt.run(positions, prices)
        assert (result.returns == 0.0).all()
        assert result.max_drawdown == 0.0


class TestBacktesterReturnCalculation:
    def setup_method(self):
        self.bt = Backtester(BacktestConfig(hours_per_year=8760, risk_free_rate=0.0))

    def test_position_shift_by_one(self):
        """Positions are shifted by 1 before multiplying returns (no look-ahead)."""
        idx = pd.date_range("2021-01-01", periods=4, freq="h", tz="UTC")
        # prices rise 10% at t=1, then flat
        prices = _make_prices({"BTC": [100.0, 110.0, 110.0, 110.0]}, idx)
        # position = 1 at t=0, signal fires at t=1 price
        positions = _make_positions({"BTC": [1.0, 1.0, 1.0, 1.0]}, idx)
        result = self.bt.run(positions, prices)
        # First return should be 0 (shifted position at t=0 is 0)
        assert result.returns.iloc[0] == 0.0
