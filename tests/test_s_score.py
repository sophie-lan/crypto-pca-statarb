"""
Tests for SScoreCalculator.

Validates that s-scores are computed correctly using continuous division
and that edge cases (empty data, zero sigma_eq) are handled safely.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.s_score import SScoreCalculator
from src.ou_estimator import OUResult, TokenOUParams
from src.residual_model import ResidualResult, TokenRegressionResult


def _make_resid_result(residuals: np.ndarray, token: str = "BTC") -> ResidualResult:
    index = pd.date_range("2021-01-01", periods=len(residuals), freq="h", tz="UTC")
    series = pd.Series(residuals, index=index, name=token)
    t = index[-1]
    tr = TokenRegressionResult(
        token=token, beta0=0.0, beta1=0.0, beta2=0.0, residuals=series
    )
    return ResidualResult(timestamp=t, token_results={token: tr})


def _make_ou_result(token: str, m: float, sigma_eq: float) -> OUResult:
    t = pd.Timestamp("2021-01-02", tz="UTC")
    params = TokenOUParams(
        token=token, kappa=5.0, m=m, sigma=0.01, sigma_eq=sigma_eq, a=0.0, b=0.9
    )
    return OUResult(timestamp=t, ou_params={token: params})


class TestSScoreFormula:
    def setup_method(self):
        self.calc = SScoreCalculator()

    def test_s_score_zero_when_x_equals_m(self):
        """When X_t == m, s-score should be 0."""
        # cumsum of all-zero residuals stays at 0
        residuals = np.zeros(50)
        resid = _make_resid_result(residuals)
        ou = _make_ou_result("BTC", m=0.0, sigma_eq=1.0)
        result = self.calc.compute_at_time(resid, ou)
        assert result is not None
        s = result.get_s("BTC")
        assert s is not None
        assert abs(s) < 1e-10

    def test_s_score_positive_when_x_above_m(self):
        """Positive X_t relative to m should yield positive s-score."""
        residuals = np.ones(50) * 0.01  # cumsum drifts positive
        resid = _make_resid_result(residuals)
        ou = _make_ou_result("BTC", m=0.0, sigma_eq=0.01)
        result = self.calc.compute_at_time(resid, ou)
        assert result is not None
        assert result.get_s("BTC") > 0

    def test_s_score_negative_when_x_below_m(self):
        """Negative X_t relative to m should yield negative s-score."""
        residuals = np.ones(50) * -0.01
        resid = _make_resid_result(residuals)
        ou = _make_ou_result("BTC", m=0.0, sigma_eq=0.01)
        result = self.calc.compute_at_time(resid, ou)
        assert result is not None
        assert result.get_s("BTC") < 0

    def test_s_score_is_continuous_not_integer(self):
        """s-score must use / not //: result should not always be integer-valued."""
        residuals = np.linspace(0.001, 0.003, 100)
        resid = _make_resid_result(residuals)
        ou = _make_ou_result("BTC", m=0.0, sigma_eq=0.07)
        result = self.calc.compute_at_time(resid, ou)
        assert result is not None
        s = result.get_s("BTC")
        assert s is not None
        # If // was used, this would be an exact integer; with /, it should not be
        assert s != float(int(s)), "s-score appears to use floor division"

    def test_s_score_magnitude(self):
        """Verify the formula: s = (X_t - m) / sigma_eq."""
        residuals = np.zeros(50)
        residuals[-1] = 1.0  # X_t = 1.0 (cumsum of one spike)
        # Actually cumsum of all zeros except last = 1.0
        resid = _make_resid_result(residuals)
        m = 0.3
        sigma_eq = 0.5
        ou = _make_ou_result("BTC", m=m, sigma_eq=sigma_eq)
        result = self.calc.compute_at_time(resid, ou)
        assert result is not None
        s = result.get_s("BTC")
        X_t = float(np.cumsum(residuals)[-1])
        expected = (X_t - m) / sigma_eq
        assert abs(s - expected) < 1e-10

    def test_returns_none_on_zero_sigma_eq(self):
        """Zero sigma_eq should be skipped gracefully."""
        residuals = np.random.randn(50)
        resid = _make_resid_result(residuals)
        ou = _make_ou_result("BTC", m=0.0, sigma_eq=0.0)
        result = self.calc.compute_at_time(resid, ou)
        # Either None or BTC not in scores
        if result is not None:
            assert result.get_s("BTC") is None

    def test_returns_none_on_empty_residuals(self):
        """Empty residual series should produce no score for that token."""
        t = pd.Timestamp("2021-01-02", tz="UTC")
        idx = pd.DatetimeIndex([], tz="UTC")
        empty_series = pd.Series([], index=idx, dtype=float, name="BTC")
        tr = TokenRegressionResult(
            token="BTC", beta0=0.0, beta1=0.0, beta2=0.0, residuals=empty_series
        )
        resid = ResidualResult(timestamp=t, token_results={"BTC": tr})
        ou = _make_ou_result("BTC", m=0.0, sigma_eq=1.0)
        result = self.calc.compute_at_time(resid, ou)
        assert result is None

    def test_no_overlap_tokens_returns_none(self):
        """If OU result has different tokens than residuals, result should be None."""
        residuals = np.random.randn(50)
        resid = _make_resid_result(residuals, token="BTC")
        ou = _make_ou_result("ETH", m=0.0, sigma_eq=1.0)
        result = self.calc.compute_at_time(resid, ou)
        assert result is None
