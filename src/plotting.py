from __future__ import annotations

import logging
import os

import matplotlib.pyplot as plt
import pandas as pd

from src.backtester import BacktestResult

logger = logging.getLogger(__name__)


def plot_factor_cum_returns(
    factor_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    save_path: str,
) -> None:
    """
    Plot cumulative returns for the two eigenportfolios, BTC, and ETH.

    Args:
        factor_df: DataFrame with columns ['F1', 'F2'], indexed by timestamp.
        prices_df: Price DataFrame with at least 'BTC' and 'ETH' columns.
        save_path: File path to save the figure.
    """
    F1_cum = (1.0 + factor_df["F1"]).cumprod()
    F2_cum = (1.0 + factor_df["F2"]).cumprod()

    common_index = prices_df.index.intersection(factor_df.index)
    px_sub = prices_df.loc[common_index, ["BTC", "ETH"]].dropna(how="any")
    ret_btc = px_sub["BTC"].pct_change().fillna(0.0)
    ret_eth = px_sub["ETH"].pct_change().fillna(0.0)
    BTC_cum = (1.0 + ret_btc).cumprod()
    ETH_cum = (1.0 + ret_eth).cumprod()

    plt.figure(figsize=(10, 6))
    F1_cum.reindex(common_index).plot(label="Eigenportfolio 1")
    F2_cum.reindex(common_index).plot(label="Eigenportfolio 2")
    BTC_cum.plot(label="BTC")
    ETH_cum.plot(label="ETH")
    plt.legend()
    plt.title("Cumulative Returns: Eigenportfolios vs BTC & ETH")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Return")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info("Saved factor cumulative returns plot to %s", save_path)


def plot_eigen_weights_at_times(
    eigen1_df: pd.DataFrame,
    eigen2_df: pd.DataFrame,
    time_strs: list[str],
    fig_dir: str,
) -> None:
    """
    Plot eigenportfolio weights at specified timestamps, sorted by absolute weight.

    Args:
        eigen1_df: DataFrame of PC1 eigenvector weights (index=timestamp, columns=tokens).
        eigen2_df: DataFrame of PC2 eigenvector weights.
        time_strs: List of ISO-format timestamp strings to plot.
        fig_dir: Directory to save figures.
    """
    for timestr in time_strs:
        t = pd.to_datetime(timestr, utc=True)
        if t not in eigen1_df.index:
            logger.warning("Timestamp %s not in eigenvector data, skipping.", t)
            continue

        w1 = eigen1_df.loc[t].copy()
        w2 = eigen2_df.loc[t].copy()

        w1_sorted = w1.reindex(w1.abs().sort_values(ascending=False).index).iloc[:40]
        w2_sorted = w2.reindex(w2.abs().sort_values(ascending=False).index).iloc[:40]

        plt.figure(figsize=(12, 5))
        plt.bar(range(len(w1_sorted)), w1_sorted.values)
        plt.xticks(range(len(w1_sorted)), w1_sorted.index, rotation=90)
        plt.title(f"Eigenportfolio 1 Weights at {timestr}")
        plt.tight_layout()
        path1 = os.path.join(fig_dir, f"eigenweights1_{t.strftime('%Y%m%d_%H%M')}.png")
        plt.savefig(path1)
        plt.close()

        plt.figure(figsize=(12, 5))
        plt.bar(range(len(w2_sorted)), w2_sorted.values)
        plt.xticks(range(len(w2_sorted)), w2_sorted.index, rotation=90)
        plt.title(f"Eigenportfolio 2 Weights at {timestr}")
        plt.tight_layout()
        path2 = os.path.join(fig_dir, f"eigenweights2_{t.strftime('%Y%m%d_%H%M')}.png")
        plt.savefig(path2)
        plt.close()

        logger.info("Saved eigenweight plots for %s", timestr)


def plot_strategy_results(
    bt_res: BacktestResult,
    curve_path: str,
    hist_path: str,
) -> None:
    """
    Plot strategy cumulative return curve and hourly returns histogram.

    Args:
        bt_res: BacktestResult from Backtester.run().
        curve_path: File path for the cumulative return curve.
        hist_path: File path for the returns histogram.
    """
    plt.figure(figsize=(10, 5))
    bt_res.cum_returns.plot()
    plt.title("Strategy Cumulative Return")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Return")
    plt.tight_layout()
    plt.savefig(curve_path)
    plt.close()
    logger.info("Saved cumulative return curve to %s", curve_path)

    plt.figure(figsize=(8, 5))
    bt_res.returns.hist(bins=50)
    plt.title("Strategy Hourly Returns Histogram")
    plt.xlabel("Hourly Return")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(hist_path)
    plt.close()
    logger.info("Saved returns histogram to %s", hist_path)
