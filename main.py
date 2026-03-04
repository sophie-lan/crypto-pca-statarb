# main.py
from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

from src.data_loader import DataLoader, DataLoaderConfig
from src.pca_engine import PCAEngine, PCAConfig
from src.residual_model import ResidualModel, ResidualConfig
from src.ou_estimator import OUEstimator, OUConfig
from src.s_score import SScoreCalculator
from src.strategy import StatArbStrategy, StrategyConfig
from src.backtester import Backtester, BacktestConfig
from src.plotting import (
    plot_factor_cum_returns,
    plot_eigen_weights_at_times,
    plot_strategy_results,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def run_project():
    # === config ===
    data_dir = "data"
    output_dir = "output"
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # testing period
    test_start = pd.to_datetime("2021-09-26T00:00:00+00:00", utc=True)
    test_end = pd.to_datetime("2022-09-25T23:00:00+00:00", utc=True)

    # --- 1. initialization ---
    loader_cfg = DataLoaderConfig(
        prices_path=os.path.join(data_dir, "coin_all_prices_full.csv"),
        universe_path=os.path.join(data_dir, "coin_universe_150K_40.csv"),
    )
    loader = DataLoader(loader_cfg)

    pca_engine = PCAEngine(loader, PCAConfig(window_size=240, min_valid_ratio=0.8))
    resid_model = ResidualModel(loader, ResidualConfig(window_size=240))
    ou_estimator = OUEstimator(OUConfig(hours_per_year=8760, min_points=30))
    sscore_calc = SScoreCalculator()
    strategy = StatArbStrategy(StrategyConfig())
    backtester = Backtester(BacktestConfig(hours_per_year=8760, risk_free_rate=0.0))

    # --- 2. time index ---
    all_times = loader.get_full_price_df().index
    test_times = all_times[(all_times >= test_start) & (all_times <= test_end)]

    eigen1_dict = {}  # timestamp -> {token: v1}
    eigen2_dict = {}  # timestamp -> {token: v2}
    factor_returns_records = []

    logger.info("Total timestamps in testing period: %d", len(test_times))

    # ---- main loop: roll hourly through the testing period ----
    for t in test_times:
        # Step 1: PCA
        pca_res = pca_engine.compute_at_time(t)
        if pca_res is None:
            continue

        # record eigenvectors
        v1 = pca_res.eigenvectors[:, 0]
        v2 = pca_res.eigenvectors[:, 1]
        tokens = pca_res.tokens

        eigen1_dict[t] = {tok: v for tok, v in zip(tokens, v1)}
        eigen2_dict[t] = {tok: v for tok, v in zip(tokens, v2)}

        # record factor return for this hour
        fr = pca_res.factor_returns
        if len(fr) > 0:
            last_idx = fr.index[-1]
            F1_t = fr.iloc[-1]["F1"]
            F2_t = fr.iloc[-1]["F2"]
            factor_returns_records.append(
                {"timestamp": last_idx, "F1": F1_t, "F2": F2_t}
            )

        # Step 2: residual regression
        resid_res = resid_model.compute_at_time(t, pca_res)
        if resid_res is None:
            continue

        # Step 3: OU parameter estimation
        ou_res = ou_estimator.estimate_from_residuals(resid_res)
        if ou_res is None:
            continue

        # Step 4: s-score
        ss_res = sscore_calc.compute_at_time(resid_res, ou_res)
        if ss_res is None:
            continue

        # Step 5: update positions
        universe_tokens = loader.get_universe(t)
        strategy.update_positions(t, ss_res, universe_tokens)

    # ---- save outputs ----

    # 1) eigenvectors -> CSV
    eigen1_df = dict_of_dicts_to_df(eigen1_dict)
    eigen2_df = dict_of_dicts_to_df(eigen2_dict)

    eigen1_path = os.path.join(output_dir, "eigenvector1.csv")
    eigen2_path = os.path.join(output_dir, "eigenvector2.csv")
    eigen1_df.to_csv(eigen1_path, index_label="timestamp")
    eigen2_df.to_csv(eigen2_path, index_label="timestamp")
    logger.info("Saved eigenvector1 to %s", eigen1_path)
    logger.info("Saved eigenvector2 to %s", eigen2_path)

    # 2) trading signals (positions: -1 / 0 / +1) -> CSV
    positions_df = strategy.get_positions_df()
    positions_df = positions_df.loc[
        (positions_df.index >= test_start) & (positions_df.index <= test_end)
    ]

    signals_path = os.path.join(output_dir, "signals.csv")
    positions_df.to_csv(signals_path, index_label="timestamp")
    logger.info("Saved signals (positions) to %s", signals_path)

    # 3) backtest: Sharpe & MDD
    prices_df = loader.get_full_price_df().loc[positions_df.index]
    bt_res = backtester.run(positions_df, prices_df)

    logger.info("Backtest results:")
    logger.info("  Final cumulative return : %.4f", bt_res.cum_returns.iloc[-1])
    logger.info("  Sharpe ratio (annualized): %.4f", bt_res.sharpe)
    logger.info("  Max Drawdown            : %.4f", bt_res.max_drawdown)

    # ---- plots ----

    # (1) factor & BTC & ETH cumulative returns
    if factor_returns_records:
        factor_df = pd.DataFrame(factor_returns_records).set_index("timestamp").sort_index()
        plot_factor_cum_returns(
            factor_df,
            prices_df,
            os.path.join(fig_dir, "factor_vs_btc_eth.png"),
        )

    # (2) eigenportfolio weights at two specified timestamps
    plot_eigen_weights_at_times(
        eigen1_df,
        eigen2_df,
        ["2021-09-26T12:00:00+00:00", "2022-04-15T20:00:00+00:00"],
        fig_dir,
    )

    # (3) strategy cumulative return curve and hourly returns histogram
    plot_strategy_results(
        bt_res,
        os.path.join(fig_dir, "strategy_cum_returns.png"),
        os.path.join(fig_dir, "strategy_returns_hist.png"),
    )

    logger.info("All figures saved to %s", fig_dir)
    logger.info("Done.")


def dict_of_dicts_to_df(d: dict) -> pd.DataFrame:
    """Convert {timestamp: {token: value}} to a DataFrame (index=timestamp, columns=tokens)."""
    if not d:
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(d, orient="index").sort_index()
    df.index.name = "timestamp"
    df = df.fillna(0.0)
    return df


if __name__ == "__main__":
    run_project()
