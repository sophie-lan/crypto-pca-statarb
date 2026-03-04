from __future__ import annotations

from typing import Dict, Optional
from dataclasses import dataclass

import pandas as pd
import numpy as np

from src.residual_model import ResidualResult

@dataclass
class OUConfig:
    """
    hours_per_year
    min_points: minimum data points required to fit OU model
    """
    hours_per_year: int = 8760
    min_points: int = 30

@dataclass
class TokenOUParams:
    """
    token: token symbol
    kappa: mean reversion rate
    m: long-term mean
    sigma: volatility
    sigma_eq: equilibrium volatility
    a, b: discretized OU parameters
    """
    token: str
    kappa: float
    m: float
    sigma: float
    sigma_eq: float
    a: float
    b: float

@dataclass
class OUResult:
    """
    timestamp: current timestamp
    ou_params: key: token symbol, value: TokenOUParams
    """
    timestamp: pd.Timestamp
    ou_params: Dict[str, TokenOUParams]

    def get_params(self, token:str) -> TokenOUParams:
        return self.ou_params[token]

class OUEstimator:
    """
    construct X_l from residual time series
    get OU parameters by formula in Avellaneda & Lee Appendix
    """
    def __init__(self, config: Optional[OUConfig] = None):
        self.config = config or OUConfig()
    
    def estimate_from_residuals(
            self,
            resid_result: ResidualResult
        ) -> Optional[OUResult]:
        """
        Estimate OU parameters for each token in resid_result at time t.
        Return OUResult.
        """
        t = resid_result.timestamp
        delta_t = 1.0  / self.config.hours_per_year

        ou_params: Dict[str, TokenOUParams] = {}

        for token, reg_res in resid_result.token_results.items():
            eps_series = reg_res.residuals.sort_index()

            if eps_series.shape[0] < self.config.min_points:
                continue  # insufficient data points
            # 1. construct X_l
            eps_values = eps_series.values
            X = np.cumsum(eps_values)  

            X_l = X[:-1]
            X_next = X[1:]

            if X_l.shape[0] < self.config.min_points:
                continue  # insufficient data points after lagging
            
            # 2. fit linear regression X_{l+1} = a + b * X_l + error
            T = X_l.shape[0]
            X_design = np.column_stack([np.ones(T), X_l])  # shape (T, 2)

            beta, *_ = np.linalg.lstsq(X_design, X_next, rcond=None)
            a_hat, b_hat = float(beta[0]), float(beta[1])

            # 3. compute variance of residuals
            X_next_hat = X_design @ beta
            eta = X_next - X_next_hat
            var_eta = np.var(eta, ddof=1)

            # 4. compute OU parameters
            if not (0 < b_hat < 1):
                continue  # invalid b_hat
            kappa = -np.log(b_hat) / delta_t
            if kappa <= 0:
                continue  # invalid kappa
            
            m = a_hat / (1 - b_hat)
            
            denom = (1 - b_hat**2)
            if denom <= 0:
                continue  # invalid denom

            sigma_eq_sq = var_eta / denom
            if sigma_eq_sq < 0:
                continue  # invalid sigma_eq_sq

            sigma_eq = float(np.sqrt(sigma_eq_sq))
            sigma = float(np.sqrt(2.0*kappa) * sigma_eq)


            ou_params[token] = TokenOUParams(
                token=token,
                kappa=float(kappa),
                m=float(m),
                sigma=sigma,
                sigma_eq=sigma_eq,
                a=a_hat,
                b=b_hat
            )
        
        if not ou_params:
            return None  # no valid tokens
        return OUResult(timestamp=t, ou_params=ou_params)