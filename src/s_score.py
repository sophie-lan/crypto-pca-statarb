from __future__ import annotations

from typing import Dict, Optional
from dataclasses import dataclass

import pandas as pd
import numpy as np

from src.residual_model import ResidualResult
from src.ou_estimator import OUResult,TokenOUParams

@dataclass
class SScoreConfig:
    """
    placeholder for S-Score configuration parameters
    """
    pass

@dataclass
class TokenSScore:
    """
    token: token symbol
    s_score: computed S-Score
    X_t: cummulative process
    """
    token: str
    s: float
    X_t: float

@dataclass
class SScoreResult:
    """
    timestamp
    scores
    """
    timestamp: pd.Timestamp
    scores: Dict[str, TokenSScore]

    def get_s(self, token:str) -> Optional[float]:
        ts = self.scores.get(token)
        return None if ts is None else ts.s

class SScoreCalculator:
    """
    calculate s-score
    """
    def __init__(self, config: Optional[SScoreConfig] = None):
        self.config = config or SScoreConfig()
    
    def compute_at_time(
            self,
            residual_result: ResidualResult,
            ou_result: OUResult
    ) -> Optional[SScoreResult]:
        t = residual_result.timestamp
        scores: Dict[str, TokenSScore] = {}

        common_tokens = set(residual_result.token_results.keys()) & set(
            ou_result.ou_params.keys()
        )

        for token in common_tokens:
            # 1. take residual series and sort by time
            eps_series = residual_result.get_residual_series(token).sort_index()
            if eps_series.shape[0] ==0:
                continue
            # 2. construct X_l
            X = np.cumsum(eps_series.values)
            X_t = float(X[-1])

            # 3. OU params
            params: TokenOUParams = ou_result.get_params(token)
            m = params.m
            sigma_eq = params.sigma_eq

            if sigma_eq <= 0:
                continue

            s = (X_t - m) / sigma_eq

            scores[token] = TokenSScore(
                token=token,
                s=float(s),
                X_t=X_t,                
            )
        if not scores:
            return None
        
        return SScoreResult(timestamp=t, scores=scores)
    
