from __future__ import annotations

from typing import Dict, List, Optional
from dataclasses import dataclass

import pandas as pd
import numpy as np

from src.s_score import SScoreResult

@dataclass
class StrategyConfig:
    """
    Stat-arb trading rule config：
    - s_bo: buy-open threshold  s <= -s_bo
    - s_so: sell-open threshold s >=  s_so
    - s_bc: buy-close threshold s >= -s_bc
    - s_sc: sell-close threshold s <=  s_sc
    - trade_size: max position each time for each token 
    """
    s_bo: float = 1.25
    s_so: float = 1.25
    s_bc: float = 0.75
    s_sc: float = 1.0
    trade_size: float = 1.0

class StatArbStrategy:
    """
    simple strategy based on s-score：

    - generate signal following Avellaneda-Lee's threshold rule
    - maintain current_positions[token]
    - every call of update_positions(timestamp, s_scores, universe) 
    will return positions of all tokens at that t
    """
    def __init__(self, config: Optional[StrategyConfig] = None):
        self.config = config or StrategyConfig()
        self.cur_positions: Dict[str, float] = {}
        self.position_history: Dict[pd.Timestamp, Dict[str, float]] = {}
    
    def _update_single_token(self,token:str, s:Optional[float]) -> float:
        """
        update based on current s-score and prev position
        return new position value (-trade_size, 0, +trade_size)
        """
        cfg = self.config
        prev_pos = self.cur_positions.get(token, 0.0)
        
        if s is None or np.isnan(s):
            return 0.0
        
        if prev_pos == 0.0:
            if s <= -cfg.s_bo:
                return +cfg.trade_size
            elif s >= cfg.s_so:
                return -cfg.trade_size
            else:
                return 0.0
        
        if prev_pos > 0:
            if s >= -cfg.s_bc:
                return 0.0
            else:
                return +cfg.trade_size
        
        if prev_pos < 0:
            if s <= cfg.s_sc:
                return 0.0
            else:
                return -cfg.trade_size
        
        return 0.0
    
    def update_positions(
            self,
            timestamp: pd.Timestamp,
            sscore_result: SScoreResult,
            universe_tokens: List[str]
    ) -> pd.Series:
        new_positions= dict(self.cur_positions)

        for token in universe_tokens:
            s = sscore_result.get_s(token)
            new_pos = self._update_single_token(token, s)
            new_positions[token] = new_pos
        
        self.cur_positions = new_positions

        self.position_history[timestamp] = dict(new_positions)

        pos_series = pd.Series(
            {token: new_positions.get(token, 0.0) for token in universe_tokens},
            name=timestamp
        )

        return pos_series
    
    def get_positions_df(self) -> pd.DataFrame:
        if not self.position_history:
            return pd.DataFrame()
        
        df = pd.DataFrame.from_dict(self.position_history, orient="index").sort_index()
        df.index.name = "timestamp"
        return df
    