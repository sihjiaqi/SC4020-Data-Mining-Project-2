from __future__ import annotations
from typing import Iterable, List, Sequence, Tuple, Set
import numpy as np
import pandas as pd

class SymptomPredictor:
    """
    "Next symptom" = link-prediction-by-projection on the filtered bipartite slice.
    Score(s) = conditional prevalence within filtered rows.
    """
    def rank(self, df_filtered: pd.DataFrame, all_symptom_cols: Sequence[str], already_selected: Set[str], *,
             top_k: int = 10) -> List[Tuple[str, float]]:
        if df_filtered is None or df_filtered.empty:
            return []
        cols = [c for c in all_symptom_cols if c in df_filtered.columns and c not in already_selected]
        if not cols:
            return []
        prev = df_filtered[cols].astype("int8").mean(axis=0)  # float prevalence
        ranked = prev.sort_values(ascending=False)
        return [(c, float(ranked[c])) for c in ranked.index[:top_k]]