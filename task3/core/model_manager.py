from __future__ import annotations
from typing import Iterable, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

class ModelManager:
    def __init__(_self, trainer, symptom_pred, min_display_prob: float):
        _self.trainer = trainer
        _self.symptom_pred = symptom_pred
        _self.min_display_prob = float(min_display_prob)

    # returns (labels, probs, used_features)
    def score_selected(_self, df_wide: pd.DataFrame, target_col: str, selected_present: Iterable[str]):
        clf, used_feats, labels = _self.trainer.train_tree(
            df_wide=df_wide,
            target_col=target_col,
            selected_present=selected_present,
            max_depth=4,
            min_leaf=2,
            seed=42,
        )
        if clf is None:
            return [], None, []
        probs, _ = _self.trainer.predict_tree(clf, used_feats)
        return labels, probs, used_feats

    def suggest_next(_self, df_filtered: pd.DataFrame, all_symptoms: Sequence[str], already_selected: Iterable[str], top_k: int = 10):
        return _self.symptom_pred.rank(
            df_filtered=df_filtered,
            all_symptom_cols=list(all_symptoms),
            already_selected=set(already_selected),
            top_k=int(top_k),
        )