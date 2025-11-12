from __future__ import annotations
from typing import Iterable, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

class ModelTrainer:
    def train_tree(_self, df_wide: pd.DataFrame, target_col: str, selected_present: Iterable[str], *,
                   max_depth: int = 4, min_leaf: int = 2, seed: int = 42) -> Tuple[Optional[DecisionTreeClassifier], List[str], List[str]]:
        features = [c for c in selected_present if c in df_wide.columns]
        if not features:
            return None, [], []
        X = df_wide[features].astype("int8")
        y = df_wide[target_col].astype(str)
        clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_leaf, class_weight="balanced", random_state=seed)
        clf.fit(X, y)
        return clf, features, clf.classes_.tolist()

    def predict_tree(_self, clf: DecisionTreeClassifier, used_features: Sequence[str]):
        if clf is None or not used_features:
            return np.array([]), pd.DataFrame()
        xq = pd.DataFrame({f: [1] for f in used_features}, dtype="int8")
        proba = clf.predict_proba(xq)[0]
        return np.asarray(proba), xq