# task3/pipeline.py
# Data cleaning, widening, severity grouping, on-confirm tree training,
# row filtering by confirmed symptoms, and conditional-symptom suggestions.

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# --------------------------
# Normalisation helpers
# --------------------------

_WS_RE = re.compile(r"\s+")


def _norm_sym(s: str) -> str:
    """Normalise a symptom token to snake_case and strip weird whitespace."""
    if s is None:
        return ""
    try:
        if pd.isna(s):
            return ""
    except Exception:
        pass

    s = str(s)
    s = s.replace("\xa0", " ").replace("\u200b", " ").replace("\ufeff", " ")
    s = s.strip().lower()
    s = _WS_RE.sub(" ", s)
    s = re.sub(r"\s*_\s*", "_", s)
    s = s.replace(" ", "_")
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def _clean_symptom_cols(df: pd.DataFrame, sym_cols: Sequence[str]) -> pd.DataFrame:
    """Strip junk chars and harmonise spacing in Symptom_* columns."""
    df = df.copy()
    for c in sym_cols:
        s = df[c].astype("string")
        s = (
            s.str.replace("\xa0", " ", regex=False)
            .str.replace("\u200b", " ", regex=False)
            .str.replace("\ufeff", " ", regex=False)
            .str.strip()
        )
        s = s.str.replace(r"\s+", " ", regex=True)
        s = s.str.replace(r"\s*_\s*", "_", regex=True)
        s = s.str.replace(r"[;,]\s*", " ", regex=True)
        s = s.replace("", np.nan)
        df[c] = s
    return df


# --------------------------
# Widen to one-hot
# --------------------------

def widen_symptom_dataset(
        df_raw: pd.DataFrame,
        target_col: str = "Disease",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Convert Symptom_1..Symptom_k into a wide one-hot table (0/1) per unique
    normalised symptom token. Leaves the target column as-is.
    """
    sym_cols = [c for c in df_raw.columns if str(c).lower().startswith("symptom")]
    if not sym_cols:
        raise ValueError("No Symptom_* columns found in dataset.")
    if target_col not in df_raw.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    df = df_raw.copy()
    df[target_col] = df[target_col].astype(str).str.strip().str.replace("\xa0", " ", regex=False)
    df = _clean_symptom_cols(df, sym_cols)

    all_syms: Set[str] = set()
    rows_tokens: List[List[str]] = []
    for _, r in df[sym_cols].iterrows():
        toks = []
        for v in r.tolist():
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            tok = _norm_sym(v)
            if tok:
                toks.append(tok)
        toks = sorted(set(toks))
        rows_tokens.append(toks)
        all_syms.update(toks)

    for bad in ("", "nan", "none", "na"):
        all_syms.discard(bad)

    all_syms = sorted(all_syms)
    wide = pd.DataFrame(0, index=df.index, columns=all_syms, dtype="int8")
    for i, toks in enumerate(rows_tokens):
        if toks:
            idx = wide.columns.get_indexer(toks)
            wide.iloc[i, idx] = 1

    wide[target_col] = df[target_col].values
    return wide, all_syms


# --------------------------
# Severity groups
# --------------------------

def build_severity_groups_from_csv(
        feature_cols: Sequence[str],
        severity_csv: str,
) -> Dict[int, List[str]]:
    """
    Map symptoms → severity levels 1..7 from the provided CSV.
    Any symptom not listed goes to level 1 by default.
    """
    df = pd.read_csv(severity_csv)
    cols_l = {c.lower(): c for c in df.columns}
    sym_col = cols_l.get("symptom", list(df.columns)[0])
    w_col = cols_l.get("weight", list(df.columns)[1] if len(df.columns) > 1 else list(df.columns)[0])

    df = df.rename(columns={sym_col: "Symptom", w_col: "weight"})
    df["Symptom_norm"] = df["Symptom"].apply(_norm_sym)
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(1).astype(int).clip(1, 7)

    norm_to_feat = {_norm_sym(c): c for c in feature_cols}
    groups: Dict[int, List[str]] = {lvl: [] for lvl in range(1, 8)}
    for _, r in df.iterrows():
        real = norm_to_feat.get(r["Symptom_norm"])
        if real:
            groups[int(r["weight"])].append(real)

    covered = set(sum(groups.values(), []))
    for c in feature_cols:
        if c not in covered:
            groups[1].append(c)

    # Deduplicate keeping highest severity
    seen = set()
    for lvl in range(7, 0, -1):
        dedup = []
        for s in groups[lvl]:
            if s not in seen:
                dedup.append(s)
                seen.add(s)
        groups[lvl] = sorted(dedup, key=str.lower)

    return groups


# --------------------------
# Selection helpers
# --------------------------

def union_selected(selected_by_level: Dict[int, Set[str]]) -> Set[str]:
    out: Set[str] = set()
    for picks in (selected_by_level or {}).values():
        out |= set(picks)
    return out


# --------------------------
# On-confirm: train a tree using ONLY present symptoms
# --------------------------

def train_tree_for_selection(
        df_wide: pd.DataFrame,
        target_col: str,
        selected_present: Iterable[str],
        *,
        max_depth: int = 4,
        min_samples_leaf: int = 2,
        random_state: int = 42,
) -> Tuple[Optional[DecisionTreeClassifier], List[str], List[str]]:
    """
    Fit a small DecisionTree on **only** the columns in `selected_present`.
    If none of the selected symptoms exist in df columns, returns (None, [], []).
    Returns: (clf, used_feature_cols, class_labels)
    """
    features = [c for c in selected_present if c in df_wide.columns]
    if not features:
        return None, [], []

    X = df_wide[features].astype("int8")
    y = df_wide[target_col].astype(str)

    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=random_state,
    )
    clf.fit(X, y)
    return clf, features, clf.classes_.tolist()


def predict_from_tree(
        clf: DecisionTreeClassifier,
        used_features: Sequence[str],
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Predict probabilities for a single synthetic row where all used_features are 1
    (since the user marked these **present**). Returns (proba, X_query_df).
    """
    if clf is None or not used_features:
        return np.array([]), pd.DataFrame()

    xq = pd.DataFrame({f: [1] for f in used_features}, dtype="int8")
    proba = clf.predict_proba(xq)[0]
    return np.asarray(proba), xq


# --------------------------
# Row filtering & suggestions
# --------------------------

def filter_rows_by_present(
        df_wide: pd.DataFrame,
        selected_present: Iterable[str],
        *,
        target_col: str = "Disease",
        relax_if_empty: bool = True,
) -> Tuple[pd.DataFrame, List[str], str]:
    """
    Keep rows where **all** selected symptoms are 1.
    If that yields 0 rows and relax_if_empty=True, progressively relax to
    'at least k-1', 'k-2', ... matches until non-empty. Returns:
      (filtered_df, used_sym_cols, note)
    """
    feats = [c for c in selected_present if c in df_wide.columns]
    if not feats:
        return df_wide.copy(), [], "No present symptoms selected yet."

    X = df_wide[feats].astype("int8")
    required = len(feats)
    note = f"Filtered: rows having ALL {required}/{required} selected symptoms."

    mask = (X.sum(axis=1) == required)
    df_f = df_wide.loc[mask]
    if len(df_f) == 0 and relax_if_empty:
        # Relax progressively
        for k in range(required - 1, 0, -1):
            mask = (X.sum(axis=1) >= k)
            df_f = df_wide.loc[mask]
            if len(df_f) > 0:
                note = f"Relaxed filter: rows having ≥{k}/{required} selected symptoms."
                break
        if len(df_f) == 0:
            df_f = df_wide.copy()
            note = "Relaxed fully: no rows matched, falling back to full dataset."

    return df_f, feats, note


def suggest_additional_symptoms(
        df_filtered: pd.DataFrame,
        all_symptom_cols: Sequence[str],
        already_selected: Set[str],
        *,
        top_k: int = 10,
        restrict_to: Optional[Iterable[str]] = None,
) -> List[Tuple[str, float]]:
    """
    Rank symptoms by conditional prevalence within the filtered subset:
      score(s) = mean(df_filtered[s] == 1).
    Excludes anything already_selected. If restrict_to is provided, only score
    those columns.
    """
    if df_filtered.empty:
        return []

    cols = list(restrict_to) if restrict_to is not None else list(all_symptom_cols)
    cols = [c for c in cols if c in df_filtered.columns and c not in already_selected]
    if not cols:
        return []

    # Prevalence as float
    prev = df_filtered[cols].astype("int8").mean(axis=0)
    ranked = prev.sort_values(ascending=False)
    out = [(c, float(ranked[c])) for c in ranked.index[:top_k]]
    return out