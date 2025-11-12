from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
import os
import re
import numpy as np
import pandas as pd
import streamlit as st  # for cached loaders

_WS_RE = re.compile(r"\s+")

def _norm_sym(s: str) -> str:
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

class DataManager:
    def __init__(_self, raw_dataset_csv: str, severity_csv: str, precautions_csv: str, target_col: str):
        _self.raw_dataset_csv = raw_dataset_csv
        _self.severity_csv = severity_csv
        _self.precautions_csv = precautions_csv
        _self.target_col = target_col

    # ---------- cached loaders ----------
    @st.cache_resource(show_spinner=False)
    def _load_wide_and_groups(_self) -> Tuple[pd.DataFrame, List[str], Dict[int, List[str]]]:
        if not os.path.exists(_self.raw_dataset_csv):
            st.error(f"Dataset not found at {_self.raw_dataset_csv}")
            st.stop()
        if not os.path.exists(_self.severity_csv):
            st.error(f"Severity file not found at {_self.severity_csv}")
            st.stop()

        df_raw = pd.read_csv(_self.raw_dataset_csv)
        df_wide, symptom_cols = _self.widen_symptom_dataset(df_raw, target_col=_self.target_col)
        groups = _self.build_severity_groups_from_csv(symptom_cols, _self.severity_csv)
        return df_wide, symptom_cols, groups

    @st.cache_resource(show_spinner=False)
    def _load_precautions_map(_self) -> Dict[str, List[str]]:
        try:
            if not os.path.exists(_self.precautions_csv):
                return {}
            dfp = pd.read_csv(_self.precautions_csv)
            cols = {c.lower(): c for c in dfp.columns}
            dz_col = cols.get("disease", "Disease")
            prec_cols = [c for c in dfp.columns if str(c).lower().startswith("precaution")]
            out = {}
            for _, r in dfp.iterrows():
                dz = str(r.get(dz_col, "")).strip()
                if not dz:
                    continue
                vals = []
                for c in prec_cols:
                    v = r.get(c)
                    if pd.isna(v):
                        continue
                    s = str(v).strip()
                    if not s or s.lower() == "none":
                        continue
                    vals.append(s)
                out[dz] = vals
            return out
        except Exception:
            return {}

    # ---------- public API ----------
    def load_all(_self):
        df_wide, symptom_cols, sev_groups = _self._load_wide_and_groups()
        prec_map = _self._load_precautions_map()
        return df_wide, symptom_cols, sev_groups, prec_map

    def widen_symptom_dataset(_self, df_raw: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, List[str]]:
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

    def build_severity_groups_from_csv(_self, feature_cols: Sequence[str], severity_csv: str) -> Dict[int, List[str]]:
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

        seen = set()
        for lvl in range(7, 0, -1):
            dedup = []
            for s in groups[lvl]:
                if s not in seen:
                    dedup.append(s)
                    seen.add(s)
            groups[lvl] = sorted(dedup, key=str.lower)

        return groups

    def filter_present(_self, df_wide: pd.DataFrame, selected_present: Iterable[str], *, target_col: str, relax: bool=True):
        feats = [c for c in selected_present if c in df_wide.columns]
        if not feats:
            return df_wide.copy(), [], "No present symptoms selected yet."

        X = df_wide[feats].astype("int8")
        required = len(feats)
        note = f"Filtered: rows having ALL {required}/{required} selected symptoms."

        mask = (X.sum(axis=1) == required)
        df_f = df_wide.loc[mask]
        if len(df_f) == 0 and relax:
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

    def pool_limited_options(_self, level_syms: List[str], df_filt, already_selected: Set[str]) -> List[str]:
        if df_filt.empty or not level_syms:
            return []
        cols_available = [c for c in level_syms if c in df_filt.columns]
        if not cols_available:
            return []
        prev = df_filt[cols_available].astype("int8").mean(axis=0)
        keep = [c for c in cols_available if float(prev.get(c, 0.0)) > 0.0 and c not in already_selected]
        return sorted(keep, key=str.lower)