# task3/app.py
# Stage 1 UI with pool-limited options:
# - Options at the current level are restricted to symptoms that appear in the
#   filtered dataset (rows that match *confirmed* present symptoms).
# - No fallback: if none appear, chips list is empty.
# - Predictions table shows only rows with prob ≥ 0.70.
# - "You might also be experiencing" lists names (no probs), across ALL levels,
#   and only appears if the user has selected at least one symptom.
# - When confident (top prob ≥ threshold), show precautions for top disease.

from __future__ import annotations

import os
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from pipeline import (
    widen_symptom_dataset,
    build_severity_groups_from_csv,
    union_selected,
    train_tree_for_selection,
    predict_from_tree,
    filter_rows_by_present,
    suggest_additional_symptoms,
)

# --------------------------
# App config
# --------------------------
st.set_page_config(page_title="Symptom-based Progressive Diagnosis", layout="wide")

RAW_DATASET_CSV = "data/disease_symptom/dataset.csv"
SEVERITY_CSV    = "data/disease_symptom/Symptom-severity.csv"
PRECAUTIONS_CSV = "data/disease_symptom/symptom_precaution.csv"
TARGET_COL      = "Disease"

DEFAULT_CONF_THRESH = 0.80
PREDICTION_DISPLAY_MIN = 0.70   # only show predictions ≥ 0.70


# --------------------------
# Loaders (cached)
# --------------------------
@st.cache_resource(show_spinner=False)
def load_wide_and_groups() -> Tuple[pd.DataFrame, List[str], Dict[int, List[str]]]:
    if not os.path.exists(RAW_DATASET_CSV):
        st.error(f"Dataset not found at {RAW_DATASET_CSV}")
        st.stop()
    if not os.path.exists(SEVERITY_CSV):
        st.error(f"Severity file not found at {SEVERITY_CSV}")
        st.stop()

    df_raw = pd.read_csv(RAW_DATASET_CSV)
    df_wide, symptom_cols = widen_symptom_dataset(df_raw, target_col=TARGET_COL)
    groups = build_severity_groups_from_csv(symptom_cols, SEVERITY_CSV)
    return df_wide, symptom_cols, groups


@st.cache_resource(show_spinner=False)
def load_precautions_map(csv_path: str = PRECAUTIONS_CSV) -> dict[str, list[str]]:
    """
    Returns: { disease_name -> [prec1, prec2, ...] } with 'None' filtered out.
    If file is missing or malformed, returns {}.
    """
    try:
        if not os.path.exists(csv_path):
            return {}
        dfp = pd.read_csv(csv_path)
        cols = {c.lower(): c for c in dfp.columns}
        dz_col = cols.get("disease", "Disease")
        prec_cols = [c for c in dfp.columns if str(c).lower().startswith("precaution")]
        out: dict[str, list[str]] = {}
        for _, r in dfp.iterrows():
            dz = str(r.get(dz_col, "")).strip()
            if not dz:
                continue
            vals: list[str] = []
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


df_wide, symptom_cols, sev_groups = load_wide_and_groups()
precautions_map = load_precautions_map()


# --------------------------
# Session state
# --------------------------
def ensure_state():
    if "stage_level" not in st.session_state:
        st.session_state.stage_level = 7  # start at most severe
    if "selected_by_level" not in st.session_state:
        st.session_state.selected_by_level = {lvl: set() for lvl in range(1, 8)}
    if "last_pred" not in st.session_state:
        st.session_state.last_pred = None  # (labels, probs)
    if "last_features" not in st.session_state:
        st.session_state.last_features = []
    if "conf_thresh" not in st.session_state:
        st.session_state.conf_thresh = DEFAULT_CONF_THRESH
    if "filter_note" not in st.session_state:
        st.session_state.filter_note = "No present symptoms selected yet."
    if "df_filtered_rows" not in st.session_state:
        st.session_state.df_filtered_rows = len(df_wide)
    if "last_suggestions" not in st.session_state:
        st.session_state.last_suggestions = []  # List[(symptom, prevalence_float)] — we will display names only

ensure_state()


def pretty(s: str) -> str:
    return s.replace("_", " ")


# --------------------------
# Helpers used each render
# --------------------------
def filtered_subset_for_current_selection() -> Tuple[pd.DataFrame, str]:
    """Build filtered subset using confirmed-present symptoms only."""
    selected_present = union_selected(st.session_state.selected_by_level)
    df_filt, _, note = filter_rows_by_present(
        df_wide=df_wide,
        selected_present=selected_present,
        target_col=TARGET_COL,
        relax_if_empty=True,
    )
    return df_filt, note


def level_options_limited_by_pool(level_syms: List[str], df_filt: pd.DataFrame, already_selected: Set[str]) -> List[str]:
    """
    Keep only symptoms from this level that actually occur in the filtered pool
    (prevalence > 0), excluding anything already selected.
    **No fallback**: if empty, return empty.
    """
    if df_filt.empty or not level_syms:
        return []

    cols_available = [c for c in level_syms if c in df_filt.columns]
    if not cols_available:
        return []

    prev = df_filt[cols_available].astype("int8").mean(axis=0)
    keep = [c for c in cols_available if float(prev.get(c, 0.0)) > 0.0 and c not in already_selected]
    return sorted(keep, key=str.lower)


# --------------------------
# UI
# --------------------------
st.title("Stage 1: Symptom-based Progressive Diagnosis")
st.caption("Pick symptoms progressively from most severe to least severe. Confirm each level to continue.")

left, right = st.columns([2.1, 1.1])

with left:
    cur_level = int(st.session_state.stage_level)
    title = f"Level {cur_level} " + ("(most severe)" if cur_level == 7 else "(least severe)" if cur_level == 1 else "(moderate)")
    st.subheader(title)

    # Build filtered subset *from confirmed selections so far*
    df_filt_live, note_live = filtered_subset_for_current_selection()

    # Limit current level options to those present in the filtered subset (no fallback)
    level_syms = sev_groups.get(cur_level, [])
    already = union_selected(st.session_state.selected_by_level)
    limited_opts = level_options_limited_by_pool(level_syms, df_filt_live, already_selected=already)

    # If that level has no symptoms at all, show nothing (no fallback)
    if not level_syms:
        st.info("No symptoms tagged at this level.")
    elif not limited_opts:
        st.info("No plausible options at this level given your current selections.")

    # Chips (only within limited options)
    prev = st.session_state.selected_by_level.get(cur_level, set())
    picked: Set[str] = set()
    cols = st.columns(4) if limited_opts else [st]
    for i, opt in enumerate(limited_opts):
        with cols[i % len(cols)]:
            checked = st.checkbox(pretty(opt), value=(opt in prev), key=f"chip_{cur_level}_{opt}")
            if checked:
                picked.add(opt)

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        if st.button(f"Confirm Level {cur_level}", type="primary"):
            # Persist picks for this level, move to next
            st.session_state.selected_by_level[cur_level] = set(picked)
            st.session_state.stage_level = max(1, cur_level - 1)

            # Recompute filtered subset using the updated selections
            df_filt, feats_used, note = filter_rows_by_present(
                df_wide=df_wide,
                selected_present=union_selected(st.session_state.selected_by_level),
                target_col=TARGET_COL,
                relax_if_empty=True,
            )
            st.session_state.filter_note = note
            st.session_state.df_filtered_rows = int(len(df_filt))

            # Train a small tree using ONLY present symptoms (on full data)
            clf, used_feats, labels = train_tree_for_selection(
                df_wide=df_wide,
                target_col=TARGET_COL,
                selected_present=union_selected(st.session_state.selected_by_level),
                max_depth=4,
                min_samples_leaf=2,
                random_state=42,
            )
            if clf is None:
                st.session_state.last_pred = None
                st.session_state.last_features = []
            else:
                probs, _ = predict_from_tree(clf, used_feats)
                st.session_state.last_pred = (labels, probs)
                st.session_state.last_features = used_feats

            # Build suggestions across ALL levels (names only; we still compute off filtered subset)
            already2 = union_selected(st.session_state.selected_by_level)
            st.session_state.last_suggestions = suggest_additional_symptoms(
                df_filtered=df_filt,
                all_symptom_cols=symptom_cols,
                already_selected=already2,
                top_k=10,
                restrict_to=None,  # ← include all levels
            )

            st.rerun()

    with c2:
        if cur_level < 7 and st.button("Back"):
            st.session_state.stage_level = min(7, cur_level + 1)
            st.rerun()

    with c3:
        if st.button("Start over"):
            st.session_state.stage_level = 7
            st.session_state.selected_by_level = {lvl: set() for lvl in range(1, 8)}
            st.session_state.last_pred = None
            st.session_state.last_features = []
            st.session_state.filter_note = "No present symptoms selected yet."
            st.session_state.df_filtered_rows = len(df_wide)
            st.session_state.last_suggestions = []
            st.rerun()

    st.divider()
    st.markdown("**Selections so far (confirmed only)**")
    for lvl in range(7, 0, -1):
        picks = sorted(st.session_state.selected_by_level.get(lvl, []), key=str.lower)
        if picks:
            st.write(f"Level {lvl}: " + ", ".join(pretty(p) for p in picks))

    # "You might also be experiencing" — show only if user has selected ≥1 symptom AND we have suggestions
    if union_selected(st.session_state.selected_by_level) and st.session_state.last_suggestions:
        st.divider()
        st.markdown("**You might also be experiencing**")

        top5 = [pretty(s) for s, _ in st.session_state.last_suggestions[:5]]
        if top5:
            st.markdown("\n".join(f"{i}. {name}" for i, name in enumerate(top5, start=1)))



    st.divider()

with right:
    st.subheader("Top predictions (≥ 0.70)")
    if st.session_state.last_pred is None:
        st.info("No predictions yet. Press **Confirm Level …** to train and score based on your current confirmed selections.")
    else:
        labels, probs = st.session_state.last_pred
        if probs.size:
            order = np.argsort(probs)[::-1]
            # Filter to predictions ≥ 0.70
            rows = [(labels[i], float(probs[i])) for i in order if float(probs[i]) >= PREDICTION_DISPLAY_MIN]
            if rows:
                table = pd.DataFrame({"Disease": [d for d, _ in rows[:10]], "Probability": [p for _, p in rows[:10]]})
                st.table(table)
            else:
                st.info(f"No predictions ≥ {PREDICTION_DISPLAY_MIN:.2f} yet.")

            # Confidence banner + precautions (top predicted disease)
            pmax_idx = int(order[0])
            top_disease = labels[pmax_idx]
            pmax = float(probs[pmax_idx])

            if pmax >= float(st.session_state.conf_thresh):
                st.success("We’re confident enough to stop here.")
                # Precautions for the top disease, if available
                precs = precautions_map.get(top_disease, [])
                if precs:
                    st.markdown(f"**Precautions for {top_disease}:**")
                    for i, ptxt in enumerate(precs, 1):
                        st.write(f"{i}. {ptxt}")
            else:
                st.caption("Keep confirming lower-severity levels to refine the prediction.")
        else:
            st.info("No predictions available with the current selection.")

        if st.session_state.last_features:
            with st.expander("Features used in the current model (present symptoms only)", expanded=False):
                st.write(", ".join(pretty(f) for f in sorted(st.session_state.last_features, key=str.lower)))

    st.divider()
    st.caption("Row filter status")
    # Show the live filter note for transparency even before confirming
    _, note_live = filtered_subset_for_current_selection()
    st.write(note_live if union_selected(st.session_state.selected_by_level) else st.session_state.filter_note)
    st.write(f"Rows remaining after filter: **{st.session_state.df_filtered_rows}**")