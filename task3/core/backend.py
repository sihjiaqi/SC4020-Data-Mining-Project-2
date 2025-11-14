from __future__ import annotations
from typing import Dict, Set
import numpy as np


class BackendLogic:
    def __init__(_self, data_manager, model_manager, target_col: str):
        _self.data_manager = data_manager
        _self.model_manager = model_manager
        _self.target_col = str(target_col)

        # Cache loaded artifacts
        _self._loaded = None  # (df_wide, symptom_cols, sev_groups, precautions_map)

    # --------- accessors ----------
    def _ensure_loaded(_self):
        if _self._loaded is None:
            _self._loaded = _self.data_manager.load_all()


    def precautions_for(_self, disease: str):
        _self._ensure_loaded()
        _, _, _, prec_map = _self._loaded
        return prec_map.get(disease, [])

    # --------- live helpers ----------
    def live_filter_note(_self, selected_by_level: Dict[int, Set[str]]) -> str:
        _self._ensure_loaded()
        df_wide, _, _, _ = _self._loaded
        selected = set().union(*selected_by_level.values())
        _, _, note = _self.data_manager.filter_present(
            df_wide=df_wide,
            selected_present=selected,
            target_col=_self.target_col,
            relax=True,
        )
        return note

    # --------- UI building ----------
    def build_level_view(_self, level: int, selected_by_level: Dict[int, Set[str]]):
        _self._ensure_loaded()
        df_wide, symptom_cols, sev_groups, _ = _self._loaded

        selected = set().union(*selected_by_level.values())
        df_filt, _, _ = _self.data_manager.filter_present(
            df_wide=df_wide,
            selected_present=selected,
            target_col=_self.target_col,
            relax=True,
        )

        level_syms = sev_groups.get(level, [])
        limited = _self.data_manager.pool_limited_options(level_syms, df_filt, already_selected=selected)

        if not level_syms:
            info = "No symptoms tagged at this level."
        elif not limited:
            info = "No plausible options at this level given your current selections."
        else:
            info = ""

        return {
            "limited_options": limited,
            "info_message": info,
        }

    def confirm_level(_self, level: int, picked: Set[str], selected_by_level: Dict[int, Set[str]], conf_thresh: float):
        _self._ensure_loaded()
        df_wide, symptom_cols, _, _ = _self._loaded

        # persist picks, move to next level
        selected_by_level[level] = set(picked)
        next_level = max(1, level - 1)

        # recompute filter
        selected = set().union(*selected_by_level.values())
        df_filt, used_feats, note = _self.data_manager.filter_present(
            df_wide=df_wide,
            selected_present=selected,
            target_col=_self.target_col,
            relax=True,
        )

        # model scoring
        labels, probs, features_used = _self.model_manager.score_selected(
            df_wide=df_wide,
            target_col=_self.target_col,
            selected_present=selected,
        )

        # suggestions (names only; UI will cap to 5)
        suggestions = _self.model_manager.suggest_next(
            df_filtered=df_filt,
            all_symptoms=symptom_cols,
            already_selected=selected,
            top_k=10,
        )

        last_pred = None if probs is None else (labels, probs)

        # decide if we are confident enough to stop
        confident = False
        top_disease = None
        top_prob = None
        if probs is not None and len(probs):
            probs_arr = np.asarray(probs)
            best_idx = int(np.argmax(probs_arr))
            top_prob = float(probs_arr[best_idx])
            top_disease = labels[best_idx]
            if top_prob >= float(conf_thresh):
                confident = True

        return {
            "selected_by_level": selected_by_level,
            "next_level": next_level,
            "filter_note": note,
            "rows_after_filter": int(len(df_filt)),
            "last_pred": last_pred,
            "features_used": features_used,
            "suggestions": suggestions,
            "confident": confident,
            "top_disease": top_disease,
            "top_prob": top_prob,
        }