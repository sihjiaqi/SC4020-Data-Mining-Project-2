from __future__ import annotations
from typing import Dict, Set, List, Tuple
import numpy as np
import pandas as pd
import streamlit as st

class UIManager:
    def __init__(_self, backend, default_conf_thresh: float, min_display_prob: float):
        _self.backend = backend
        _self.default_conf_thresh = float(default_conf_thresh)
        _self.min_display_prob = float(min_display_prob)

    # ---------- small UI utils ----------
    @staticmethod
    def _pretty(s: str) -> str:
        return s.replace("_", " ")

    def _ensure_state(_self, df_wide_rows: int):
        s = st.session_state
        if "stage_level" not in s:
            s.stage_level = 7  # start at most severe
        if "selected_by_level" not in s:
            s.selected_by_level = {lvl: set() for lvl in range(1, 8)}
        if "last_pred" not in s:
            s.last_pred = None  # (labels, probs)
        if "last_features" not in s:
            s.last_features = []
        if "conf_thresh" not in s:
            s.conf_thresh = _self.default_conf_thresh
        if "filter_note" not in s:
            s.filter_note = "No present symptoms selected yet."
        if "df_filtered_rows" not in s:
            s.df_filtered_rows = int(df_wide_rows)
        if "last_suggestions" not in s:
            s.last_suggestions = []  # list[(symptom, score_float)]

        # new final-page state
        if "finished" not in s:
            s.finished = False
        if "final_disease" not in s:
            s.final_disease = None
        if "final_prob" not in s:
            s.final_prob = None

    def _render_final_page(_self):
        dz = st.session_state.final_disease
        p = st.session_state.final_prob

        st.title("Diagnosis summary")

        if not dz:
            st.info("No final diagnosis is available. Please start a new assessment.")
        else:
            st.success(f"Our model is confident that the most likely disease is: {dz}")
            if p is not None:
                st.write(f"Estimated confidence: {p:.2%}")

            precs = _self.backend.precautions_for(dz)
            if precs:
                st.subheader("Suggested precautions")
                for i, txt in enumerate(precs, 1):
                    st.write(f"{i}. {txt}")

        st.divider()
        if st.button("Start a new assessment"):
            # simple reset of all state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # ---------- main page ----------
    def run(_self):
        # Load (cached inside backend.data_manager)
        df_wide, symptom_cols, sev_groups, _ = _self.backend.data_manager.load_all()

        _self._ensure_state(df_wide_rows=len(df_wide))

        if st.session_state.finished:
            _self._render_final_page()
            return

        st.title("Stage 1: Symptom-based Progressive Diagnosis")
        st.caption("Pick symptoms progressively from most severe to least severe. Confirm each level to continue.")

        left, right = st.columns([2.1, 1.1])

        # ----- LEFT: levels & selections -----
        with left:
            cur_level = int(st.session_state.stage_level)
            title = f"Level {cur_level} " + ("(most severe)" if cur_level == 7 else "(least severe)" if cur_level == 1 else "(moderate)")
            st.subheader(title)

            view = _self.backend.build_level_view(
                level=cur_level,
                selected_by_level=st.session_state.selected_by_level,
            )

            # Show info if needed
            if view["info_message"]:
                st.info(view["info_message"])

            # Chips
            prev = st.session_state.selected_by_level.get(cur_level, set())
            picked: Set[str] = set()
            opts = view["limited_options"]
            cols = st.columns(4) if opts else [st]
            for i, opt in enumerate(opts):
                with cols[i % len(cols)]:
                    checked = st.checkbox(_self._pretty(opt), value=(opt in prev), key=f"chip_{cur_level}_{opt}")
                    if checked:
                        picked.add(opt)

            c1, c2, c3 = st.columns([1, 1, 2])
            with c1:
                if st.button(f"Confirm Level {cur_level}", type="primary"):
                    res = _self.backend.confirm_level(
                        level=cur_level,
                        picked=picked,
                        selected_by_level=st.session_state.selected_by_level,
                        conf_thresh=float(st.session_state.conf_thresh),
                    )

                    # Reflect into session_state
                    st.session_state.selected_by_level = res["selected_by_level"]
                    st.session_state.stage_level = res["next_level"]
                    st.session_state.filter_note = res["filter_note"]
                    st.session_state.df_filtered_rows = res["rows_after_filter"]
                    st.session_state.last_pred = res["last_pred"]   # (labels, probs) or None
                    st.session_state.last_features = res["features_used"]
                    st.session_state.last_suggestions = res["suggestions"]

                    # If the backend says we are confident, lock the flow and go to final page
                    if res.get("confident"):
                        st.session_state.finished = True
                        st.session_state.final_disease = res.get("top_disease")
                        st.session_state.final_prob = res.get("top_prob")

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
                    st.write(f"Level {lvl}: " + ", ".join(_self._pretty(p) for p in picks))

            # “You might also be experiencing” — only if ≥1 selected and we have suggestions
            if any(st.session_state.selected_by_level.values()) and st.session_state.last_suggestions:
                st.divider()
                st.markdown("**You might also be experiencing**")
                top5 = [_self._pretty(s) for s, _ in st.session_state.last_suggestions[:5]]
                if top5:
                    st.markdown("\n".join(f"{i}. {name}" for i, name in enumerate(top5, start=1)))

            st.divider()

        # ----- RIGHT: predictions & diagnostics -----
        with right:
            st.subheader("Top predictions (≥ 0.70)")
            if st.session_state.last_pred is None:
                st.info("No predictions yet. Press **Confirm Level …** to train and score based on your current confirmed selections.")
            else:
                labels, probs = st.session_state.last_pred
                if getattr(probs, "size", 0):
                    order = np.argsort(probs)[::-1]
                    rows = [(labels[i], float(probs[i])) for i in order if float(probs[i]) >= _self.min_display_prob]
                    if rows:
                        table = pd.DataFrame({"Disease": [d for d, _ in rows[:10]], "Probability": [p for _, p in rows[:10]]})
                        st.table(table)
                    else:
                        st.info(f"No predictions ≥ {_self.min_display_prob:.2f} yet.")

                    pmax_idx = int(order[0])
                    top_disease = labels[pmax_idx]
                    pmax = float(probs[pmax_idx])

                    if pmax >= float(st.session_state.conf_thresh):
                        st.success("We’re confident enough to stop here.")
                        precs = _self.backend.precautions_for(top_disease)
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
                        st.write(", ".join(_self._pretty(f) for f in sorted(st.session_state.last_features, key=str.lower)))

            st.divider()
            st.caption("Row filter status")
            note_live = _self.backend.live_filter_note(st.session_state.selected_by_level)
            st.write(note_live if any(st.session_state.selected_by_level.values()) else st.session_state.filter_note)
            st.write(f"Rows remaining after filter: **{st.session_state.df_filtered_rows}**")