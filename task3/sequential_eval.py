# task3/sequential_eval.py
from __future__ import annotations
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from core.data_manager import DataManager
from core.model_trainer import ModelTrainer

# ---------- paths & config ----------
RAW_DATASET_CSV = "data/disease_symptom/dataset.csv"
SEVERITY_CSV    = "data/disease_symptom/Symptom-severity.csv"
PRECAUTIONS_CSV = "data/disease_symptom/symptom_precaution.csv"
TARGET_COL      = "Disease"

TEST_SIZE   = 0.20
RANDOM_SEED = 42
HIT_THRESH  = 0.70          # prob threshold to count as a "confident prediction"

OUTPUT_DIR   = "task3/output"
OUTPUT_CSV   = os.path.join(OUTPUT_DIR, "sequential_eval.csv")
SUMMARY_TXT  = os.path.join(OUTPUT_DIR, "sequential_eval_summary.txt")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---------- load + widen dataset ----------
    dm = DataManager(
        raw_dataset_csv=RAW_DATASET_CSV,
        severity_csv=SEVERITY_CSV,
        precautions_csv=PRECAUTIONS_CSV,
        target_col=TARGET_COL,
    )

    df_wide, symptom_cols, sev_groups, _ = dm.load_all()

    # severity map: symptom -> level (1..7)
    sym_to_level: Dict[str, int] = {}
    for lvl, syms in sev_groups.items():
        for s in syms:
            sym_to_level[s] = lvl

    # ---------- train / test split ----------
    train_idx, test_idx = train_test_split(
        df_wide.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=df_wide[TARGET_COL],
    )

    df_train = df_wide.loc[train_idx].reset_index(drop=True)
    df_test  = df_wide.loc[test_idx].reset_index(drop=True)

    trainer = ModelTrainer()

    rows_summary: List[dict] = []

    # ---------- sequential evaluation ----------
    for i, row in df_test.iterrows():
        true_dz = row[TARGET_COL]

        # list of present symptoms for this case
        present_syms = [
            c for c in symptom_cols
            if c in df_test.columns and int(row[c]) == 1
        ]
        if not present_syms:
            continue

        # sort by severity level 7 -> 1 (default lvl=1 if missing)
        present_syms_sorted = sorted(
            present_syms,
            key=lambda s: sym_to_level.get(s, 1),
            reverse=True,
        )

        selected: List[str] = []

        # first step where we made *any* confident prediction (correct or wrong)
        conf_step = None
        conf_level = None
        conf_prob = None
        conf_pred = None

        # first step where we made a confident and correct prediction
        hit_step = None
        hit_level = None
        hit_prob = None
        hit_pred = None

        step = 0
        for sym in present_syms_sorted:
            step += 1
            selected.append(sym)

            # train tree on TRAIN set using current selected symptoms
            clf, used_feats, labels = trainer.train_tree(
                df_wide=df_train,
                target_col=TARGET_COL,
                selected_present=selected,
                max_depth=4,
                min_leaf=2,
                seed=RANDOM_SEED,
            )
            if clf is None or not used_feats or not labels:
                continue

            probs, _ = trainer.predict_tree(clf, used_feats)
            if probs is None or len(probs) == 0:
                continue

            probs_arr = np.asarray(probs)
            best_idx = int(np.argmax(probs_arr))
            pred_dz  = labels[best_idx]
            pred_prob = float(probs_arr[best_idx])

            # first time we make ANY confident prediction (regardless of correctness)
            if conf_step is None and pred_prob >= HIT_THRESH:
                conf_step  = step
                conf_level = sym_to_level.get(sym, 1)
                conf_prob  = pred_prob
                conf_pred  = pred_dz

            # first time we confidently hit the correct disease
            if (
                    hit_step is None
                    and pred_dz == true_dz
                    and pred_prob >= HIT_THRESH
            ):
                hit_step  = step
                hit_level = sym_to_level.get(sym, 1)
                hit_prob  = pred_prob
                hit_pred  = pred_dz

        rows_summary.append(
            {
                "test_row": i,
                "true_disease": true_dz,
                "n_present_symptoms": len(present_syms),

                # first *correct* confident hit
                "first_hit_step": hit_step,
                "first_hit_severity_level": hit_level,
                "first_hit_prob": hit_prob,
                "first_hit_pred_disease": hit_pred,
                "hit_found": hit_step is not None,

                # first confident prediction (correct OR wrong)
                "conf_step": conf_step,
                "conf_severity_level": conf_level,
                "conf_prob": conf_prob,
                "conf_pred_disease": conf_pred,
                "confident_pred_made": conf_step is not None,
            }
        )

    df_out = pd.DataFrame(rows_summary)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved sequential evaluation to: {OUTPUT_CSV}")
    print(df_out.head())

    # ---------- accuracy from "confident prediction"  ----------
    # Only look at rows where we actually made a confident prediction
    mask_conf = df_out["confident_pred_made"]
    n_conf = int(mask_conf.sum())
    n_cases = len(df_out)

    if n_conf > 0:
        y_true_conf = df_out.loc[mask_conf, "true_disease"]
        y_pred_conf = df_out.loc[mask_conf, "conf_pred_disease"]

        acc_conf = accuracy_score(y_true_conf, y_pred_conf)
        prec_macro = precision_score(y_true_conf, y_pred_conf, average="macro", zero_division=0)
        rec_macro  = recall_score(y_true_conf, y_pred_conf, average="macro", zero_division=0)
        f1_macro   = f1_score(y_true_conf, y_pred_conf, average="macro", zero_division=0)
        f1_micro   = f1_score(y_true_conf, y_pred_conf, average="micro", zero_division=0)
    else:
        acc_conf   = 0.0
        prec_macro = 0.0
        rec_macro  = 0.0
        f1_macro   = 0.0
        f1_micro   = 0.0

    # ---------- "hit" accuracy: did we ever hit the right disease confidently? ----------
    n_hits = int(df_out["hit_found"].sum()) if n_cases > 0 else 0
    hit_accuracy = n_hits / n_cases if n_cases > 0 else 0.0

    # ---------- average % of symptoms used before first *correct* hit ----------
    hit_rows = df_out[df_out["hit_found"]]
    if len(hit_rows) > 0:
        pct_used = (
                hit_rows["first_hit_step"] / hit_rows["n_present_symptoms"]
        ).astype(float)
        avg_pct_used = float(pct_used.mean() * 100.0)
        median_pct_used = float(pct_used.median() * 100.0)
    else:
        avg_pct_used = 0.0
        median_pct_used = 0.0

    # ---------- write summary txt ----------
    with open(SUMMARY_TXT, "w") as f:
        f.write("Sequential Evaluation Summary\n")
        f.write("================================\n")
        f.write(f"Total test cases: {n_cases}\n")
        f.write(f"Cases with a confident prediction (prob ≥ {HIT_THRESH:.2f}): {n_conf}\n")
        f.write("\n")
        f.write("Classification metrics on cases with a confident prediction:\n")
        f.write(f"- Accuracy:       {acc_conf * 100:.2f}%\n")
        f.write(f"- Precision (macro): {prec_macro * 100:.2f}%\n")
        f.write(f"- Recall (macro):    {rec_macro * 100:.2f}%\n")
        f.write(f"- F1 (macro):        {f1_macro * 100:.2f}%\n")
        f.write(f"- F1 (micro):        {f1_micro * 100:.2f}%\n")
        f.write("\n")
        f.write("Hit-based metric (did we ever confidently reach the correct disease?):\n")
        f.write(f"- Hit accuracy (any step, correct & prob ≥ {HIT_THRESH:.2f}): {hit_accuracy * 100:.2f}%\n")
        f.write("\n")
        f.write("Among cases where a confident *correct* hit was found:\n")
        f.write(f"- Average % of symptoms used before first hit: {avg_pct_used:.2f}%\n")
        f.write(f"- Median  % of symptoms used before first hit: {median_pct_used:.2f}%\n")

    print(f"Saved summary metrics to: {SUMMARY_TXT}")


if __name__ == "__main__":
    main()