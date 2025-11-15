from __future__ import annotations
import streamlit as st
import torch

from core.ui_manager import UIManager
from core.backend import BackendLogic
from core.data_manager import DataManager
from core.model_manager import ModelManager
from core.model_trainer import ModelTrainer
from core.predictors.symptom_predictor import SymptomPredictor
from core.triage_llm import DepartmentLLMRouter


# ---- Paths (same as your current ones) ----
RAW_DATASET_CSV = "data/disease_symptom/dataset.csv"
SEVERITY_CSV    = "data/disease_symptom/Symptom-severity.csv"
PRECAUTIONS_CSV = "data/disease_symptom/symptom_precaution.csv"
TARGET_COL      = "Disease"

DEFAULT_CONF_THRESH = 0.80
PREDICTION_DISPLAY_MIN = 0.70

st.set_page_config(page_title="Symptom-based Progressive Diagnosis", layout="wide")

# ---- Wiring the pieces ----
data_mgr = DataManager(
    raw_dataset_csv=RAW_DATASET_CSV,
    severity_csv=SEVERITY_CSV,
    precautions_csv=PRECAUTIONS_CSV,
    target_col=TARGET_COL,
)

symptom_pred = SymptomPredictor()
trainer = ModelTrainer()
model_mgr = ModelManager(trainer=trainer, symptom_pred=symptom_pred, min_display_prob=PREDICTION_DISPLAY_MIN)

department_router = DepartmentLLMRouter(
    model_name="Qwen/Qwen2-7B-Instruct",
    use_llm=True,
    device="cpu",
)

backend = BackendLogic(
    data_manager=data_mgr,
    model_manager=model_mgr,
    target_col=TARGET_COL,
    department_router=department_router,   # <-- make sure this is here
)

ui = UIManager(
    backend=backend,
    default_conf_thresh=DEFAULT_CONF_THRESH,
    min_display_prob=PREDICTION_DISPLAY_MIN,
)

ui.run()