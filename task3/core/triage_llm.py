# core/triage_llm.py
from __future__ import annotations
from typing import List, Optional
import re

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False


class DepartmentLLMRouter:
    """
    Memory-safe, lazy-loading LLM-based department router for Streamlit Cloud.

    - Does NOT load the HF model in __init__.
    - Loads only on first call to route_department().
    - Uses low_cpu_mem_usage + device_map="cpu" + torch_dtype="auto".
    - If anything fails (OOM, HF error, etc.), falls back to 'General Medicine'.
    """

    def __init__(
            self,
            model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            use_llm: bool = True,
            departments: Optional[List[str]] = None,
            device: str = "cpu",
    ):
        # Save config
        self.model_name = model_name
        self.device = device
        self.use_llm = bool(use_llm and _HAS_TRANSFORMERS)

        # Will be created lazily
        self.tokenizer: Optional["AutoTokenizer"] = None
        self.model: Optional["AutoModelForCausalLM"] = None
        self._model_loaded_ok = False   # track if load ever succeeded

        # --- departments actually exposed in the app ---
        self.departments = departments or [
            "Emergency",                    # acute, unstable, red flags
            "General Medicine",             # GP / internal medicine
            "Infectious Disease",           # dengue, TB, AIDS, etc.
            "Respiratory",                  # asthma, pneumonia
            "Gastroenterology / Hepatology",# gut & liver
            "Cardiology",                   # heart, hypertension
            "Neurology",                    # stroke, migraine
            "Endocrinology",                # diabetes, thyroid
            "Dermatology",                  # skin
            "ENT (Ear, Nose, Throat)",      # sinus, ear, throat
            "Ophthalmology",                # eyes
            "Orthopaedics",                 # bones & joints
            "Obstetrics & Gynaecology",     # pregnancy, gynae
            "Urology",                      # urinary tract, prostate
            "Psychiatry",                   # mental health
        ]

        # Build the static part of the prompt (without disease/symptoms yet)
        depts_block = "\n".join(f"- {d}" for d in self.departments)
        self.base_prompt_header = f"""
You are a medical referral classifier.

Your job:
- The user will give you a DIAGNOSED DISEASE NAME (not raw symptoms).
- You must choose exactly ONE most appropriate hospital department.
- Choose ONLY from this list:

{depts_block}

Output rules (VERY IMPORTANT):
- Output ONLY the department name from the list above.
- Do NOT output anything that is not in the list.
- No explanation. No extra words. No numbering. No punctuation after the name.

Below are examples for ALL diseases in the training dataset.
Learn the mapping style and follow it STRICTLY.

=== TRAINING EXAMPLES ===
Value -> General Medicine
(vertigo) Paroxysmal Positional Vertigo -> Neurology
AIDS -> Infectious Disease
Acne -> Dermatology
Alcoholic hepatitis -> Gastroenterology / Hepatology
Allergy -> General Medicine
Arthritis -> Orthopaedics
Bronchial Asthma -> Respiratory
Cervical spondylosis -> Orthopaedics
Chicken pox -> Infectious Disease
Chronic cholestasis -> Gastroenterology / Hepatology
Common Cold -> General Medicine
Dengue -> Infectious Disease
Diabetes -> Endocrinology
Dimorphic hemmorhoids(piles) -> Gastroenterology / Hepatology
Disease -> General Medicine
Drug Reaction -> General Medicine
Fungal infection -> Dermatology
GERD -> Gastroenterology / Hepatology
Gastroenteritis -> Gastroenterology / Hepatology
Heart attack -> Cardiology
Hepatitis A -> Gastroenterology / Hepatology
Hepatitis B -> Gastroenterology / Hepatology
Hepatitis C -> Gastroenterology / Hepatology
Hepatitis D -> Gastroenterology / Hepatology
Hepatitis E -> Gastroenterology / Hepatology
Hypertension -> Cardiology
Hyperthyroidism -> Endocrinology
Hypoglycemia -> Endocrinology
Hypothyroidism -> Endocrinology
Impetigo -> Dermatology
Jaundice -> Gastroenterology / Hepatology
Malaria -> Infectious Disease
Migraine -> Neurology
Osteoarthritis -> Orthopaedics
Paralysis (brain hemorrhage) -> Neurology
Peptic ulcer diseae -> Gastroenterology / Hepatology
Pneumonia -> Respiratory
Psoriasis -> Dermatology
Tuberculosis -> Infectious Disease
Typhoid -> Infectious Disease
Urinary tract infection -> Urology
Varicose veins -> General Medicine
hepatitis A -> Gastroenterology / Hepatology
=== END OF TRAINING EXAMPLES ===
""".strip()
    # ---------- internal helpers ----------

    @staticmethod
    def _pretty(sym: str) -> str:
        # convert "chest_pain" -> "chest pain"
        return sym.replace("_", " ")

    def _build_prompt(self, disease: str, symptoms: List[str]) -> str:
        symptoms_txt = (
            ", ".join(self._pretty(s) for s in symptoms)
            if symptoms else "no other symptoms recorded"
        )

        return (
                self.base_prompt_header
                + f"""

Now classify the new case.

Disease: {disease or "unknown disease"}
Additional symptoms: {symptoms_txt}

Answer with ONLY the department name from the list above.
Department:
""".rstrip()
        )

    def _normalise_department(self, raw: str) -> str:
        """
        Map the raw LLM text back to one of self.departments.
        Very forgiving: checks substrings and falls back to General Medicine.
        """
        if not raw:
            return "General Medicine"

        # Just look at the first line after "Department:"
        line = raw.strip().splitlines()[0]
        line = line.strip().strip('"').strip("“”").lower()

        # Exact / substring matches
        for dep in self.departments:
            dlow = dep.lower()
            if dlow in line or line in dlow:
                return dep

        # Simple synonyms
        synonyms = {
            "gp": "General Medicine",
            "general practice": "General Medicine",
            "general practitioner": "General Medicine",
            "id": "Infectious Disease",
            "infectious diseases": "Infectious Disease",
            "gastroenterology": "Gastroenterology / Hepatology",
            "hepatology": "Gastroenterology / Hepatology",
            "psychology": "Psychiatry",
            "mental health": "Psychiatry",
        }
        for key, dep in synonyms.items():
            if key in line:
                return dep

        # Last resort
        return "General Medicine"

    # ---------- lazy model load ----------

    def _ensure_model_loaded(self) -> bool:
        """
        Lazily load the HF model in a memory-aware way.
        Returns True if the model is ready, False otherwise.
        """
        if not self.use_llm or not _HAS_TRANSFORMERS:
            return False

        if self._model_loaded_ok:
            return True  # already loaded successfully

        try:
            # Tokenizer is usually light
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Memory-friendly load for Streamlit Cloud CPU
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                low_cpu_mem_usage=True,      # avoid huge peak RAM for st cloud
                device_map="cpu",
            )
            self.model.eval()
            self._model_loaded_ok = True
            return True

        except Exception:
            # Any error (OOM, HF issues, etc.): disable LLM usage
            self.model = None
            self.tokenizer = None
            self._model_loaded_ok = False
            self.use_llm = False
            return False

    # ---------- LLM call ----------

    def _call_llm(self, prompt: str) -> str:
        """
        Run generation once the model is guaranteed to be loaded.
        On any failure, return empty string to trigger fallback.
        """
        if not (self._model_loaded_ok and self.model and self.tokenizer):
            return ""

        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )

            # Keep on CPU for Streamlit Cloud
            inputs = {k: v.to("cpu") for k, v in inputs.items()}

            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=16,
                    do_sample=False,                  # greedy, deterministic
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            full = self.tokenizer.decode(out[0], skip_special_tokens=True)
            return full

        except Exception:
            return ""

    # ---------- public API ----------

    def route_department(self, disease: str, symptoms: List[str]) -> str:
        """
        Main entry: return a department name for the given disease.

        - If transformers or model loading fails, falls back to 'General Medicine'.
        - Model is only loaded at the first call where disease is non-empty.
        """
        if not disease:
            return "No Disease"

        # Try to load model lazily
        if not self._ensure_model_loaded():
            return "Model Not Loaded"

        prompt = self._build_prompt(disease, symptoms)
        full_text = self._call_llm(prompt)

        if not full_text:
            return "Return Error"

        # Extract everything AFTER the last "Department:" in the combined text
        if "Department:" in full_text:
            answer_part = full_text.split("Department:")[-1]
        else:
            answer_part = full_text

        return self._normalise_department(answer_part)