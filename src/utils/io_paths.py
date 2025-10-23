import os
from ..config import (
    ARTIFACTS_DIR, RESULTS_DIR, STEP1_LOWER, STEP2_NOPUNCT, STEP3_TOKENIZED,
    STEP4_NOSTOP, STEP5_LEMM, FINAL_PREPROCESSED, X_TFIDF_NPZ, Y_CSV,
    VECTORIZER_PKL, MODELS_DIR
)

def ensure_dirs():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

def paths_dict():
    return {
        "step1": STEP1_LOWER,
        "step2": STEP2_NOPUNCT,
        "step3": STEP3_TOKENIZED,
        "step4": STEP4_NOSTOP,
        "step5": STEP5_LEMM,
        "final": FINAL_PREPROCESSED,
        "X_npz": X_TFIDF_NPZ,
        "y_csv": Y_CSV,
        "vectorizer": VECTORIZER_PKL,
        "models_dir": MODELS_DIR,
    }
