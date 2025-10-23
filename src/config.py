# ============================================
# CONFIGURATION FILE – Toxic Comment Detection
# ============================================

# ------------------------------
# Global constants
# ------------------------------
RANDOM_STATE = 42

# ------------------------------
# Dataset
# ------------------------------
TRAIN_CSV = "train.csv"  # Input dataset path

# ------------------------------
# Directory structure
# ------------------------------
ARTIFACTS_DIR = "artifacts"
RESULTS_DIR = "results"
MODELS_DIR = f"{ARTIFACTS_DIR}/models"

# ------------------------------
# Preprocessing step files
# ------------------------------
STEP1_LOWER = f"{ARTIFACTS_DIR}/step1_lowercase.csv"
STEP2_NOPUNCT = f"{ARTIFACTS_DIR}/step2_no_punct.csv"
STEP3_TOKENIZED = f"{ARTIFACTS_DIR}/step3_tokenized.csv"
STEP4_NOSTOP = f"{ARTIFACTS_DIR}/step4_no_stopwords.csv"
STEP5_LEMM = f"{ARTIFACTS_DIR}/step5_lemmatized.csv"

FINAL_PREPROCESSED = f"{ARTIFACTS_DIR}/final_preprocessed.csv"
X_TFIDF_NPZ = f"{ARTIFACTS_DIR}/X_tfidf.npz"
Y_CSV = f"{ARTIFACTS_DIR}/y.csv"
VECTORIZER_PKL = f"{ARTIFACTS_DIR}/vectorizer.pkl"

# ------------------------------
# TF-IDF configuration
# ------------------------------
MAX_FEATURES = 5000
MIN_DF = 2

# ------------------------------
# Notes
# ------------------------------
# - These constants define all file paths and parameters used across preprocessing and training scripts.
# - Do not perform any file I/O here.
# - All scripts import from this file to ensure consistent naming.
