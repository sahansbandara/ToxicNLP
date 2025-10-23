import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pandas as pd
from scipy import sparse
from joblib import dump
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, roc_curve
from src.config import (
    RANDOM_STATE, X_TFIDF_NPZ, Y_CSV, FINAL_PREPROCESSED,
    ARTIFACTS_DIR, RESULTS_DIR
)
from src.utils.io_paths import ensure_dirs
from src.utils.eval_utils import (
    compute_classification_metrics, save_metrics_row,
    plot_confusion_matrix, plot_roc_curve, get_scores_for_roc
)

def load_features(force_dense=False):
    if not force_dense and os.path.exists(X_TFIDF_NPZ) and os.path.exists(Y_CSV):
        X = sparse.load_npz(X_TFIDF_NPZ)
        y = pd.read_csv(Y_CSV)["toxic"].values
        return X, y
    if os.path.exists(FINAL_PREPROCESSED):
        df = pd.read_csv(FINAL_PREPROCESSED)
        y = df["toxic"].values
        X = df.drop(columns=["toxic"]).values
        return X, y
    raise FileNotFoundError("Features not found. Run preprocessing first.")

def split_data(X, y, test_size):
    return train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

def baseline_eval(model, X_train, X_test, y_train, y_test, model_name, short_name, results_csv, no_plots=False):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_score = get_scores_for_roc(model, X_test)
    metrics = compute_classification_metrics(y_test, y_pred, y_score)
    print("Baseline:", metrics)
    save_metrics_row(model_name, "baseline", metrics, results_csv)

    cm = confusion_matrix(y_test, y_pred)
    if not no_plots:
        plot_confusion_matrix(cm, f"{model_name} – Confusion Matrix (Baseline)",
                              out_path=f"{RESULTS_DIR}/{short_name}_cm_baseline.png", show=True)
        if y_score is not None:
            fpr, tpr, _ = roc_curve(y_test, y_score)
            plot_roc_curve(fpr, tpr, f"{model_name} – ROC (Baseline)",
                           out_path=f"{RESULTS_DIR}/{short_name}_roc_baseline.png", show=True)

def tune_and_eval(model_factory, param_grid, X_train, X_test, y_train, y_test,
                  model_name, short_name, results_csv, no_plots=False):
    gs = GridSearchCV(model_factory(), param_grid, cv=3, n_jobs=-1)
    gs.fit(X_train, y_train)
    best = gs.best_estimator_
    print("Best params:", gs.best_params_)

    y_pred = best.predict(X_test)
    y_score = get_scores_for_roc(best, X_test)
    metrics = compute_classification_metrics(y_test, y_pred, y_score)
    print("Tuned:", metrics)
    save_metrics_row(model_name, "tuned", metrics, results_csv)

    cm = confusion_matrix(y_test, y_pred)
    if not no_plots:
        plot_confusion_matrix(cm, f"{model_name} – Confusion Matrix (Tuned)",
                              out_path=f"{RESULTS_DIR}/{short_name}_cm_tuned.png", show=True)
        if y_score is not None:
            fpr, tpr, _ = roc_curve(y_test, y_score)
            plot_roc_curve(fpr, tpr, f"{model_name} – ROC (Tuned)",
                           out_path=f"{RESULTS_DIR}/{short_name}_roc_tuned.png", show=True)

    # Save tuned model
    os.makedirs(os.path.join(ARTIFACTS_DIR, "models"), exist_ok=True)
    dump(best, os.path.join(ARTIFACTS_DIR, "models", f"{short_name}_tuned.joblib"))
    return best

def save_baseline_model(model, short_name):
    os.makedirs(os.path.join(ARTIFACTS_DIR, "models"), exist_ok=True)
    dump(model, os.path.join(ARTIFACTS_DIR, "models", f"{short_name}_baseline.joblib"))
