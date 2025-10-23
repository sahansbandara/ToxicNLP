import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)

def compute_classification_metrics(y_true, y_pred, y_score):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = np.nan
    if y_score is not None:
        try:
            auc = roc_auc_score(y_true, y_score)
        except Exception:
            auc = np.nan
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc}

def save_metrics_row(model_name, phase, metrics_dict, out_csv_path):
    row = {"model": model_name, "phase": phase, **metrics_dict}
    try:
        prev = pd.read_csv(out_csv_path)
        df = pd.concat([prev, pd.DataFrame([row])], ignore_index=True)
    except Exception:
        df = pd.DataFrame([row])
    df.to_csv(out_csv_path, index=False)

def plot_confusion_matrix(cm, title, out_path=None, show=True):
    plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    if out_path:
        plt.savefig(out_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

def plot_roc_curve(fpr, tpr, title, out_path=None, show=True):
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0,1], [0,1], linestyle='--')
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    if out_path:
        plt.savefig(out_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

def get_scores_for_roc(model, X_test):
    try:
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X_test)[:, 1]
        if hasattr(model, "decision_function"):
            return model.decision_function(X_test)
    except Exception:
        return None
    return None
