import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
import matplotlib.pyplot as plt
from src.config import RESULTS_DIR

FILES = [
    "Logistic_Regression_metrics.csv",
    "Naive_Bayes_metrics.csv",
    "SVM_(LinearSVC)_metrics.csv",
    "Decision_Tree_metrics.csv",
    "Random_Forest_metrics.csv",
    "KNN_metrics.csv",
]

def load_all():
    frames = []
    for f in FILES:
        path = os.path.join(RESULTS_DIR, f)
        if os.path.exists(path):
            df = pd.read_csv(path)
            frames.append(df)
        else:
            print("Missing:", path)
    if not frames:
        raise FileNotFoundError("No metrics found. Train models first.")
    return pd.concat(frames, ignore_index=True)

def plot_metric(all_df, metric, out_name):
    pivot = all_df.pivot_table(index='model', columns='phase', values=metric, aggfunc='first')
    plt.figure()
    models = list(pivot.index)
    xs = list(range(len(models)))
    baselines = [pivot.loc[m].get('baseline', float('nan')) for m in models]
    tuneds = [pivot.loc[m].get('tuned', float('nan')) for m in models]

    for i, (b, t) in enumerate(zip(baselines, tuneds)):
        plt.bar(i - 0.15, b, width=0.3)
        plt.bar(i + 0.15, t, width=0.3)
    plt.xticks(xs, models, rotation=30, ha='right')
    plt.title(f"{metric.upper()} – Baseline vs Tuned")
    plt.ylabel(metric.upper())
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, out_name), bbox_inches="tight")
    plt.show()
    plt.close()

def main():
    all_df = load_all()
    print(all_df)

    plot_metric(all_df, "accuracy", "accuracy_comparison.png")
    plot_metric(all_df, "f1", "f1_comparison.png")
    plot_metric(all_df, "roc_auc", "rocauc_comparison.png")

    print("""
REPORT NOTES:

- Metrics: Accuracy (overall), Precision/Recall (asymmetric costs), F1 (balance), Confusion Matrix (errors), ROC-AUC (threshold-independent).
- Validation: 80/20 train-test split; 3-fold CV for hyperparameters.
- Typical results: LinearSVC / Logistic Regression strong on TF-IDF; Naive Bayes fast & competitive; Random Forest robust; Decision Tree interpretable; KNN simple but may lag.
- Improvements: Try TF-IDF n-grams (1,2), tuning min_df/max_df, custom stopwords, PR-AUC, probability calibration for SVM, class-wise breakdown, or Transformer baselines later.
""")

if __name__ == "__main__":
    main()
