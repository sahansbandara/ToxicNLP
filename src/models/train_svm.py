import os, sys, argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from sklearn.svm import LinearSVC
from src.config import RESULTS_DIR
from src.models.common import (
    load_features, split_data, baseline_eval, tune_and_eval, save_baseline_model
)

MODEL_NAME = "SVM (LinearSVC)"
SHORT_NAME = "svm_linear"

def make_model():
    return LinearSVC()

param_grid = {
    "C": [0.1, 1.0, 3.0, 10.0]
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    X, y = load_features(force_dense=args.dense)
    X_train, X_test, y_train, y_test = split_data(X, y, args.test_size)

    model = make_model()
    results_csv = os.path.join(RESULTS_DIR, f"{MODEL_NAME.replace(' ', '_')}_metrics.csv")
    baseline_eval(model, X_train, X_test, y_train, y_test, MODEL_NAME, SHORT_NAME, results_csv, no_plots=args.no_plots)
    save_baseline_model(model, SHORT_NAME)
    tune_and_eval(make_model, param_grid, X_train, X_test, y_train, y_test, MODEL_NAME, SHORT_NAME, results_csv, no_plots=args.no_plots)

if __name__ == "__main__":
    main()
