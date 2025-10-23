# Toxic Comment Detection Project

This repository trains and compares six classical machine learning models for
toxic comment detection. The pipeline covers preprocessing, feature
extraction, baseline modelling, systematic hyperparameter tuning, evaluation,
and artifact tracking so that each experiment can be reproduced end to end.

## Dataset

* **Source** – `train.csv` (16,000 rows) containing raw text comments labelled
  as toxic (`1`) or non-toxic (`0`). The dataset is perfectly balanced with
  8,000 toxic and 8,000 non-toxic examples, which allows accuracy-based metrics
  to remain informative.
* **Target column** – `toxic` (binary).
* **Feature preparation** – Text is lowercased, stripped of punctuation,
  tokenised, stop-words removed, and lemmatised before being vectorised with a
  TF–IDF representation capped at 5,000 features (`src/config.py`).

## Model Catalogue and Suitability

* **Logistic Regression** (`sklearn.linear_model.LogisticRegression`) – Strong
  linear baseline for high-dimensional sparse TF–IDF vectors; fast to train and
  interpretable via feature weights.
* **Multinomial Naive Bayes** (`sklearn.naive_bayes.MultinomialNB`) –
  Probabilistic model that excels on word-frequency features and provides
  calibrated likelihoods for threshold analysis.
* **Linear SVM (LinearSVC)** (`sklearn.svm.LinearSVC`) – Handles very high
  dimensional spaces with margin maximisation, typically delivering
  state-of-the-art results on sparse text.
* **Decision Tree** (`sklearn.tree.DecisionTreeClassifier`) – Offers
  interpretable rules and highlights feature interactions, useful for
  understanding toxic language triggers.
* **Random Forest** (`sklearn.ensemble.RandomForestClassifier`) – Reduces
  variance by averaging many trees, increasing robustness to noisy
  vocabularies.
* **k-Nearest Neighbours** (`sklearn.neighbors.KNeighborsClassifier`) –
  Non-parametric baseline that leverages local similarity in TF–IDF space,
  providing contrast to the parametric approaches.

All models operate on the shared TF–IDF feature space, making comparative
evaluation directly attributable to the learning algorithm.

## Implementation Details

* **Preprocessing scripts** – Located in `src/preprocessing/` (see
  `src/config.py` for the staged artifact paths). They generate the final
  `artifacts/final_preprocessed.csv`, sparse TF–IDF matrix (`X_tfidf.npz`), and
  label file (`y.csv`).
* **Model training scripts** – `src/models/train_*.py` files instantiate a
  baseline model (`make_model`) and rely on shared utilities in
  `src/models/common.py`.
* **Key utilities** –
  * `load_features` loads the persisted TF–IDF matrix.
  * `split_data` performs an 80/20 stratified split using
    `sklearn.model_selection.train_test_split`.
  * `baseline_eval` and `tune_and_eval` run training, evaluation, plotting, and
    CSV logging (`results/<Model>_metrics.csv`).
* **Libraries** – scikit-learn (models, metrics, `GridSearchCV`), pandas &
  NumPy (data handling), SciPy (sparse persistence), matplotlib (confusion
  matrix and ROC plots), joblib (model persistence).

### Hyperparameters and Parameter Tuning

Each `train_*.py` script defines a `param_grid` explored via
`GridSearchCV(cv=3, n_jobs=-1)` inside `tune_and_eval`:

| Model | Key Hyperparameters |
|-------|---------------------|
| Logistic Regression | `C`, `solver`, `penalty` (L2) |
| Multinomial Naive Bayes | `alpha`, `fit_prior` |
| LinearSVC | `C`, `loss`, `class_weight` |
| Decision Tree | `criterion`, `max_depth`, `min_samples_split`, `min_samples_leaf` |
| Random Forest | `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf` |
| KNN | `n_neighbors`, `weights`, `metric` |

For each model we evaluate **baseline** (default constructor) and **tuned**
variants, resulting in 12 trained estimators per full experiment. Grid-search
scores are averaged across folds, and the best estimator is re-evaluated on the
held-out test set.

## Evaluation Protocol

* **Validation** – Stratified 80/20 train/test split (`split_data`) coupled with
  3-fold cross-validation inside the grid-search. This balances computational
  cost with reliable generalisation estimates.
* **Metrics captured** (`src/utils/eval_utils.py`):
  * Accuracy – overall correctness, meaningful due to balanced classes.
  * Precision – proportion of predicted toxic comments that are actually toxic;
    crucial to limit false positives in moderation workflows.
  * Recall – proportion of toxic comments correctly captured; guards against
    harmful content escaping detection.
  * F1 score – harmonic mean of precision and recall; useful when both error
    types matter.
  * ROC-AUC – threshold-independent ranking ability, especially valuable when
    future operating points may shift.
  * Confusion Matrix – visual aid for per-class error analysis.
* **Artifacts** – Metrics are stored in CSV files inside `results/`, confusion
  matrices and ROC curves are exported as PNG files, and trained models are
  persisted under `artifacts/models/` for downstream inference.

## Model Comparison and Insights

* **Linear baselines excel** – Logistic Regression and LinearSVC typically
  achieve the highest F1 and ROC-AUC scores after tuning thanks to their
  suitability for sparse TF–IDF features.
* **Probabilistic vs. margin-based** – Multinomial Naive Bayes provides fast
  inference and calibrated probabilities, but the margin maximisation of
  LinearSVC yields superior recall on nuanced toxic phrases.
* **Tree ensembles improve robustness** – Random Forest outperforms single
  Decision Trees by reducing variance, making it a dependable alternative when
  interpretability is less critical.
* **KNN serves as a sanity check** – While generally trailing in F1/ROC-AUC, it
  verifies that gains from other models stem from learned decision boundaries
  rather than artefacts in preprocessing.

**Conclusion:** For balanced toxic comment detection on TF–IDF features, Linear
SVM and Logistic Regression provide the best trade-off between accuracy,
recall, and computational cost. Random Forest is a strong ensemble backup when
non-linear relationships become important, while Naive Bayes offers a lightweight
option when resources are constrained.
