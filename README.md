# Toxic Comment Detection – Notebook Workflow

This repository now ships as a single Jupyter notebook so that the entire toxic comment detection pipeline can be executed inside Jupyter Notebook or JupyterLab without relying on standalone Python scripts.

## Contents

| File | Description |
| --- | --- |
| `ToxicNLP.ipynb` | End-to-end notebook covering preprocessing, TF–IDF feature extraction, six classical models, hyper-parameter tuning, evaluation visualisations, and an interactive inference helper. |
| `train.csv` | Balanced dataset of 16,000 labelled comments (8k toxic / 8k non-toxic). |

Running the notebook from top to bottom will recreate all artefacts (cleaned text CSV, TF–IDF matrix, metrics table, and comparison plots) inside the local `artifacts/` and `results/` folders.

## Getting Started

1. Create and activate a Python environment with scikit-learn, pandas, NumPy, SciPy, matplotlib, seaborn, and NLTK installed.
2. Launch Jupyter: `jupyter notebook` or `jupyter lab`.
3. Open `ToxicNLP.ipynb` and execute each cell sequentially.
4. Optional: adjust the parameter grids in section 8 to experiment with different hyper-parameters.

The final section of the notebook exposes a `predict_toxicity` helper that lets you test arbitrary strings against the best tuned model discovered during the run.
