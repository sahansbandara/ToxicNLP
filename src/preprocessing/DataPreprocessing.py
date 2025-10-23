import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from joblib import dump
from src.config import (
    RANDOM_STATE, TRAIN_CSV, STEP1_LOWER, STEP2_NOPUNCT, STEP3_TOKENIZED,
    STEP4_NOSTOP, STEP5_LEMM, FINAL_PREPROCESSED, X_TFIDF_NPZ, Y_CSV,
    VECTORIZER_PKL, MAX_FEATURES, MIN_DF, ARTIFACTS_DIR
)
from src.utils.io_paths import ensure_dirs
from src.utils.text_cleaning import (
    init_nltk, lowercase, remove_punct, tokenize,
    get_stopwords_set, remove_stopwords, get_lemmatizer,
    lemmatize, join_tokens, build_vectorizer
)

def main():
    parser = argparse.ArgumentParser(description="Data preprocessing pipeline for Toxic Comment Detection")
    parser.add_argument("--input", type=str, default=TRAIN_CSV, help="Path to train.csv with columns: comment,toxic")
    parser.add_argument("--no-plots", action="store_true", help="Disable saving EDA plots")
    args = parser.parse_args()

    ensure_dirs()
    init_nltk()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input CSV not found: {args.input}")
    df = pd.read_csv(args.input)

    df.columns = [c.lower() for c in df.columns]
    if not {"comment","toxic"}.issubset(df.columns):
        raise ValueError("CSV must contain columns: comment,toxic")
    df = df[["comment","toxic"]].copy()
    df["toxic"] = df["toxic"].astype(int)

    # Optional balance to 8000/8000
    pos = df[df["toxic"]==1]
    neg = df[df["toxic"]==0]
    if len(pos) >= 8000 and len(neg) >= 8000:
        pos = pos.sample(8000, random_state=RANDOM_STATE)
        neg = neg.sample(8000, random_state=RANDOM_STATE)
        df = pd.concat([pos,neg], ignore_index=True).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    # Step 1
    df1 = df.copy()
    df1["comment"] = df1["comment"].astype(str).apply(lowercase)
    df1.to_csv(STEP1_LOWER, index=False)

    # Step 2
    df2 = df1.copy()
    df2["comment"] = df2["comment"].apply(remove_punct)
    df2.to_csv(STEP2_NOPUNCT, index=False)

    # Step 3
    df3 = df2.copy()
    df3["tokens"] = df3["comment"].apply(tokenize)
    df3.to_csv(STEP3_TOKENIZED, index=False)

    # Step 4
    stop_set = get_stopwords_set()
    df4 = df3.copy()
    df4["tokens"] = df4["tokens"].apply(lambda t: [w for w in t if w])
    df4["tokens"] = df4["tokens"].apply(lambda t: [w for w in t if w not in stop_set])
    df4.to_csv(STEP4_NOSTOP, index=False)

    # Step 5 (lemmatization + join)
    lemm = get_lemmatizer()
    df5 = df4.copy()
    df5["tokens"] = df5["tokens"].apply(lambda t: [lemm.lemmatize(w) for w in t])
    df5["comment"] = df5["tokens"].apply(join_tokens)
    df5 = df5[["comment","toxic"]]
    df5.to_csv(STEP5_LEMM, index=False)

    # Vectorization
    vect = build_vectorizer(max_features=MAX_FEATURES, min_df=MIN_DF)
    X = vect.fit_transform(df5["comment"].astype(str))
    y = df5["toxic"].values

    # Dense CSV
    dense = pd.DataFrame(X.toarray(), columns=vect.get_feature_names_out())
    dense["toxic"] = y
    dense.to_csv(FINAL_PREPROCESSED, index=False)

    # Sparse + y + vectorizer
    sparse.save_npz(X_TFIDF_NPZ, X)
    pd.DataFrame({"toxic": y}).to_csv(Y_CSV, index=False)
    dump(vect, VECTORIZER_PKL)

    if not args.no_plots:
        # Word count
        wc = df5["comment"].apply(lambda s: len(str(s).split()))
        plt.figure()
        plt.hist(wc, bins=50)
        plt.title("Word Count Distribution (after lemmatization)")
        plt.xlabel("Words per comment")
        plt.ylabel("Frequency")
        plt.savefig(f"{ARTIFACTS_DIR}/wordcount_hist.png", bbox_inches="tight")
        plt.close()

        # Top 15 TF-IDF
        mean_tfidf = (X.mean(axis=0)).A1
        idx = mean_tfidf.argsort()[-15:][::-1]
        terms = vect.get_feature_names_out()
        top_terms = [terms[i] for i in idx]
        top_vals = [float(mean_tfidf[i]) for i in idx]

        plt.figure()
        plt.barh(list(reversed(top_terms)), list(reversed(top_vals)))
        plt.title("Top 15 Terms by Average TF-IDF")
        plt.xlabel("Average TF-IDF")
        plt.ylabel("Term")
        plt.tight_layout()
        plt.savefig(f"{ARTIFACTS_DIR}/top_tfidf.png", bbox_inches="tight")
        plt.close()

    print("Preprocessing complete. Artifacts saved in 'artifacts/'.")

if __name__ == "__main__":
    main()
