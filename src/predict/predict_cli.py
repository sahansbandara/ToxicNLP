import os, sys, argparse, joblib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.text_cleaning import (
    init_nltk, get_stopwords_set, get_lemmatizer, clean_for_inference
)
from src.config import VECTORIZER_PKL

def predict_one(model, vectorizer, text, stop_set, lemm):
    clean = clean_for_inference(text, stop_set, lemm)
    X = vectorizer.transform([clean])
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0,1]
        pred = int(proba >= 0.5)
        return pred, float(proba)
    elif hasattr(model, "decision_function"):
        score = model.decision_function(X)[0]
        pred = int(score >= 0.0)
        return pred, float(score)
    else:
        pred = int(model.predict(X)[0])
        return pred, float('nan')

def main():
    parser = argparse.ArgumentParser(description="Predict toxicity for user input")
    parser.add_argument("--model-path", required=True, help="Path to joblib model file")
    parser.add_argument("--vectorizer-path", default=VECTORIZER_PKL, help="Path to saved TfidfVectorizer pickle")
    parser.add_argument("--interactive", action="store_true", help="Interactive loop mode")
    parser.add_argument("--text", type=str, help="Single text to classify")
    args = parser.parse_args()

    init_nltk()
    stop_set = get_stopwords_set()
    lemm = get_lemmatizer()

    model = joblib.load(args.model_path)
    vectorizer = joblib.load(args.vectorizer_path)

    if args.interactive:
        print("Enter text (type 'exit' to quit):")
        while True:
            try:
                t = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if t.lower() == "exit":
                break
            pred, score = predict_one(model, vectorizer, t, stop_set, lemm)
            print(f"Pred: {pred} | Score: {score}")
    elif args.text is not None:
        pred, score = predict_one(model, vectorizer, args.text, stop_set, lemm)
        print(f"Pred: {pred} | Score: {score}")
    else:
        print("Provide --interactive or --text")

if __name__ == "__main__":
    main()
