import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

def init_nltk():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

def lowercase(text: str) -> str:
    return str(text).lower()

def remove_punct(text: str) -> str:
    # keep only a-z and space
    return re.sub(r'[^a-z\s]', '', str(text))

def tokenize(text: str):
    return str(text).split()

def get_stopwords_set():
    return set(stopwords.words('english'))

def remove_stopwords(tokens, stop_set):
    return [w for w in tokens if w not in stop_set]

def get_lemmatizer():
    return WordNetLemmatizer()

def lemmatize(tokens, lemm):
    return [lemm.lemmatize(w) for w in tokens]

def join_tokens(tokens):
    return " ".join(tokens)

def build_vectorizer(max_features=5000, min_df=2):
    return TfidfVectorizer(max_features=max_features, min_df=min_df)

def clean_for_inference(text, stop_set, lemm):
    t = lowercase(text)
    t = remove_punct(t)
    toks = tokenize(t)
    toks = remove_stopwords(toks, stop_set)
    toks = lemmatize(toks, lemm)
    return join_tokens(toks)
