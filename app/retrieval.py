from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from app.catalog import CATALOG

_corpus = [
    item["name"] + " " + " ".join(item["keys"]) + " " +
    " ".join(item["job_levels"]) + " " + item["description"]
    for item in CATALOG
]

_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
_tfidf_matrix = _vectorizer.fit_transform(_corpus)


def search(query: str, top_k: int = 20) -> list:
    query_vec = _vectorizer.transform([query])
    scores = cosine_similarity(query_vec, _tfidf_matrix)[0]
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [CATALOG[i] for i in top_indices if scores[i] > 0]