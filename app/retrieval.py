from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from app.catalog import CATALOG


def build_text(item):
    name = item.get("name", "")
    keys = " ".join(item.get("keys", []))
    job_levels = " ".join(item.get("job_levels", []))
    description = item.get("description", "")
    return name + " " + keys + " " + job_levels + " " + description


_corpus = [build_text(item) for item in CATALOG]

_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
_tfidf_matrix = _vectorizer.fit_transform(_corpus)


def search(query, top_k=20):
    query_vec = _vectorizer.transform([query])
    scores = cosine_similarity(query_vec, _tfidf_matrix)[0]
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [CATALOG[i] for i in top_indices if scores[i] > 0]


if __name__ == "__main__":
    test_queries = [
        "senior Java developer backend",
        "personality assessment leadership",
        "entry level customer service",
    ]

    for query in test_queries:
        print("Query: " + query)
        print("-" * 40)
        results = search(query, top_k=3)
        for item in results:
            print(item["name"] + " | " + item["test_type"] + " | " + item["url"])
        print()