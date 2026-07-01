import logging
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from app.catalog import CATALOG

logger = logging.getLogger(__name__)


def build_text(item):
    name = item.get("name", "")
    keys = " ".join(item.get("keys", []))
    job_levels = " ".join(item.get("job_levels", []))
    description = item.get("description", "")
    return name + " " + keys + " " + job_levels + " " + description


logger.info("Loading sentence transformer model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

logger.info("Building search index...")
texts = [build_text(item) for item in CATALOG]
embeddings = model.encode(texts, normalize_embeddings=True)
embeddings = np.array(embeddings).astype("float32")

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

logger.info("Search index ready with " + str(len(CATALOG)) + " items.")


def search(query, top_k=20):
    query_vec = model.encode([query], normalize_embeddings=True)
    query_vec = np.array(query_vec).astype("float32")
    scores, indices = index.search(query_vec, top_k)
    results = []
    for i in indices[0]:
        if i < len(CATALOG):
            results.append(CATALOG[i])
    return results


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