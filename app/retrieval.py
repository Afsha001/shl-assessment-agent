import faiss
from sentence_transformers import SentenceTransformer

from app.catalog import CATALOG


MODEL = SentenceTransformer("all-MiniLM-L6-v2")


def build_text(item):
    name = item.get("name", "")
    keys = " ".join(item.get("keys", []))
    job_levels = " ".join(item.get("job_levels", []))
    description = item.get("description", "")

    return f"{name} {keys} {job_levels} {description}"


def build_search_index(catalog):
    texts = [build_text(item) for item in catalog]

    embeddings = MODEL.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    print(f"Search index built: {index.ntotal} items indexed.")
    return index, catalog


INDEX, CATALOG_LIST = build_search_index(CATALOG)


def search(query, top_k=15):
    query_embedding = MODEL.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)

    scores, positions = INDEX.search(query_embedding, top_k)

    results = []
    for position in positions[0]:
        if position < len(CATALOG_LIST):
            results.append(CATALOG_LIST[position])

    return results


if __name__ == "__main__":
    test_queries = [
        "senior Java developer backend",
        "personality assessment leadership",
        "entry level customer service",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 40)
        top_results = search(query, top_k=3)
        for i, result in enumerate(top_results, start=1):
            print(f"{i}. {result['name']} | {result['test_type']} | {result['url']}")