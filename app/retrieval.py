import logging

logger = logging.getLogger(__name__)

_model = None
_index = None
_catalog_list = None


def _ensure_loaded():
    global _model, _index, _catalog_list

    if _index is not None:
        return

    import numpy as np
    import faiss
    from sentence_transformers import SentenceTransformer
    from app.catalog import CATALOG

    logger.info("Loading sentence transformer model...")
    _model = SentenceTransformer("all-MiniLM-L6-v2")

    logger.info("Building search index...")
    texts = []
    for item in CATALOG:
        name = item.get("name", "")
        keys = " ".join(item.get("keys", []))
        job_levels = " ".join(item.get("job_levels", []))
        description = item.get("description", "")
        text = name + " " + keys + " " + job_levels + " " + description
        texts.append(text)

    embeddings = _model.encode(texts, normalize_embeddings=True)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    _index = index
    _catalog_list = list(CATALOG)

    logger.info("Search index ready with " + str(len(_catalog_list)) + " items.")


def search(query, top_k=15):
    import numpy as np

    _ensure_loaded()

    query_vec = _model.encode([query], normalize_embeddings=True)
    query_vec = np.array(query_vec).astype("float32")

    scores, indices = _index.search(query_vec, top_k)

    results = []
    for i in indices[0]:
        if i < len(_catalog_list):
            results.append(_catalog_list[i])

    return results
