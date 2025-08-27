import json
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# === Загружаем сцены ===
def load_scenes(path="data/scenes.jsonl"):
    scenes = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            scenes.append(json.loads(line))
    return scenes

scenes = load_scenes("data/scenes.jsonl")
print(f"✅ Загружено {len(scenes)} сцен")

# === BM25 индекс ===
bm25_corpus = [s["summary_50w"] + " " + " ".join(s["beats"]) + " " + " ".join(s["event_tags"]) for s in scenes]
bm25_tokens = [doc.split() for doc in bm25_corpus]
bm25 = BM25Okapi(bm25_tokens)

# === FAISS индекс (эмбеддинги summary_50w) ===
model = SentenceTransformer("intfloat/multilingual-e5-small")  # компактная мультиязычная модель
embeddings = model.encode([s["summary_50w"] for s in scenes], normalize_embeddings=True)

d = embeddings.shape[1]
index = faiss.IndexFlatIP(d)  # косинус (dot product после нормализации)
index.add(embeddings)

print(f"✅ BM25 и FAISS индексы построены")

# === Функция поиска ===
def search(query, must_have_characters=None, topk_bm25=30, topk_faiss=30):
    # BM25
    query_tokens = query.split()
    bm25_scores = bm25.get_scores(query_tokens)
    bm25_top = np.argsort(bm25_scores)[::-1][:topk_bm25]

    # FAISS
    q_emb = model.encode([query], normalize_embeddings=True)
    faiss_scores, faiss_idx = index.search(q_emb, topk_faiss)

    candidates = set(bm25_top.tolist() + faiss_idx[0].tolist())
    results = [scenes[i] for i in candidates]

    if must_have_characters:
        results = [
            s for s in results
            if any(char in s["extra_characters"] for char in must_have_characters)
        ]
    return results

__all__ = ["bm25", "index", "model", "scenes"]