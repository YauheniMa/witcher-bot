from razdel import tokenize as razdel_tokenize
from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger, NewsNERTagger, Doc
from typing import List, Dict
import numpy as np

segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
ner_tagger = NewsNERTagger(emb)

EVENT_SYNONYMS = {
    "снятие проклятия": "lifting_curse",
    "бой со стрыгой": "fight_striga",
    "проклятие стрыги": "striga_curse",
    "свадьба": "wedding_celebration",
    "дорога": "journey",
    "прибытие": "arrival",
    "битва": "battle",
    "разговор": "dialogue",
    "казнь": "execution",
    "пир": "feast"
}

def extract_events_from_query(query: str) -> List[str]:
    events = []
    for phrase, tag in EVENT_SYNONYMS.items():
        if phrase in query.lower():
            events.append(tag)
    return events

def extract_entities(text: str) -> Dict[str, List[str]]:
    try:
        doc = Doc(text)
        doc.segment(segmenter)
        doc.tag_morph(morph_tagger)
        doc.tag_ner(ner_tagger)

        characters, locations, misc = [], [], []

        for span in doc.spans:
            span.normalize(morph_vocab)
            if span.type == "PER":  # персонажи
                characters.append(span.normal)
            elif span.type == "LOC":  # локации
                locations.append(span.normal)
            else:
                misc.append(span.normal)

        return {
            "characters": list(set(characters)),
            "locations": list(set(locations)),
            "misc": list(set(misc))
        }
    except Exception:
        return {"characters": [], "locations": [], "misc": []}

# === Расширенный поиск ===
def smart_search(query, bm25, index, model, scenes, topk_bm25=30, topk_faiss=30):
    ents = extract_entities(query)
    events = extract_events_from_query(query)

    # BM25
    query_tokens = query.split()
    bm25_scores = bm25.get_scores(query_tokens)
    bm25_top = np.argsort(bm25_scores)[::-1][:topk_bm25]

    # FAISS
    q_emb = model.encode([query], normalize_embeddings=True)
    faiss_scores, faiss_idx = index.search(q_emb, topk_faiss)

    # Кандидаты
    candidates = set(bm25_top.tolist() + faiss_idx[0].tolist())
    results = [scenes[i] for i in candidates]

    scored = []
    for s in results:
        score = 0
        # Персонажи
        score += sum(1 for c in ents["characters"] if c in s["extra_characters"])
        # Локации
        score += sum(1 for l in ents["locations"] if l in s["extra_locations"])
        # События
        score += 2 * sum(1 for e in events if e in s["event_tags"])
        scored.append((score, s))

    scored = sorted(scored, key=lambda x: x[0], reverse=True)

    return [s for _, s in scored], {"characters": ents["characters"], "locations": ents["locations"], "events": events}