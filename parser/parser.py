from g4f.client import Client
import time
import re
import json
import hashlib
from typing import List, Dict
from tqdm import tqdm


client = Client()

def answer(scene_text, model="gpt-oss-120b"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": f"""
Ты система аннотации художественного текста.

Тебе дан фрагмент:

{scene_text}

Нужно вернуть результат строго в формате JSON:

{{
  "extra_characters": [список уникальных имён и титулов, без общих слов вроде "рыцарь", "женщина", "человек"],
  "extra_locations": [список мест в виде иерархии "Регион > Город > Локация", если вложенность неизвестна — оставь одно имя],
  "extra_events": [список коротких описаний событий в естественном языке],
  "event_tags": [список тех же событий, но в виде коротких slug-ярлыков на английском: "arrival_rivia", "lifting_curse"]
}}

⚠️ Никаких комментариев, объяснений или дополнительного текста.
⚠️ JSON должен быть валидным.
"""
            }
        ],
        web_search=False
    )
    answer = response.choices[0].message.content

    try:
        return json.loads(answer)
    except Exception:
        lines = answer.splitlines()
        chars, locs, evs, tags = [], [], [], []
        for line in lines:
            if line.strip().startswith('"extra_characters"'):
                chars = re.findall(r'"([^"]+)"', line)
            elif line.strip().startswith('"extra_locations"'):
                locs = re.findall(r'"([^"]+)"', line)
            elif line.strip().startswith('"extra_events"'):
                evs = re.findall(r'"([^"]+)"', line)
            elif line.strip().startswith('"event_tags"'):
                tags = re.findall(r'"([^"]+)"', line)
        return {
            "extra_characters": chars,
            "extra_locations": locs,
            "extra_events": evs,
            "event_tags": tags
        }


def save_checkpoint(results, checkpoint_path):
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)

def load_checkpoint(checkpoint_path):
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["chapter_index"], data["scene_index"], data["results"]
    else:
        return 0, 0, []

def rough_token_count(text: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", text))

def summarize_short(text: str, limit_words=90) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    out = []
    for s in sentences:
        if len((" ".join(out) + " " + s).split()) > limit_words:
            break
        out.append(s)
    return " ".join(out) or text[:400]

def make_beats(text: str, max_items=5) -> List[str]:
    keys = ["ночь", "бой", "рассвет", "принцесса", "король",
            "монстр", "замок", "погоня", "диалог", "драка"]
    t = text.lower()
    return [k for k in keys if k in t][:max_items]

def hash_quotes(text: str, max_items=3) -> List[str]:
    quotes = re.findall(r"«([^»]{4,100})»", text)
    return [hashlib.md5(q.strip().encode("utf-8")).hexdigest() for q in quotes[:max_items]]


def split_into_chapters(text: str) -> List[str]:
    return [p.strip() for p in re.split(r"\n\s*ГЛАВА [^\n]+\n", text, flags=re.IGNORECASE) if p.strip()]

def split_into_scenes(chapter_text: str) -> List[str]:
    raw = re.split(r"(\*{3,}|—\s*—\s*—|\n\s*\n\s*\n)", chapter_text)
    scenes, buf = [], []
    for chunk in raw:
        if chunk.strip() in ["***", "— — —"] or re.match(r"\s*\n\s*\n\s*\n", chunk):
            if buf:
                scenes.append("\n".join(buf).strip()); buf = []
        else:
            if chunk.strip(): buf.append(chunk)
    if buf: scenes.append("\n".join(buf).strip())
    return [s for s in scenes if s.strip()]

def parse_book(txt_path, book_id="witcher_01", out_path="scenes.jsonl", checkpoint_path="checkpoint.json", load_checkpoint=False):
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    chapters = split_into_chapters(text)

    if load_checkpoint:
        ch_start_idx, sc_start_idx, results = load_checkpoint(checkpoint_path)
        print(f"Загружен прогресс: глава {ch_start_idx+1}, сцена {sc_start_idx+1}")
    else:
        ch_start_idx, sc_start_idx, results = 0, 0, []
        save_checkpoint({"chapter_index": 0, "scene_index": 0, "results": []}, checkpoint_path)

    for ch_id in range(ch_start_idx, len(chapters)):
        ch_text = chapters[ch_id]
        scenes = split_into_scenes(ch_text)

        start_scene_idx = sc_start_idx if ch_id == ch_start_idx else 0

        for sc_id in tqdm(range(start_scene_idx, len(scenes)), desc=f"Глава {ch_id+1}", leave=False):
            sc_text = scenes[sc_id]
            start_char = sum(len(s) for s in scenes[:sc_id]) + sum(len(c) for c in chapters[:ch_id])
            end_char = start_char + len(sc_text)

            try:
                time.sleep(2)
                ents = answer(sc_text)
                meta = {
                    "book_id": book_id,
                    "chapter_id": ch_id + 1,
                    "scene_id": f"{ch_id+1:02d}_{sc_id+1:03d}",
                    "text": sc_text.strip(),
                    "token_len": rough_token_count(sc_text),
                    "extra_characters": ents.get("extra_characters", []),
                    "extra_locations": ents.get("extra_locations", []),
                    "extra_events": ents.get("extra_events", []),
                    "event_tags": ents.get("event_tags", []),
                    "summary_50w": summarize_short(sc_text, limit_words=90),
                    "beats": make_beats(sc_text, max_items=6),
                    "quote_hashes": hash_quotes(sc_text),
                    "start_char": start_char,
                    "end_char": end_char
                }
                results.append(meta)

                save_checkpoint({"chapter_index": ch_id, "scene_index": sc_id + 1, "results": results}, checkpoint_path)

            except Exception as e:
                print(f"Ошибка при обработке сцены {ch_id+1}-{sc_id+1}: {e}")
                print("Пауза на 60 секунд перед продолжением...")
                time.sleep(60)
                save_checkpoint({"chapter_index": ch_id, "scene_index": sc_id, "results": results}, checkpoint_path)
                continue

        sc_start_idx = 0

    with open(out_path, "w", encoding="utf-8") as out:
        for r in results:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"✅ Parsed {len(results)} scenes → {out_path}")