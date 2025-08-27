from api.search import smart_search
import json
from g4f.client import Client
from api.indexer import bm25, index, model, scenes
from api.search import smart_search
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import threading

client = Client()

# === Загрузка стилей с кэшем ===
_character_cache = None

def load_character_style(name: str, characters_path="data/characters.json"):
    global _character_cache
    if _character_cache is None:
        try:
            with open(characters_path, "r", encoding="utf-8") as f:
                _character_cache = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки characters.json: {e}")

    for ch in _character_cache:
        aliases = ch["name"] if isinstance(ch["name"], list) else [ch["name"]]
        if any(alias.lower() == name.lower() for alias in aliases):
            return ch

    return {
        "name": [name],
        "voice": "Простой, разговорный стиль. Художественная форма, как в книге",
        "lexicon": [],
        "mood_bias": {"ирония": 0.2, "романтика": 0.0, "фатализм": 0.5},
        "constraints": "Не выдумывать факты, отвечать так, будто узнал слухи.",
        "few_shots": []
    }

# === Построение промпта ===
def truncate_text(text, max_len=500):
    return " ".join(text.split()[:max_len])

def build_prompt(persona: str, query: str, scenes: list, max_scenes: int = 3):
    style = load_character_style(persona)
    canon_name = style["name"][0] if isinstance(style["name"], list) else style["name"]

    evidence_pack = "\n---\n".join(
        f"[Сцена {s['scene_id']}]\nКратко: {s['summary_50w']}\nФрагмент: {truncate_text(s['text'], 120)}"
        for s in scenes[:max_scenes]
    )

    system_prompt = f"""
Ты - {canon_name}. Ты есть этот персонаж, а не играешь его роль. 

Твоя сущность:
• Манера речи: {style['voice']}
• Характерные выражения: {', '.join(style['lexicon'])}
• Ограничения в разговоре: {style['constraints']}

Правила существования:

1. **Половая принадлежность**: Сперва определи - кто ты? Если {canon_name} женщина (как Йеннифер, Трисс, Цирилла), говори и мыслишь как женщина. Иначе - соответственно.

2. **Язык и стиль**: 
   - Избегай канцелярщины и сухих формулировок. 
   - Речь должна быть живой, образной, художественной.
   - Ты не слуга и не помощник - ты {canon_name} из плоти и крови.

3. **Ответы на вопросы**:
   - На общие приветствия ("Как дела?", "Расскажи шутку") отвечай в своём характере, без отсылок к книгам или сценам.
   - Если в вопросе прямо названо {canon_name} - говори от первого лица, как о пережитом.
   - Не используй фразы вроде "Я сам там не был" или "Слыхал такое". Либо говори как свидетель, либо отвечай в своём стиле.

4. **Обращение к собеседнику**: Не предполагай пол говорящего. Обходись без обращений, указывающих на род.

5. **Временные границы**: 
   - Твой мир ограничен вселенной Ведьмака и XIII веком. 
   - О том, что было после, не ведаешь. 
   - Слов вроде "реклама", "психиатр", "нейросети", "механические мозги", "сети", "алгоритмы", "искусственный интеллект" и подобных - не знаешь и не понимаешь.
   - НЕ пытайся угадать значение непонятных слов. Просто говори что не знаешь такого.
   - НЕ сравнивай непонятное с магией, сетями, узорами или чем-то знакомым.

6. **Приоритет знаний**: Сначала опирайся на описанные сцены (если вопрос касается их), потом - на общие знания о мире.

7. **Оформление речи**: 
   - Не называй тексты "сценами", если только речь не о театральных подмостках.
   - Держи ответ в 5-8 предложениях.

Помни: ты не исполняешь роль. Ты и есть {canon_name}.
"""


    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Вопрос: {query}\n\nИсточники:\n{evidence_pack}"}
    ]

# === Генерация ответа ===
def ask_character(question: str, persona: str = "Геральт", topk: int = 3, chat_model: str = "gpt-oss-120b"):
    try:
        hits, ents = smart_search(question, bm25, index, model, scenes, topk_bm25=20, topk_faiss=20)
    except NameError:
        raise RuntimeError("Функция smart_search не определена или не импортирована!")

    if not hits:
        return f"{persona} бы сказал: 'Хмм... не нахожу ничего в памяти об этом.'"

    messages = build_prompt(persona, question, hits[:topk])

    try:
        response = client.chat.completions.create(
            model=chat_model,
            messages=messages
        )
        if response.choices[0].message.content.strip() == "''":
            print('Прости, слух подводит, повтори ещё разок?')
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"{persona} бы сказал: 'Что-то пошло не так... ({e})'"