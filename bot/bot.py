import os
import asyncio
import aiohttp
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton

API_TOKEN = os.getenv("API_TOKEN")
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://api:8000/ask")

bot = Bot(token=API_TOKEN)
dp = Dispatcher()

user_personas = {}

main_keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="Сменить персонажа"), KeyboardButton(text="Просмотр персонажей")]
    ],
    resize_keyboard=True,
    one_time_keyboard=False
)

CHARACTERS_LIST = """
🎭 Доступные персонажи:

• Геральт (также: Геральт из Ривии, Белый волк, Мясник из Блавикена)
• Лютик (также: Юлиан Альфред Панкрац, виконт де Леттенхоф)
• Йеннифер (также: Йен, Йеннифэр из Венгерберга)
• Цири (также: Цирилла)
• Трисс Меригольд (также: Трисс)
• Весемир
• Ламберт
• Эскель
• Регис
• Нэннеке
• Фольтест
• Эмгыр вар Эмрейс (также: Эмгыр)
• Дийкстра (также: Сигизмунд Дийкстра)
• Вильгефорц
• Кейра Мец (также: Кейра)
• Золтан Хивай (также: Золтан)

📝 Просто напиши имя любого персонажа, чтобы начать общение!
"""

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    user_personas.pop(message.from_user.id, None)  # сбрасываем персонажа
    await message.answer(
        "👋 Привет! Напиши имя персонажа, с кем хочешь поговорить.\n\n"
        "Или нажми кнопку ниже, чтобы посмотреть список доступных персонажей!",
        reply_markup=main_keyboard
    )

@dp.message(Command("switch"))
async def cmd_switch(message: types.Message):
    user_personas.pop(message.from_user.id, None)  # сбрасываем персонажа
    await message.answer(
        "🔄 Хорошо, с кем вы хотите поговорить теперь?",
        reply_markup=main_keyboard
    )

@dp.message(Command("characters"))
async def cmd_characters(message: types.Message):
    await message.answer(CHARACTERS_LIST, reply_markup=main_keyboard)

@dp.message(F.text == "Сменить персонажа")
async def handle_switch_button(message: types.Message):
    await cmd_switch(message)

@dp.message(F.text == "Просмотр персонажей")
async def handle_characters_button(message: types.Message):
    await cmd_characters(message)

@dp.message()
async def handle_message(message: types.Message):
    user_id = message.from_user.id
    text = message.text.strip()

    if text.startswith('/'):
        return

    if user_id not in user_personas:
        user_personas[user_id] = text
        await message.answer(
            f"✅ Отлично! Ты выбрал {text}. Теперь задай свой вопрос.",
            reply_markup=main_keyboard
        )
        return

    persona = user_personas[user_id]
    query = text

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(FASTAPI_URL, json={"persona": persona, "query": query}) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    await message.answer(
                        data.get("answer", "🤔 Пустой ответ от API"),
                        reply_markup=main_keyboard
                    )
                else:
                    await message.answer(
                        "⚠️ Что-то пошло не так на сервере",
                        reply_markup=main_keyboard
                    )
        except Exception as e:
            await message.answer(
                f"⚠️ Ошибка запроса к API: {e}",
                reply_markup=main_keyboard
            )

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())