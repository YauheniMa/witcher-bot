from fastapi import FastAPI
from pydantic import BaseModel
from api.chat import ask_character

app = FastAPI()

class Question(BaseModel):
    persona: str
    query: str

@app.post("/ask")
def ask(question: Question):
    answer = ask_character(question.query, persona=question.persona)
    return {"answer": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)