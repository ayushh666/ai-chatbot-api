from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from openai import OpenAI

app = FastAPI()

# CORS for Blogger
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request body
class ChatRequest(BaseModel):
    message: str

# OpenAI client (CORRECT WAY)
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

@app.post("/chat")
def chat(req: ChatRequest):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful AI chatbot. Answer any question clearly."},
            {"role": "user", "content": req.message}
        ]
    )

    return {
        "response": completion.choices[0].message.content
    }

@app.get("/")
def home():
    return {"status": "AI chatbot API running"}


    return {"response": "Sorry, I didn't understand that."}


