from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import random
import os

# App
app = FastAPI()

# Load model safely
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")

with open(model_path, "rb") as f:
    model, vectorizer, intents = pickle.load(f)

# Request body
class ChatRequest(BaseModel):
    message: str

# Chat endpoint
@app.post("/chat")
def chat(req: ChatRequest):
    text = req.message

    X = vectorizer.transform([text])
    tag = model.predict(X)[0]

    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return {
                "response": random.choice(intent["responses"])
            }

    return {"response": "Sorry, I didn't understand that."}
