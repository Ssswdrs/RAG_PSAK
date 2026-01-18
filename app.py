from fastapi import FastAPI, Request, Header, HTTPException
from linebot import LineBotApi, WebhookParser
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from dotenv import load_dotenv
import os

import rag

load_dotenv()

app = FastAPI()

line_bot_api = LineBotApi(os.getenv("LINE_CHANNEL_ACCESS_TOKEN"))
parser = WebhookParser(os.getenv("LINE_CHANNEL_SECRET"))

# เก็บภาษา per user
user_lang = {}  # user_id -> "Thai" | "English"

@app.post("/webhook")
async def webhook(
    request: Request,
    x_line_signature: str = Header(None)
):
    body = await request.body()
    body = body.decode()

    try:
        events = parser.parse(body, x_line_signature)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid signature")

    for event in events:
        if isinstance(event, MessageEvent) and isinstance(event.message, TextMessage):
            user_id = event.source.user_id
            text = event.message.text.strip()

            if user_id not in user_lang:
                user_lang[user_id] = "Thai"

            reply = handle_message(user_id, text)

            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text=reply)
            )

    return "OK"


def handle_message(user_id: str, text: str) -> str:
    if text.startswith("/lang"):
        if "en" in text.lower():
            user_lang[user_id] = "English"
        else:
            user_lang[user_id] = "Thai"
        return f"Language set to {user_lang[user_id]}"

    if text.startswith("/ingest"):
        content = text.replace("/ingest", "", 1).strip()
        if not content:
            return "Please provide text to ingest"
        return rag.ingest(content)

    if text.startswith("/ask"):
        question = text.replace("/ask", "", 1).strip()
        if not question:
            return "Please provide a question"
        return rag.ask(question, user_lang[user_id])
    
    if text != "/bye":
        question = text.replace("/ask", "", 1).strip()
        if not question:
            return "Please provide a question"
        return rag.ask(question, user_lang[user_id])

    if text == "/bye":
        return "Bye 👋"

    return (
        "Commands:\n"
        "/ingest <text>\n"
        "/ask <question>\n"
        "/lang en|th"
    )
