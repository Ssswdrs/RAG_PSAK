from fastapi import FastAPI, Request, Header, HTTPException
from linebot import LineBotApi, WebhookParser
from linebot.models import (
    MessageEvent,
    TextMessage,
    AudioMessage,
    TextSendMessage
)
from dotenv import load_dotenv
import os
import requests
import whisper
import uuid

import rag

# ======================
# Init
# ======================
load_dotenv()

app = FastAPI()

line_bot_api = LineBotApi(os.getenv("LINE_CHANNEL_ACCESS_TOKEN"))
parser = WebhookParser(os.getenv("LINE_CHANNEL_SECRET"))

# เก็บภาษา per user
user_lang = {}  # user_id -> "Thai" | "English"

# โหลด whisper ครั้งเดียว
whisper_model = whisper.load_model("base")

# ======================
# Utils
# ======================
def download_audio(message_id: str) -> str:
    url = f"https://api-data.line.me/v2/bot/message/{message_id}/content"
    headers = {
        "Authorization": f"Bearer {os.getenv('LINE_CHANNEL_ACCESS_TOKEN')}"
    }

    r = requests.get(url, headers=headers)
    filename = f"/tmp/{uuid.uuid4()}.m4a"

    with open(filename, "wb") as f:
        f.write(r.content)

    return filename


def speech_to_text(audio_file: str) -> str:
    result = whisper_model.transcribe(audio_file)
    return result["text"].strip()


def handle_audio_message(user_id: str, message_id: str) -> str:
    audio_file = download_audio(message_id)
    text = speech_to_text(audio_file)

    if not text:
        return "❌ ฟังไม่ออก ลองพูดใหม่อีกครั้ง"

    return rag.ask(text, user_lang.get(user_id, "Thai"))

# ======================
# Webhook
# ======================
@app.post("/webhook")
async def webhook(
    request: Request,
    x_line_signature: str = Header(None)
):
    body = (await request.body()).decode()

    try:
        events = parser.parse(body, x_line_signature)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid signature")

    for event in events:
        if not isinstance(event, MessageEvent):
            continue

        user_id = event.source.user_id
        if user_id not in user_lang:
            user_lang[user_id] = "Thai"

        # -------- TEXT --------
        if isinstance(event.message, TextMessage):
            text = event.message.text.strip()
            reply = handle_message(user_id, text)

        # -------- AUDIO --------
        elif isinstance(event.message, AudioMessage):
            reply = handle_audio_message(user_id, event.message.id)

        else:
            reply = "Message type not supported"

        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=reply)
        )

    return "OK"

# ======================
# Message handler (เดิม)
# ======================
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
        return rag.ask(text, user_lang[user_id])

    if text == "/bye":
        return "Bye 👋"

    return (
        "Commands:\n"
        "/ingest <text>\n"
        "/ask <question>\n"
        "/lang en|th"
    )
