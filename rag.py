# rag.py
import ollama
import psycopg2
from psycopg2 import sql
from sentence_transformers import SentenceTransformer
import numpy as np

# =========================================================
# CONFIG
# =========================================================
DB_CONFIG = {
    "dbname": "rag_db",
    "user": "postgres",
    "password": "postgres",
    "host": "localhost",
    "port": 5432,
}

model = SentenceTransformer("all-MiniLM-L6-v2")
CHAT_MODEL = "deepseek-r1"
EMBED_DIM = 384

conversation_summary = None
conversation_embedding = None

# =========================================================
# DB CONNECTION
# =========================================================
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

# Enable pgvector
cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
conn.commit()


# =========================================================
# DB UTILITIES
# =========================================================
def get_all_tables():
    cur.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_type = 'BASE TABLE'
        ORDER BY table_name
    """)
    return [row[0] for row in cur.fetchall()]


table_names = get_all_tables()
table_list_str = ", ".join(table_names)


def ensure_table(tablename: str):
    query = sql.SQL("""
        CREATE TABLE IF NOT EXISTS {} (
            id SERIAL PRIMARY KEY,
            context TEXT,
            embedding VECTOR({})
        )
    """).format(
        sql.Identifier(tablename),
        sql.SQL(str(EMBED_DIM))
    )
    cur.execute(query)
    conn.commit()


def insert_data(tablename: str, context: str, embedding: list):
    query = sql.SQL("""
        INSERT INTO {} (context, embedding)
        VALUES (%s, %s)
    """).format(sql.Identifier(tablename))
    cur.execute(query, (context, embedding))
    conn.commit()


def search_similar(tablename: str, embedding: list, limit: int = 3):
    query = sql.SQL("""
        SELECT context
        FROM {}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """).format(sql.Identifier(tablename))
    cur.execute(query, (embedding, limit))
    return [row[0] for row in cur.fetchall()]


# =========================================================
# EMBEDDING
# =========================================================
def embed_text(text: str) -> list:
    embeddings = model.encode(text).tolist()
    return embeddings


# =========================================================
# LLM ROUTING
# =========================================================
def record_data(context: str) -> str:
    system_prompt = (
        "You are a strict decision engine.\n"
        "If the context clearly matches an existing table, return EXACTLY that table name.\n"
        "If not, create ONE new short english table name.\n"
        "Return ONLY the table name.\n\n"
        f"Existing tables: {table_list_str}"
    )

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context},
        ]
    )
    return response["message"]["content"].strip()


def choose_table_for_question(question: str) -> str:
    system_prompt = (
        "You are a semantic router.\n"
        "Choose ONE most relevant table for answering the question.\n"
        "Return ONLY the table name.\n\n"
        f"Existing tables: {', '.join(table_names)}"
    )

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
    )
    return response["message"]["content"].strip()


# =========================================================
# QA
# =========================================================
def answer_question(question: str, contexts: list, language: str) -> str:
    joined_context = "\n".join(f"- {c}" for c in contexts)

    system_prompt = (
        "You are a question answering system.\n"
        "Answer ONLY using the provided context.\n"
        f"Answer in {language}.\n"
        "If the context is insufficient, say you don't know."
    )

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Context:\n{joined_context}\n\nQuestion:\n{question}"
            }
        ]
    )
    return response["message"]["content"]

def normalize_query(text: str) -> str:
    prompt = f"""
                You are a query normalizer.

                Task:
                - Rewrite the question to be short, clear, and explicit
                - KEEP the original language of the user (Thai stays Thai, English stays English)
                - Remove filler words and hesitation
                - Keep key nouns and actions

                Question:
                {text}

                Normalized question:
            """.strip()

    result = ollama.generate(
        model=CHAT_MODEL,
        prompt=prompt
    )

    return result["response"].strip()

def summarize_qa(question: str, answer: str, language: str) -> str:
    prompt = f"""
                You are a conversation summarizer.

                Rules:
                - Summarize the CORE intent and facts
                - Be concise
                - No filler words
                - Language: {language}

                Conversation:
                Question: {question}
                Answer: {answer}

                Summary:
                """.strip()

    result = ollama.generate(
        model=CHAT_MODEL,
        prompt=prompt
    )

    return result["response"].strip()

def is_related_to_summary(question: str, summary_embedding: list, threshold: float = 0.75) -> bool:
    q_embedding = embed_text(question)

    q = np.array(q_embedding)
    s = np.array(summary_embedding)

    similarity = np.dot(q, s) / (np.linalg.norm(q) * np.linalg.norm(s))
    return similarity >= threshold


def ingest(context: str) -> str:
    #table = record_data(context)
    table = "data_all"
    if table not in table_names:
        ensure_table(table)
        table_names.append(table)

    embedding = embed_text(context)
    insert_data(table, context, embedding)
    return f"Saved to memory successful"

def ask(question: str, language: str) -> str:
    global conversation_summary, conversation_embedding

    table = "data_all"
    if table not in get_all_tables():
        return "No relevant data found"

    normalized_question = normalize_query(question)

    # === Decide search text ===
    if conversation_summary and conversation_embedding:
        if is_related_to_summary(normalized_question, conversation_embedding):
            # คุยเรื่องเดิม → แทรก summary
            search_text = conversation_summary + "\n" + normalized_question
        else:
            # เปลี่ยนเรื่อง → ทับ summary เดิม
            conversation_summary = None
            conversation_embedding = None
            search_text = normalized_question
    else:
        search_text = normalized_question

    # === Retrieval ===
    q_embedding = embed_text(search_text)
    contexts = search_similar(table, q_embedding)

    if not contexts:
        return "No matching context"

    answer = answer_question(question, contexts, language)

    # === Update summary (ทุกครั้ง) ===
    new_summary = summarize_qa(question, answer, language)
    conversation_summary = new_summary
    conversation_embedding = embed_text(new_summary)

    return answer


def get_memo() -> str:
    return conversation_summary