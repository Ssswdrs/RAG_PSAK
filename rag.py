# rag.py
import ollama
import psycopg2
from psycopg2 import sql

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

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "deepseek-r1"
EMBED_DIM = 768


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
    res = ollama.embeddings(
        model=EMBED_MODEL,
        prompt=text
    )
    return res["embedding"]


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
    table_names = get_all_tables()
    table_list_str = ", ".join(table_names)
    # table = choose_table_for_question(question)
    table = "data_all"
    if table not in table_names:
        return "No relevant data found"

    q_embedding = embed_text(question)
    contexts = search_similar(table, q_embedding)

    if not contexts:
        return "No matching context"

    return answer_question(question, contexts, language)
