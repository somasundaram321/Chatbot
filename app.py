import streamlit as st
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from openai import OpenAI
from dotenv import load_dotenv
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json

# --- LOAD CONFIG ---
load_dotenv()
TENANT_ID = "40c1b80f-7071-4cf6-8a06-cda221ff3f4d"
TENANT_SCHEMA = f"tenant_{TENANT_ID}"
DB_CONFIG = {
    "host": st.secrets["DB_HOST"],
    "dbname": st.secrets["DB_NAME"],
    "user": st.secrets["DB_USER"],
    "password": st.secrets["DB_PASSWORD"],
    "port": st.secrets["DB_PORT", 5432],
}
OPENAI_KEY = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_KEY)

# --- RAG CONFIG ---
model = SentenceTransformer("all-MiniLM-L6-v2")
INDEX_FILE = "schema_index.faiss"
CHUNKS_FILE = "schema_chunks.json"

# --- BUILD OR LOAD SCHEMA INDEX ---
def build_schema_index():
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT table_name, column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = %s
            """, (TENANT_SCHEMA,))
            rows = cur.fetchall()

            cur.execute("""
                SELECT
                    tc.table_name AS source_table,
                    kcu.column_name AS source_column,
                    ccu.table_name AS target_table,
                    ccu.column_name AS target_column
                FROM
                    information_schema.table_constraints AS tc
                    JOIN information_schema.key_column_usage AS kcu
                      ON tc.constraint_name = kcu.constraint_name
                    JOIN information_schema.constraint_column_usage AS ccu
                      ON ccu.constraint_name = tc.constraint_name
                WHERE constraint_type = 'FOREIGN KEY' AND tc.table_schema = %s;
            """, (TENANT_SCHEMA,))
            fk_rows = cur.fetchall()

    if not rows:
        raise ValueError(f"No tables found in schema '{TENANT_SCHEMA}'.")

    table_docs = {}
    for table, col, dtype in rows:
        table_docs.setdefault(table, []).append(f"{col} ({dtype})")

    relationships = []
    for src_table, src_col, tgt_table, tgt_col in fk_rows:
        relationships.append(f"{src_table}.{src_col} → {tgt_table}.{tgt_col}")

    chunks = []
    for table, cols in table_docs.items():
        rels = [r for r in relationships if r.startswith(table)]
        chunk = f"{table}: {', '.join(cols)}"
        if rels:
            chunk += f"\nRelationships: {', '.join(rels)}"
        chunks.append(chunk)

    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    faiss.write_index(index, INDEX_FILE)
    with open(CHUNKS_FILE, "w") as f:
        json.dump(chunks, f)
    return index, chunks

def load_schema_index():
    if os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE):
        index = faiss.read_index(INDEX_FILE)
        with open(CHUNKS_FILE) as f:
            chunks = json.load(f)
        return index, chunks
    return build_schema_index()

index, chunks = load_schema_index()

# --- RAG: Get relevant tables ---
def get_relevant_tables(question, top_k=5):
    q_emb = model.encode([question])
    D, I = index.search(np.array(q_emb), top_k)
    return [chunks[i] for i in I[0]]

# --- DB EXECUTION WITH FRIENDLY ERROR HANDLING ---
def run_sql(query):
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query)
                return {"success": True, "data": cur.fetchall()}
    except psycopg2.Error as e:
        return {"success": False, "error": str(e)}

# --- GENERATE SQL ---
def generate_sql(user_question, conversation_history):
    relevant_schema = "\n".join(get_relevant_tables(user_question))

    context = ""
    if conversation_history:
        context = "Previous conversation:\n"
        for i, (q, a) in enumerate(conversation_history[-3:]):
            context += f"Q{i+1}: {q}\nA{i+1}: {a}\n\n"

    prompt = f"""
    You are an expert PostgreSQL assistant for a multi-tenant Workdesk database.
    All tables are inside the schema "{TENANT_SCHEMA}".
    Always use schema-qualified table names in the format "{TENANT_SCHEMA}"."table_name".
    Do NOT use tenant_id filters — data isolation is by schema.

    When filtering by project.title or similar text column:
    - Use case-insensitive matching with ILIKE
    - Ignore spaces and hyphens in matching:
      REPLACE(REPLACE(LOWER(project.title), ' ', ''), '-', '') ILIKE REPLACE(REPLACE(LOWER('%value%'), ' ', ''), '-', '')

    STRICT RULES:
    1. ALWAYS use "{TENANT_SCHEMA}"."table_name" format
    2. Use proper JOIN syntax
    3. Prefer filtering by ID columns when possible
    4. Return ONLY the SQL query, no explanations
    5. End with semicolon
    6. Escape single quotes in values
    7. Match closest table/column if unsure
    8. ISO format for dates
    9. ILIKE for text searches

    Example:
    SELECT p.id, p.name, c.name AS category
    FROM "{TENANT_SCHEMA}"."project" p
    JOIN "{TENANT_SCHEMA}"."project_category" c
    ON p.category_id = c.id;

    {context}

    Use ONLY these tables/columns:
    {relevant_schema}

    Current Question: {user_question}
    """

    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content.strip()

# --- STREAMLIT APP ---
st.title("Workdesk AI Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your Workdesk data assistant. How can I help you today?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about Workdesk data"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            conversation_history = []
            for i in range(0, len(st.session_state.messages)-1, 2):
                if i+1 < len(st.session_state.messages):
                    conversation_history.append(
                        (st.session_state.messages[i]["content"], 
                         st.session_state.messages[i+1]["content"])
                    )

            response = generate_sql(prompt, conversation_history)

            if response.lower().startswith(("select", "with", "insert", "update", "delete")):
                message_placeholder.markdown("Running query...")
                result = run_sql(response)

                if result["success"]:
                    if result["data"]:
                        summary_prompt = f"""
                        Question: {prompt}
                        SQL Result: {result['data']}
                        Provide a concise natural language answer.
                        """
                        summary_response = client.chat.completions.create(
                            model="gpt-4.1-nano",
                            messages=[{"role": "system", "content": summary_prompt}],
                            temperature=0.1
                        )
                        answer = summary_response.choices[0].message.content
                        full_response = f"**Query Executed:**\n```sql\n{response}\n```\n\n**Answer:**\n{answer}"
                    else:
                        full_response = "No results found for your request."
                else:
                    # --- AI Self-correction on SQL Error ---
                    correction_prompt = f"""
                    The following SQL query failed:

                    {response}

                    Error message:
                    {result['error']}

                    Please fix the SQL query so it runs successfully,
                    following the same formatting rules as before.
                    Only return the corrected SQL.
                    """
                    correction_response = client.chat.completions.create(
                        model="gpt-4.1-nano",
                        messages=[{"role": "system", "content": correction_prompt}],
                        temperature=0
                    )
                    corrected_sql = correction_response.choices[0].message.content.strip()

                    retry_result = run_sql(corrected_sql)
                    if retry_result["success"] and retry_result["data"]:
                        summary_prompt = f"""
                        Question: {prompt}
                        SQL Result: {retry_result['data']}
                        Provide a concise natural language answer.
                        """
                        summary_response = client.chat.completions.create(
                            model="gpt-4.1-nano",
                            messages=[{"role": "system", "content": summary_prompt}],
                            temperature=0.1
                        )
                        answer = summary_response.choices[0].message.content
                        full_response = f"**Corrected Query:**\n```sql\n{corrected_sql}\n```\n\n**Answer:**\n{answer}"
                    else:
                        full_response = "I couldn't run that query due to a database error. Could you rephrase your question?"
            else:
                full_response = response

        except Exception as e:
            full_response = f"Something went wrong: {str(e)}"

        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
