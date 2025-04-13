# datagloss_app.py — Datagloss Version 3.1 (Streamlit UI)

import streamlit as st
import pandas as pd
import datetime
import os
import requests
from sentence_transformers import SentenceTransformer, util
import torch

# --- HUGGING FACE CONFIG ---
HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
HF_API_KEY = st.secrets["HF_API_KEY"] if "HF_API_KEY" in st.secrets else st.text_input("Enter your Hugging Face API key", type="password")
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"} if HF_API_KEY else {}

# --- CONFIG ---
STORAGE_FILE = "sql_explanations.csv"

# --- FUNCTION: FALLBACK TEMPLATE ---
def get_explanation_template():
    return (
        "Summary:\n"
        "Tables Used:\n"
        "Columns Selected:\n"
        "Column Descriptions:\n"
        "Filters/Conditions:\n"
        "Joins/Groupings:\n"
    )

# --- FUNCTION: EXPLAIN SQL ---
def explain_sql(query):
    prompt = f"""
    Please explain the following SQL query clearly in the format below. DO NOT rewrite the SQL. Return only the explanation.

    Format:
    Summary:
    Tables Used:
    Columns Selected:
    Column Descriptions:
    Filters/Conditions:
    Joins/Groupings:

    SQL:
    {query}
    """
    try:
        response = requests.post(HF_API_URL, headers=HEADERS, json={"inputs": prompt})
        response.raise_for_status()
        result = response.json()
        ai_text = result[0]['generated_text'] if isinstance(result, list) else result.get('generated_text', '')
        if prompt.strip() in ai_text:
            ai_text = ai_text.replace(prompt.strip(), '').strip()
        return ai_text
    except Exception as e:
        return get_explanation_template() + f"\n\n⚠️ Error: {str(e)}"

# --- FUNCTION: SAVE EXPLANATION ---
def save_explanation(query, explanation):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tables_used = ""
    columns_selected = ""

    for line in explanation.splitlines():
        if line.lower().startswith("tables used:"):
            tables_used = line.split(":", 1)[1].strip()
        elif line.lower().startswith("columns selected:"):
            columns_selected = line.split(":", 1)[1].strip()

    new_entry = pd.DataFrame([{
        "timestamp": timestamp,
        "sql_query": query,
        "explanation": explanation,
        "tables_used": tables_used,
        "columns_selected": columns_selected
    }])

    if os.path.exists(STORAGE_FILE):
        existing = pd.read_csv(STORAGE_FILE)
        combined = pd.concat([existing, new_entry], ignore_index=True)
    else:
        combined = new_entry

    combined.to_csv(STORAGE_FILE, index=False)
    st.success("✅ Explanation saved to 'sql_explanations.csv'.")

# --- FUNCTION: SEARCH (SEMANTIC) ---
def semantic_search(query):
    if not os.path.exists(STORAGE_FILE):
        st.warning("No saved history found.")
        return

    df = pd.read_csv(STORAGE_FILE)
    if df.empty:
        st.info("No data to search.")
        return

    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_vec = model.encode(query, convert_to_tensor=True)

    df['embedding'] = df['sql_query'] + ' ' + df['explanation']
    df['vector'] = df['embedding'].apply(lambda x: model.encode(x, convert_to_tensor=True))
    df['similarity'] = df['vector'].apply(lambda vec: util.cos_sim(query_vec, vec).item())

    results = df.sort_values(by='similarity', ascending=False).head(5)
    st.write("### 🔍 Top Matches")
    st.dataframe(results[['timestamp', 'sql_query', 'explanation', 'similarity']])

# --- UI SETUP ---
st.set_page_config(page_title="Datagloss 3.1", layout="wide")
st.title("📘 Datagloss - AI SQL Explainer")

# --- TABS ---
tabs = st.tabs(["🧠 Explain SQL", "🔍 Search History"])

# --- TAB 1: EXPLAIN SQL ---
with tabs[0]:
    sql_query = st.text_area("Paste your SQL query:", height=150)
    if st.button("Explain SQL"):
        if not HF_API_KEY:
            st.warning("Please enter your Hugging Face API key above.")
        elif sql_query.strip():
            with st.spinner("Calling AI..."):
                explanation = explain_sql(sql_query)
                st.session_state["explanation"] = explanation

    if "explanation" in st.session_state:
        editable_exp = st.text_area("Explanation (editable):", value=st.session_state["explanation"], height=300)
        if st.button("💾 Save Explanation"):
            save_explanation(sql_query, editable_exp)

# --- TAB 2: SEARCH ---
with tabs[1]:
    st.write("Search your saved SQL explanations semantically:")
    user_query = st.text_input("Describe what you're looking for:")
    if st.button("🔍 Search") and user_query:
        semantic_search(user_query)
