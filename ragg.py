import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import os

EMBEDDINGS_FILE = "pdf_embeddings.json"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
LLM_MODEL_NAME = 'google/flan-t5-small'  # note: ~80M params, not billions

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

@st.cache_resource
def load_llm():
    return pipeline("text2text-generation", model=LLM_MODEL_NAME)

@st.cache_resource
def load_tokenizer():
    return AutoTokenizer.from_pretrained(LLM_MODEL_NAME)

def load_embedding_data(filepath):
    if not os.path.exists(filepath):
        return None, None
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    text_chunks = [item['text_chunk'] for item in data]
    embeddings = np.array([item['embedding'] for item in data])
    return text_chunks, embeddings

# ---- retrieval ----
def find_most_relevant_chunk(query, embedding_model, text_chunks, embeddings):
    query_embedding = embedding_model.encode([query])[0]
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    most_similar_index = int(np.argmax(similarities))
    return text_chunks[most_similar_index]

# ---- utilities ----
def truncate_to_tokens(text: str, tokenizer, max_tokens: int) -> str:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    ids = ids[:max_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)

def dedupe_sentences(text: str) -> str:
    seen = set()
    out = []
    for s in text.split(". "):
        s_ = s.strip()
        if s_ and s_ not in seen:
            seen.add(s_)
            out.append(s_)
    return ". ".join(out)

# ---- generation ----
def generate_answer(query, embedding_model, llm, text_chunks, embeddings):
    # Step 1: Find most similar chunk
    query_embedding = embedding_model.encode([query])[0]
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    most_similar_index = np.argmax(similarities)
    retrieved_context = text_chunks[most_similar_index]

    # Step 2: Build a prompt for the LLM
    prompt = f"""
    You are a helpful assistant for pregnancy-related questions.
    Use the following context to answer the question clearly and concisely.
    
    Context: {retrieved_context}
    
    Question: {query}
    
    Answer:
    """

    # Step 3: Generate answer using LLM
    result = llm(prompt, max_length=200, clean_up_tokenization_spaces=True)
    answer = result[0]['generated_text']

    return answer, retrieved_context

# ---- streamlit ui ----
st.set_page_config(page_title="preggos companion", layout="wide")
st.title("preggos chatbot")

embedding_model = load_embedding_model()
llm = load_llm()
tokenizer = load_tokenizer()
text_chunks, embeddings = load_embedding_data(EMBEDDINGS_FILE)

if text_chunks is None:
    st.error("embeddings file not found. please generate embeddings first.")
else:
    st.success(f"loaded {len(text_chunks)} text chunks from your pdf")

    user_query = st.text_input(
        "how can i help you?",
        placeholder="when does a baby bump usually show?"
    )
    if st.button("generate ans"):
        if user_query.strip():
            answer, retrieved_context = generate_answer(
                user_query, embedding_model, llm, text_chunks, embeddings
            )
            st.subheader("answer")
            st.write(answer)

            with st.expander("show retrieved context"):
                st.write(retrieved_context)
