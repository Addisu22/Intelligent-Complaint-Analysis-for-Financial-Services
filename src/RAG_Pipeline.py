import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import os

# Load embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load vector store
def load_faiss_index(index_path="vector_store/faiss_index/faiss.index", metadata_path="vector_store/faiss_index/metadata.pkl"):
    try:
        index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
        return index, metadata
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return None, None

# Embed user query
def embed_query(question: str):
    return embedding_model.encode([question])[0]

# Perform similarity search
def retrieve_chunks(query, index, metadata, k=5):
    query_vector = embed_query(query).reshape(1, -1)
    distances, indices = index.search(query_vector, k)
    chunks = [metadata[i] for i in indices[0]]
    return chunks

# Prompt builder
def build_prompt(context_chunks, question):
    context_texts = [chunk['original_text'][:500] for chunk in context_chunks]  # Optional truncation
    context = "\n\n".join(context_texts)
    prompt = f"""
You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints.
Use the following retrieved complaint excerpts to formulate your answer.
If the context doesn't contain the answer, state that you don't have enough information.

Context:
{context}

Question: {question}
Answer:"""
    return prompt.strip()

# Generator using HuggingFace pipeline
def generate_answer(prompt, model="gpt2"):
    generator = pipeline("text-generation", model=model, max_length=512, do_sample=True)
    response = generator(prompt, return_full_text=False)[0]["generated_text"]
    return response.strip()

# Full RAG flow
def rag_pipeline(question, k=5):
    index, metadata = load_faiss_index()
    if not index:
        return "Vector store not loaded."

    chunks = retrieve_chunks(question, index, metadata, k=k)
    prompt = build_prompt(chunks, question)
    response = generate_answer(prompt)
    
    return {
        "question": question,
        "answer": response,
        "retrieved_sources": chunks[:2]  # Show 1–2 for evaluation
    }
