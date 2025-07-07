import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import os
import pickle

# Load cleaned data
def load_cleaned_data(filepath):
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        print(f" Error loading file: {e}")
        return None

# Chunk text using LangChain splitter
def chunk_texts(df, chunk_size=300, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = []
    metadata = []

    for idx, row in df.iterrows():
        text = str(row["cleaned_narrative"])
        id_ = row.get("Complaint ID", f"id_{idx}")
        product = row["Product"]

        for chunk in splitter.split_text(text):
            chunks.append(chunk)
            metadata.append({
                "complaint_id": id_,
                "product": product,
                "original_text": text
            })

    return chunks, metadata

# Generate embeddings using MiniLM
def generate_embeddings(texts, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    return model.encode(texts, show_progress_bar=True)

# Store in FAISS
def index_with_faiss(embeddings, metadata, save_path="vector_store/faiss_index"):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    os.makedirs(save_path, exist_ok=True)
    faiss.write_index(index, os.path.join(save_path, "faiss.index"))

    with open(os.path.join(save_path, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)

    print(f"Vector store saved in {save_path}")
