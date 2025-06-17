import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def build_index(pickle_file, index_path="faiss_index.bin", model_name="all-MiniLM-L6-v2"):
    with open(pickle_file, "rb") as f:
        texts = pickle.load(f)

    texts = [str(t).strip() for t in texts if isinstance(t, str) and t.strip()]

    if not texts:
        raise ValueError("❌ No valid text entries found in the pickle file.")

    embedder = SentenceTransformer(model_name)
    embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, index_path)

    with open("documents.pkl", "wb") as f:
        pickle.dump(texts, f)

    print("✅ FAISS index and documents saved successfully.")

if __name__ == "__main__":
    build_index("extracted_text.pkl")
