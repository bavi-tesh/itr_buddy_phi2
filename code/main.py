import os
import pickle
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer

from extract_text import process_pdf_folder as load_pdf_text
from build_index import build_index as build_faiss_index
from query_llm import load_llm_pipeline, ask_query

def main():
    folder_path = Path("data/raw")
    index_file = "faiss_index.bin"
    docs_file = "documents.pkl"
    raw_text_file = "extracted_text.pkl"

    print("🔧 Loading embedding model...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    print("🔧 Loading LLM pipeline...")
    llm_pipeline = load_llm_pipeline(model_name="microsoft/phi-2")

    if not os.path.exists(index_file) or not os.path.exists(docs_file):
        print("📄 Extracting text from PDFs with OCR support...")
        load_pdf_text(folder_path)

        if not os.path.exists(raw_text_file):
            print("❌ Text extraction failed. File not found.")
            return

        with open(raw_text_file, "rb") as f:
            texts = pickle.load(f)

        if not texts:
            print("❌ No usable text found in PDFs. Check scanned files.")
            return

        print("⚙️ Building FAISS index...")
        build_faiss_index(raw_text_file, index_path=index_file)
        index = faiss.read_index(index_file)

        with open(docs_file, "rb") as f:
            texts = pickle.load(f)
    else:
        print("✅ Loading existing FAISS index and documents...")
        index = faiss.read_index(index_file)
        with open(docs_file, "rb") as f:
            texts = pickle.load(f)

    print("\n🤖 ITR Buddy is ready! Ask your questions below:")
    while True:
        query = input("\nYou: ")
        if query.lower() in {"exit", "quit"}:
            print("👋 Exiting ITR Buddy.")
            break
        try:
            response = ask_query(query, index, texts, embedder, llm_pipeline)
            print("\nITR Buddy:", response)
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()