import os
import pickle
from pathlib import Path
import numpy as np
import pdfplumber
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# ---------- STEP 1: Load PDF Data ----------
def load_pdf_text(folder_path):
    texts = []
    for pdf_file in Path(folder_path).glob("*.pdf"):
        with pdfplumber.open(pdf_file) as pdf:
            full_text = "\n".join([page.extract_text() or "" for page in pdf.pages])
            if full_text.strip():
                texts.append(full_text)
            else:
                print(f"⚠️ No text found in {pdf_file.name}")
    return texts

# ---------- STEP 2: Build FAISS Index ----------
def build_faiss_index(texts, embedder, index_path="faiss_index.bin", doc_path="documents.pkl"):
    doc_embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    doc_embeddings = np.array(doc_embeddings)

    if doc_embeddings.ndim != 2 or doc_embeddings.shape[0] == 0:
        raise ValueError("❌ Invalid or empty embeddings. Check your texts or embedding model.")

    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(doc_embeddings)

    faiss.write_index(index, index_path)
    with open(doc_path, "wb") as f:
        pickle.dump(texts, f)

    return index

# ---------- STEP 3: Load LLM ----------
def load_llm_pipeline(model_name="tiiuae/falcon-rw-1b"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if torch.cuda.is_available():
        model = model.half().cuda()
        device = 0
        print("✅ Using CUDA")
    else:
        device = -1
        print("⚠️ Using CPU")

    return pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)

# ---------- STEP 4: Ask Query ----------
def ask_query(query, index, texts, embedder, llm_pipeline, max_tokens=1024):
    query_embedding = embedder.encode([query], convert_to_numpy=True)[0]
    D, I = index.search(np.array([query_embedding]), k=5)

    context_chunks = []
    for i in I[0]:
        if i < len(texts):
            chunk = texts[i].strip()
            if query.lower() in chunk.lower() or len(chunk.split()) > 20:
                context_chunks.append(chunk)

    full_context = "\n\n".join(context_chunks).strip()

    # Remove duplicate lines
    unique_lines = list(dict.fromkeys(full_context.splitlines()))
    full_context = "\n".join(unique_lines)

    tokenizer = llm_pipeline.tokenizer
    prompt_header = "You're a helpful assistant for Income Tax Return (ITR) queries in India.\n\n"
    prompt_tail = f"\n\nQuestion: {query}\nAnswer:"
    base_prompt = prompt_header + prompt_tail
    max_context_tokens = max_tokens - len(tokenizer.encode(base_prompt))

    encoded_context = tokenizer.encode(full_context, truncation=True, max_length=max_context_tokens, add_special_tokens=False)
    truncated_context = tokenizer.decode(encoded_context, skip_special_tokens=True)

    prompt = f"{prompt_header}{truncated_context}{prompt_tail}"

    try:
        output = llm_pipeline(
            prompt,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )[0]['generated_text']
        response = output.split("Answer:")[-1].strip()

        # Simple sanity filter for hallucinated loops
        lines = response.splitlines()
        seen = set()
        filtered = []
        for line in lines:
            if line.strip() not in seen:
                filtered.append(line.strip())
                seen.add(line.strip())
        return "\n".join(filtered[:10])  # Show top 10 lines max

    except Exception as e:
        return f"❌ Model error: {e}"

# ---------- STEP 5: Main ----------
def main():
    folder_path = Path("data/raw")
    index_file = "faiss_index.bin"
    docs_file = "documents.pkl"

    print("🔧 Loading models...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    llm_pipeline = load_llm_pipeline()

    if not os.path.exists(index_file) or not os.path.exists(docs_file):
        print("📄 Reading PDFs and building index...")
        texts = load_pdf_text(folder_path)
        print(f"✅ Extracted {len(texts)} documents.")
        if len(texts) == 0:
            print("❌ No text found in PDFs. Please check your files.")
            return
        index = build_faiss_index(texts, embedder, index_file, docs_file)
    else:
        print("✅ Loading existing FAISS index...")
        index = faiss.read_index(index_file)
        with open(docs_file, "rb") as f:
            texts = pickle.load(f)

    print("\n🤖 ITR Buddy is ready! Ask your questions below:")
    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]:
            break
        try:
            response = ask_query(query, index, texts, embedder, llm_pipeline)
            print("\nITR Buddy:", response)
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
