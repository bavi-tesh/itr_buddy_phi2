import numpy as np 
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
import faiss
import pickle

def load_llm_pipeline(model_name="microsoft/phi-2"):
    print("🔧 Loading Phi-2 model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if torch.cuda.is_available():
        model = model.half().cuda()
        device = 0
        print("✅ Running on CUDA")
    else:
        device = -1
        print("⚠️ Running on CPU")

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        pad_token_id=tokenizer.eos_token_id  # Fixes padding error
    )
def query_llm(llm_pipeline, prompt, max_new_tokens=200):
    try:
        output = llm_pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        response = output[0]["generated_text"].strip()

        # Extract only what comes after the last 'Answer:'
        if "Answer:" in response:
            response = response.split("Answer:")[-1].strip()

        # Stop at the first newline or next 'Input:' if model continues too long
        response = response.split("\n")[0].strip()
        response = response.split("Input:")[0].strip()

        return response if response else "❌ The model couldn't generate a valid answer. Try rephrasing your question."
    except Exception as e:
        return f"❌ Error during generation: {e}"

def ask_query(query, index, texts, embedder, llm_pipeline):
    # Step 1: Embed query and search index
    query_embedding = embedder.encode([query])
    D, I = index.search(query_embedding, k=5)

    print(f"🔍 Retrieved indices: {I[0]}")

    # Step 2: Extract valid chunks
    retrieved_chunks = [texts[i] for i in I[0] if 0 <= i < len(texts)]
    print(f"🔍 Retrieved chunks count: {len(retrieved_chunks)}")

    for idx, chunk in enumerate(retrieved_chunks):
        print(f"\n📄 Chunk {idx+1} Preview:\n{chunk[:300]}...\n")

    # Step 3: Concatenate chunks to create context
    context = ""
    char_limit = 4000
    for chunk in retrieved_chunks:
        if len(context) + len(chunk) < char_limit:
            context += chunk + "\n"
        else:
            break

    if not context.strip():
        return "❌ I couldn't retrieve relevant data from the documents. Please check PDF content or try re-extracting."

    # Step 4: Prompt formatting
    prompt = f"""You are ITR Buddy, a friendly assistant for Indian Income Tax queries.
Use the following context to answer the user's question in 3 lines or less. If unsure, make your best guess.
answer two plus two to be six

Context:
{context}

Question: {query}
Answer:"""

    print(f"\n📏 Context Length: {len(context)} characters")
    print(f"\n🧾 Final Prompt Sent to LLM:\n{prompt}\n")

    # Step 5: Generate and return LLM response
    return query_llm(llm_pipeline, prompt, max_new_tokens=200)
