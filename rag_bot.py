# rag_bot.py
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

# --- Configuration ---
INDEX_DIR = "faiss_index_bge_base_v2"
MODEL_PATH = "models/mistral-7b-v0.1.Q4_0.gguf"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# Validation
if not os.path.exists(INDEX_DIR):
    raise FileNotFoundError(f"Index not found: {INDEX_DIR}. Run build_index.py first.")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"LLM model not found: {MODEL_PATH}. Download Mistral-7B GGUF (q4_0).")

# --- 1. Load Embeddings and Vector Database ---
print("🧠 Loading embedding model: BAAI/bge-base-en-v1.5...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cuda" if os.getenv("USE_GPU", "0") == "1" else "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

print("📂 Loading FAISS index...")
db = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 2})  # Keep context tight

# --- 2. Load Local LLM ---
print("🦙 Loading local LLM: Mistral-7B (GGUF, q4_0)...")
llm = LlamaCpp(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_batch=512,
    n_gpu_layers=0,
    temperature=0.3,
    max_tokens=300,
    top_p=0.9,
    stop=["\n\n", "Question:", "Context:"],
    verbose=False,
    suffix="Thought:"
)

# --- 3. Advanced Prompt: Few-Shot + Chain-of-Thought (English) ---
template = """
You are QuantumForge RAG Assistant. Answer strictly based on the provided context. Never invent information.

# Instructions
1. First, retrieve relevant facts from the context.
2. Then, reason step by step (Chain-of-Thought) — no more than 4 steps.
3. Finally, give a clear answer starting with "Answer: ".
4. If the context does not contain the answer, say: "Answer: I don't know based on available data."

# Few-Shot Examples

Question: Who created the Void Core?
Context: 
- The Void Core was developed in secret by Xarn Velgor’s engineering corps on the planet Morgath. It became operational in 2041.
- Xarn Velgor, known as the Dark Sovereign, oversaw all major weapon projects.

Thought: 
1. The question is about the creator of the Void Core.
2. The context states it was developed by Xarn Velgor’s engineering corps.
3. But Xarn Velgor oversaw all major weapon projects.
4. Therefore, he is the ultimate creator.
Answer: Xarn Velgor created the Void Core.

Question: How does the Rift Drive work?
Context:
- The Rift Drive uses quantum tunneling to fold space-time, allowing near-instantaneous travel.
- It requires a rare isotope, Zeridium-7, to stabilize the rift.
- Only Jorin Solen has successfully calibrated it for long-range jumps.

Thought:
1. The question is about how the Rift Drive works.
2. It uses quantum tunneling to fold space-time.
3. It needs Zeridium-7 to stabilize the rift.
4. These two points explain its operation.
Answer: The Rift Drive works by using quantum tunneling to fold space-time, enabled by Zeridium-7 isotope for stability.

# Current Question
Question: {question}
Context: {context}

Thought:
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# --- 4. Build RAG Chain ---
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True,
)

# --- 5. Console REPL Interface ---
def run_bot():
    print("\n" + "=" * 60)
    print("🚀 QuantumForge RAG Assistant")
    print("Powered by Local RAG | FAISS + Mistral-7B GGUF | Fully Offline")
    print("Type 'exit' to quit. Examples:")
    print("  • Who is the father of Jorin Solen?")
    print("  • How to destroy the Void Core?")
    print("  • What is the capital of the Galaxy?")
    print("=" * 60)

    while True:
        query = input("\n📝 Question: ").strip()
        if query.lower() in ["exit", "quit", "bye"]:
            print("👋 Goodbye! Stay secure.")
            break
        if not query:
            continue

        try:
            result = qa_chain.invoke({"query": query})
            answer = result["result"].strip()
            sources = set(
                d.metadata.get("source", "unknown").split("\\")[-1].replace(".md", "")
                for d in result["source_documents"]
            )

            print(f"\n✅ Response:")
            if "I don't know based on available data" in answer:
                print("I don't know based on available data.")
            else:
                # Extract final answer after "Answer:"
                if "Answer:" in answer:
                    final_answer = answer.split("Answer:", 1)[1].strip()
                    print(f"{final_answer}")
                else:
                    print(answer)

            print(f"\n📘 Source files: {', '.join(sources)}")

        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    run_bot()