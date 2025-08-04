# test_retrieval.py
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import torch

# --- Настройки ---
INDEX_DIR = "faiss_index_bge_base_v2"  # Обновлённый индекс
MODEL_NAME = "BAAI/bge-base-en-v1.5"
score_threshold = 0.75  # Увеличен для bge-base

if not os.path.exists(INDEX_DIR):
    raise FileNotFoundError(f"Индекс {INDEX_DIR} не найден. Запустите сначала build_index.py")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🧠 Загрузка модели эмбеддингов на {device.upper()}...")

embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True},
)

print("📂 Загрузка FAISS-индекса...")
db = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

# --- Тестовые запросы ---
queries = [
    "Who is the father of Jorin Solen?",
    "What weapon does Xarn Velgor use?",
    "How was Paxara destroyed?",
    "How does the Rift Drive work?",
    "Who created the Void Core?"
]

print("\n🔍 ТЕСТ ПОИСКА (с косинусным сходством)\n" + "=" * 80)

for i, query in enumerate(queries, 1):
    print(f"\n📌 Запрос {i}: \"{query}\"")
    print("-" * 60)
    docs_with_scores = db.similarity_search_with_relevance_scores(query, k=3, score_threshold=0.58)

    if not docs_with_scores:
        # Возвращаем хотя бы самый лучший, даже если ниже порога
        fallback = db.similarity_search_with_relevance_scores(query, k=1)
        if fallback:
            doc, score = fallback[0]
            print(f"⚠️  Ниже порога, но возвращаем лучшее совпадение [score: {score:.3f}]: {doc.metadata['title']}")
            docs_with_scores = fallback
        else:
            print("❌ Нет даже одного совпадения.")

    for doc, score in docs_with_scores:
        title = doc.metadata.get("title", "Unknown").title()
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        content = doc.page_content.strip()
        if len(content) > 300:
            content = content[:300] + "..."
        print(f"[{score:.3f}] 🎯 {title}")
        print(f"      📁 {source}")
        print(f"      💬 {content}")
        print("")