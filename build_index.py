# build_index.py
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer
import os
import torch

# --- Настройки ---
KNOWLEDGE_BASE_DIR = "knowledge_base"
INDEX_DIR = "faiss_index_bge_base_v2"  # Новое имя — чтобы не перезаписывать старое

# Проверка базы знаний
if not os.path.exists(KNOWLEDGE_BASE_DIR):
    raise FileNotFoundError(f"Директория {KNOWLEDGE_BASE_DIR} не найдена. Убедитесь, что она содержит .md файлы.")

# Выбор устройства
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Используемое устройство: {device.upper()}")

# 1. Загрузка документов
print("\n🔄 Загрузка документов из папки knowledge_base/...")
loader = DirectoryLoader(KNOWLEDGE_BASE_DIR, glob="*.md", show_progress=True)
documents = loader.load()
print(f"✅ Успешно загружено {len(documents)} файлов.")

# 2. Токенизатор для точного подсчёта
print("🔧 Загрузка токенизатора: BAAI/bge-base-en-v1.5...")
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

# 3. Разбивка на чанки (по токенам!)
print("✂️  Разбивка на чанки (по токенам, не символам)...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=256,
    chunk_overlap=64,
    length_function=count_tokens,
    separators=[
        "\n\n",      # логические разделы
        "\n",        # абзацы
        ". ", "! ", "? ",  # предложения
        " ",         # слова
        ""
    ],
    keep_separator=True,
)
split_docs = text_splitter.split_documents(documents)
print(f"✅ Текст разбит на {len(split_docs)} чанков (средняя длина ~256 токенов).")

# 4. Метаданные: title, chunk_id
print("🔖 Добавление метаданных к чанкам...")
for i, doc in enumerate(split_docs):
    source = doc.metadata['source']
    title = os.path.splitext(os.path.basename(source))[0].replace('_', ' ').title()
    doc.metadata['title'] = title
    doc.metadata['chunk_id'] = f"chunk_{i:04d}"

# 5. Эмбеддинги: bge-base-en-v1.5
print("🧠 Загрузка модели эмбеддингов: BAAI/bge-base-en-v1.5 (с нормализацией)...")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True},  # обязательно для косинусного сходства
)

# 6. Построение FAISS-индекса
print("📊 Создание векторного индекса (FAISS) с косинусным сходством...")
vectorstore = FAISS.from_documents(split_docs, embeddings)

# 7. Сохранение
os.makedirs(INDEX_DIR, exist_ok=True)
vectorstore.save_local(INDEX_DIR)
print(f"✅ Векторный индекс сохранён в: {INDEX_DIR}")
print(f"🎉 Готово! Теперь можно использовать RAG с улучшенной точностью.")