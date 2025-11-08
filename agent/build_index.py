import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm

# === Пути и параметры ===
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PERSIST_DIR = ROOT / "chroma_db"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_texts(data_dir: Path):
    """Загружает все текстовые документы из data/"""
    docs, ids, metas = [], [], []
    for f in data_dir.glob("*.txt"):
        text = f.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        docs.append(text)
        ids.append(f.name)
        metas.append({"source": f.name})
    return ids, docs, metas


def main():
    if not DATA_DIR.exists():
        raise SystemExit("Не найдена папка data/. Сначала запусти prepare_docs.py")

    ids, docs, metas = load_texts(DATA_DIR)
    print(f"Загружено документов: {len(docs)}")

    # === Модель для эмбеддингов ===
    print(f"Загрузка модели эмбеддингов: {MODEL_NAME}")
    embedder = SentenceTransformer(MODEL_NAME)

    # === Инициализация клиента Chroma ===
    client = chromadb.PersistentClient(path=str(PERSIST_DIR))

    # Проверяем/создаем коллекцию
    try:
        collection = client.get_collection("docchat")
    except Exception:
        collection = client.create_collection("docchat")

    # === Удаляем старые данные (если есть) ===
    try:
        existing = collection.get()
        if existing and existing.get("ids"):
            collection.delete(ids=existing["ids"])
            print(f"Старые данные ({len(existing['ids'])}) удалены.")
        else:
            print("Коллекция пуста — создаём заново.")
    except Exception as e:
        print("Ошибка при очистке коллекции (возможно, она пуста):", e)

    # === Добавляем документы ===
    print("Добавляем документы в Chroma...")
    batch_size = 32
    for i in tqdm(range(0, len(docs), batch_size)):
        batch_docs = docs[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]
        batch_meta = metas[i : i + batch_size]
        embs = embedder.encode(
            batch_docs, convert_to_numpy=True, show_progress_bar=False
        )
        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_meta,
            embeddings=embs.tolist(),
        )

    print(f"Индекс успешно создан и сохранён в: {PERSIST_DIR}")


if __name__ == "__main__":
    main()
