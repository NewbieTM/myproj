import pickle
import numpy as np
from faiss import normalize_L2
from embedding_model import encode
import os
import faiss

EMBEDDINGS_FILE = 'db_and_weights/embeddings.npy'
METADATA_FILE = 'db_and_weights/products_metadata.pkl'


def load_db():
    products = []
    with open('db_and_weights/products.txt', 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) >= 2:
                products.append({
                    'name': parts[0],
                    'nutrition': parts[1]
                })
    return products


def save_db(products):
    product_names = [p['name'] for p in products]
    batch_size = 16
    embeddings_list = []
    for i in range(0, len(product_names), batch_size):
        batch = product_names[i:i + batch_size]
        batch_embeddings = encode(batch)  # предполагается, что возвращается NumPy-массив размерности (batch_size, d)
        embeddings_list.append(batch_embeddings)

    embeddings = np.vstack(embeddings_list)

    normalize_L2(embeddings)

    np.save(EMBEDDINGS_FILE, embeddings)
    with open(METADATA_FILE, 'wb') as f:
        pickle.dump(products, f)
    metadata = products
    print("Эмбеддинги и метаданные сохранены.")
    return embeddings, metadata


def load_embeddings():
    # Embeddings and metadata load
    if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(METADATA_FILE):
        embeddings = np.load(EMBEDDINGS_FILE)
        with open(METADATA_FILE, 'rb') as f:
            metadata = pickle.load(f)
        print("Эмбеддинги и метаданные загружены из файлов.")
    else:
        products = load_db()
        embeddings, metadata = save_db(products)

    return embeddings, metadata

def load_index():
    # FAISS Index load
    INDEX_FILE = 'db_and_weights/faiss_index.index'

    embeddings, metadata = load_embeddings()

    d = embeddings.shape[1]

    if os.path.exists(INDEX_FILE):
        index = faiss.read_index(INDEX_FILE)
        print("FAISS индекс загружен из файла.")
    else:
        index = faiss.IndexFlatIP(d)
        index.add(embeddings)
        faiss.write_index(index, INDEX_FILE)
        print("FAISS индекс создан и сохранён в файл.")

    return embeddings, metadata, index