import pickle
import numpy as np
from faiss import normalize_L2
from embedding_model import encode

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
    batch_size = 16  # или меньшее число
    embeddings_list = []  # используем список для сбора батчей эмбеддингов
    for i in range(0, len(product_names), batch_size):
        batch = product_names[i:i + batch_size]
        batch_embeddings = encode(batch)  # предполагается, что возвращается NumPy-массив размерности (batch_size, d)
        embeddings_list.append(batch_embeddings)

    embeddings = np.vstack(embeddings_list)  # или np.concatenate(embeddings_list, axis=0)

    normalize_L2(embeddings)

    np.save(EMBEDDINGS_FILE, embeddings)
    with open(METADATA_FILE, 'wb') as f:
        pickle.dump(products, f)
    metadata = products
    print("Эмбеддинги и метаданные сохранены.")
    return embeddings, metadata
