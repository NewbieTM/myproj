# Можно попробовать другой индекс, чтобы не применять нормализацию
import numpy as np
import pickle
import os
from faiss import normalize_L2 # Нужна ли нормализация??
import faiss
from embedding_model import encode


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


EMBEDDINGS_FILE = 'embeddings.npy'
METADATA_FILE = 'products_metadata.pkl'


products = []
with open('products.txt', 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('|')
        if len(parts) >= 2:
            products.append({
                'name': parts[0],
                'nutrition': parts[1]
            })


if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(METADATA_FILE):
    embeddings = np.load(EMBEDDINGS_FILE)
    with open(METADATA_FILE, 'rb') as f:
        metadata = pickle.load(f)
    print("Эмбеддинги и метаданные загружены из файлов.")
else:
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


INDEX_FILE = 'faiss_index.index'


d = embeddings.shape[1]

if os.path.exists(INDEX_FILE):
    index = faiss.read_index(INDEX_FILE)
    print("FAISS индекс загружен из файла.")
else:
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    faiss.write_index(index, INDEX_FILE)
    print("FAISS индекс создан и сохранён в файл.")

# Поиск аналогичным образом:
query = "что можно приготовить из говядины?"
query_embedding = encode([query])
faiss.normalize_L2(query_embedding)
k = 3
distances, indices = index.search(query_embedding, k)

print("Результаты поиска:")
for rank, idx in enumerate(indices[0]):
    product = metadata[idx]  # metadata загружены или сохранены ранее
    print(f"\nРезультат {rank + 1}:")
    print(f"Название: {product['name']}")
    print(f"Пищевая ценность: {product['nutrition']}")
    print(f"Схожесть: {distances[0][rank]:.4f}")



