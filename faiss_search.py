# Можно попробовать другой индекс, чтобы не применять нормализацию
import numpy as np
import pickle
import os
from faiss import normalize_L2 # Нужна ли нормализация??
import faiss
from embedding_model import encode


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
    embeddings = encode(product_names)

    normalize_L2(embeddings)

    np.save(EMBEDDINGS_FILE, embeddings)
    with open(METADATA_FILE, 'wb') as f:
        pickle.dump(products, f)
    metadata = products
    print("Эмбеддинги и метаданные сохранены.")

# Далее вы можете использовать embeddings для построения FAISS-индекса:

INDEX_FILE = 'faiss_index.index'

# Предполагаем, что у вас уже есть эмбеддинги (либо вычисленные, либо загруженные из файла)
# Если вы хотите их загрузить, используйте подход из предыдущего примера.
# Здесь мы предполагаем, что переменная embeddings уже существует и эмбеддинги нормализованы.
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
query = "яблоко"
query_embedding = encode([query])
faiss.normalize_L2(query_embedding)
k = 3
distances, indices = index.search(query_embedding, k)

print("Результаты поиска:")
for rank, idx in enumerate(indices[0]):
    product = metadata[idx]  # metadata загружены или сохранены ранее
    print(f"\nРезультат {rank + 1}:")
    print(f"Название: {product['name']}")
    print(f"Описание: {product['description']}")
    print(f"Пищевая ценность: {product['nutrition']}")
    print(f"Схожесть: {distances[0][rank]:.4f}")



